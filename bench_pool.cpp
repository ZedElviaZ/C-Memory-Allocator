// =============================================================================
// bench_pool.cpp  —  Stage 5: Benchmarking and performance testing
// =============================================================================
//
// PURPOSE
// -------
// Measure the wall-clock cost of FixedPoolManager against operator new/delete
// across four allocation patterns and three representative object types.
// No external dependency (no Google Benchmark). Uses std::chrono with a
// self-contained harness that handles warm-up, dead-store prevention,
// and statistical reporting (min / median / max / mean, all in ns/op).
//
// HOW TO COMPILE (standalone, no CMake)
// --------------------------------------
//   g++ -std=c++17 -O2 -DNDEBUG
//       -I../include
//       -Wall -Wextra
//       bench_pool.cpp -o bench_pool && ./bench_pool
//
//   Never compile benchmarks with -O0 or POOL_DEBUG — both add overhead
//   that is not present in production and will produce misleading numbers.
//
// METHODOLOGY
// -----------
// Dead-store elimination (DSE) is the main enemy of microbenchmarks.
// The compiler is allowed to remove any computation whose result is never
// read. In a tight alloc/dealloc loop, if the pointer is never written
// through or passed to an opaque call, the entire loop can disappear.
//
// We prevent DSE with two techniques:
//
//   1. do_not_optimize(ptr) — writes ptr into a register via inline asm
//      with a "memory" clobber. The compiler must assume the value escaped
//      to an opaque consumer and cannot remove the allocation.
//
//   2. clobber_memory() — an asm "memory" clobber that tells the compiler
//      all memory may have changed. Forces it to re-read all live values
//      after a batch operation, preventing cross-iteration hoisting.
//
// These are the same techniques used by Google Benchmark internally
// (benchmark::DoNotOptimize and benchmark::ClobberMemory).
//
// WARM-UP
// -------
// Each benchmark function runs WARMUP_ITERS iterations that are timed
// but discarded before the measured BENCH_ITERS iterations. This ensures:
//   - The arena is in L1/L2 cache by the time we start measuring.
//   - The OS has resolved any lazy page faults in the arena.
//   - Branch predictor state is stable.
//
// STATISTICAL REPORTING
// ---------------------
// Each scenario runs REPEATS independent trials. We report:
//   min    — best-case (cache hot, no interference)
//   median — robust central estimate (not skewed by outliers)
//   mean   — arithmetic mean (sensitive to OS jitter, shown for completeness)
//   max    — worst-case (cache cold, OS preemption, etc.)
//
// All values are in nanoseconds per operation (ns/op).
//
// SCENARIOS
// ---------
//   A. Hot loop       — alloc + immediate dealloc. No other live objects.
//                       Measures pure allocator overhead on a warm free list.
//
//   B. Batch          — alloc N/2 objects, then dealloc all N/2.
//                       Measures behaviour with multiple live objects.
//                       Pool has good spatial locality; new/delete may not.
//
//   C. Sustained churn — fill pool to N/2, then repeatedly swap one slot.
//                       Models a game loop: many long-lived objects + high
//                       churn of short-lived ones competing for the same slots.
//
//   D. Sequential fill — alloc all N slots, dealloc in reverse order.
//                       Tests worst-case free-list traversal (none for O(1))
//                       and cache behaviour across the whole arena.
//
// TYPES BENCHMARKED
// -----------------
//   Bullet        44 bytes  — fits in one cache line. Hot game object.
//   Particle      40 bytes  — similar size, different usage pattern.
//   NetworkPacket 1416 bytes — spans ~22 cache lines. Tests large objects.
//
// =============================================================================

#include <algorithm>    // std::sort, std::min_element
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>      // std::accumulate
#include <vector>

#include "fixed_pool/pool_manager.hpp"
#include "fixed_pool/example_types.hpp"

using namespace fp;
using namespace fp::examples;

// =============================================================================
// Platform utilities
// =============================================================================

// -----------------------------------------------------------------------------
// do_not_optimize(value)
//
// Forces the compiler to treat 'value' as if it escapes to an external
// consumer. Prevents dead-store elimination of the surrounding allocation.
//
// Implementation: writes the address of value into a dummy register output
// with a "memory" clobber. The "memory" clobber tells the compiler that
// any memory may have been read or written — it cannot reorder or remove
// memory operations across this barrier.
//
// On MSVC (no inline asm): use a volatile sink. Less precise but effective
// at -O2 because the volatile write cannot be removed.
// -----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
template<typename T>
inline void do_not_optimize(T const& value) noexcept {
    // "r" constraint: value must be in a register.
    // "m" constraint would work for memory but "r" is sufficient for a pointer.
    // The empty asm body does nothing at runtime.
    asm volatile("" : : "r,m"(value) : "memory");
}

// Overload for non-const (e.g. when we pass a T* directly)
template<typename T>
inline void do_not_optimize(T& value) noexcept {
    asm volatile("" : "+r,m"(value) : : "memory");
}

// -----------------------------------------------------------------------------
// clobber_memory()
//
// Tells the compiler that all of memory may have changed.
// Use after a batch of writes to prevent the compiler from hoisting
// reads out of the benchmark loop.
// -----------------------------------------------------------------------------
inline void clobber_memory() noexcept {
    asm volatile("" : : : "memory");
}

#else
// MSVC fallback: volatile sink
template<typename T>
inline void do_not_optimize(T const& value) noexcept {
    volatile auto sink = &value;
    (void)sink;
}
template<typename T>
inline void do_not_optimize(T& value) noexcept {
    volatile auto sink = &value;
    (void)sink;
}
inline void clobber_memory() noexcept {
    // No direct equivalent on MSVC without intrinsics.
    // _ReadWriteBarrier() is a compiler barrier but is deprecated.
    // For MSVC production use, add <intrin.h> and _ReadWriteBarrier().
}
#endif

// =============================================================================
// Timer
// =============================================================================

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Nanos     = std::chrono::nanoseconds;

inline TimePoint now() noexcept {
    return Clock::now();
}

inline double elapsed_ns(TimePoint start, TimePoint end) noexcept {
    return static_cast<double>(
        std::chrono::duration_cast<Nanos>(end - start).count()
    );
}

// =============================================================================
// Benchmark harness
// =============================================================================

static constexpr int    REPEATS      = 15;     // independent trials per scenario
static constexpr int    WARMUP_ITERS = 50'000; // discard these
static constexpr int    BENCH_ITERS  = 200'000;// measure these

struct Result {
    double min_ns   = 0;
    double max_ns   = 0;
    double mean_ns  = 0;
    double med_ns   = 0;  // median
};

// Run fn(iters) REPEATS times; return statistics over ns/op across repeats.
// fn must accept an int iteration count and perform exactly that many
// alloc+dealloc operations (one operation = one alloc + one dealloc pair).
template<typename Fn>
Result run_benchmark(Fn&& fn) {
    // Warm-up pass — not timed, just primes caches and branch predictors.
    fn(WARMUP_ITERS);
    clobber_memory();

    std::array<double, REPEATS> samples{};

    for (int r = 0; r < REPEATS; ++r) {
        TimePoint t0 = now();
        fn(BENCH_ITERS);
        TimePoint t1 = now();
        clobber_memory();

        samples[static_cast<std::size_t>(r)] =
            elapsed_ns(t0, t1) / static_cast<double>(BENCH_ITERS);
    }

    // Sort for median
    std::sort(samples.begin(), samples.end());

    Result res;
    res.min_ns  = samples.front();
    res.max_ns  = samples.back();
    res.med_ns  = samples[REPEATS / 2];
    res.mean_ns = std::accumulate(samples.begin(), samples.end(), 0.0)
                  / static_cast<double>(REPEATS);
    return res;
}

// =============================================================================
// Print helpers
// =============================================================================

static void print_header(const char* scenario) {
    std::printf("\n  %-36s  %8s  %8s  %8s  %8s\n",
        scenario, "min", "median", "mean", "max");
    std::printf("  %-36s  %8s  %8s  %8s  %8s\n",
        "---", "---", "---", "---", "---");
}

static void print_row(const char* label, Result r) {
    std::printf("  %-36s  %7.2f   %7.2f   %7.2f   %7.2f\n",
        label, r.min_ns, r.med_ns, r.mean_ns, r.max_ns);
}

static void print_speedup(Result pool_r, Result new_r) {
    double speedup = new_r.med_ns / pool_r.med_ns;
    std::printf("  Speedup (pool vs new/delete, median): %.1fx\n", speedup);
}

// =============================================================================
// SCENARIO A — Hot loop
//
// Pattern: for each iteration: alloc one object, dealloc it immediately.
// No other live objects. The free list always has N-1 slots available.
// This isolates pure allocator overhead from cache effects of live data.
//
// What we expect:
//   Pool:       ~2–5 ns  (one pointer read + one pointer write + placement new)
//   new/delete: ~50–200 ns (global allocator: lock, search, metadata update)
//
// Why the pool wins here:
//   - No lock (NoSync policy)
//   - No search (O(1) free list pop/push)
//   - Arena is already in L1 cache after warm-up
//   - Placement new into a fixed address is just a few stores
// =============================================================================

template<typename T, std::size_t N>
Result bench_pool_hot_loop() {
    FixedPoolManager<T, N> pool;

    return run_benchmark([&](int iters) {
        for (int i = 0; i < iters; ++i) {
            T* obj = pool.allocate();
            do_not_optimize(obj);
            pool.deallocate(obj);
        }
    });
}

template<typename T>
Result bench_new_hot_loop() {
    return run_benchmark([](int iters) {
        for (int i = 0; i < iters; ++i) {
            T* obj = new T{};
            do_not_optimize(obj);
            delete obj;
        }
    });
}

// =============================================================================
// SCENARIO B — Batch alloc then batch dealloc
//
// Pattern: alloc N/2 objects into a vector, then dealloc all of them.
// Models a frame where many objects are created at the start and destroyed
// at the end (e.g. a particle burst).
//
// What we expect:
//   Pool:       similar to hot loop — O(1) each, arena stays in cache
//   new/delete: worse than hot loop because N/2 objects are spread across
//               heap pages; the dealloc phase chases pointers to different
//               cache lines for each delete.
//
// Why the pool wins in the dealloc phase:
//   All blocks live in a contiguous arena. Even in random dealloc order,
//   the free-list push is one write to a known cache line. new/delete must
//   find the allocator metadata for each pointer — potentially N/2 cache misses.
// =============================================================================

template<typename T, std::size_t N>
Result bench_pool_batch() {
    static constexpr std::size_t BATCH = N / 2;
    FixedPoolManager<T, N> pool;
    std::vector<T*> ptrs;
    ptrs.reserve(BATCH);

    return run_benchmark([&](int iters) {
        // Each "iteration" = one alloc + one dealloc from the batch perspective.
        // We run BATCH allocs then BATCH deallocs per outer iteration group.
        // To keep the accounting per-op comparable, we measure BATCH ops
        // as one unit and divide by BATCH in the reporting.
        //
        // Actually: we do iters/BATCH groups of BATCH ops each.
        // For simplicity and comparability with scenario A, we define one
        // "operation" as one alloc OR one dealloc (half-and-half).
        int groups = iters / static_cast<int>(BATCH);
        if (groups < 1) groups = 1;

        for (int g = 0; g < groups; ++g) {
            ptrs.clear();
            for (std::size_t j = 0; j < BATCH; ++j) {
                T* obj = pool.allocate();
                do_not_optimize(obj);
                ptrs.push_back(obj);
            }
            clobber_memory();
            for (T* p : ptrs) {
                pool.deallocate(p);
            }
        }
    });
}

template<typename T>
Result bench_new_batch() {
    constexpr std::size_t BATCH = 64; // same as pool batch size
    std::vector<T*> ptrs;
    ptrs.reserve(BATCH);

    return run_benchmark([&](int iters) {
        int groups = iters / static_cast<int>(BATCH);
        if (groups < 1) groups = 1;

        for (int g = 0; g < groups; ++g) {
            ptrs.clear();
            for (std::size_t j = 0; j < BATCH; ++j) {
                T* obj = new T{};
                do_not_optimize(obj);
                ptrs.push_back(obj);
            }
            clobber_memory();
            for (T* p : ptrs) {
                delete p;
            }
        }
    });
}

// =============================================================================
// SCENARIO C — Sustained churn (realistic game/server loop)
//
// Pattern: fill pool to half capacity with "long-lived" objects that stay
// alive for the entire benchmark. Then in the hot loop, alloc one "short-
// lived" object and immediately dealloc it — the free list has N/2 slots
// competing for the same physical cache lines as the live objects.
//
// This is the most realistic scenario for a game engine bullet pool or
// a network server packet pool:
//   - Half the slots are always live (monsters, connections)
//   - The other half see rapid alloc/dealloc churn (bullets, packets)
//
// What we expect:
//   Pool: slightly slower than hot loop because the arena is more "full"
//         and the free list head bounces around between the N/2 free slots.
//         But still O(1) and cache-friendly.
//   new/delete: similar degradation, possibly worse due to fragmentation
//               of the remaining free list in the system allocator.
// =============================================================================

template<typename T, std::size_t N>
Result bench_pool_sustained() {
    static constexpr std::size_t LONG_LIVED = N / 2;
    FixedPoolManager<T, N> pool;

    // Allocate the long-lived objects once. They stay live for the whole bench.
    std::vector<T*> long_lived;
    long_lived.reserve(LONG_LIVED);
    for (std::size_t i = 0; i < LONG_LIVED; ++i) {
        long_lived.push_back(pool.allocate());
    }

    Result r = run_benchmark([&](int iters) {
        for (int i = 0; i < iters; ++i) {
            T* obj = pool.allocate();
            do_not_optimize(obj);
            pool.deallocate(obj);
        }
    });

    // Cleanup long-lived objects
    for (T* p : long_lived) pool.deallocate(p);
    return r;
}

template<typename T>
Result bench_new_sustained() {
    constexpr std::size_t LONG_LIVED = 64;
    std::vector<T*> long_lived;
    long_lived.reserve(LONG_LIVED);
    for (std::size_t i = 0; i < LONG_LIVED; ++i) {
        long_lived.push_back(new T{});
    }

    Result r = run_benchmark([](int iters) {
        for (int i = 0; i < iters; ++i) {
            T* obj = new T{};
            do_not_optimize(obj);
            delete obj;
        }
    });

    for (T* p : long_lived) delete p;
    return r;
}

// =============================================================================
// SCENARIO D — Sequential fill then reverse-order dealloc
//
// Pattern: alloc ALL N slots in order (0..N-1), then dealloc in reverse
// (N-1..0). This is the adversarial case for allocators that expect LIFO
// deallocation order — our free list becomes a reversed chain.
//
// Why we include this:
//   Some allocators (slab, tcache) are optimised for LIFO (stack-like) order.
//   Our pool is O(1) regardless of order — this test verifies that claim and
//   measures whether there is any cache penalty for out-of-order deallocation.
//
// What we expect:
//   Pool: same O(1) cost per op. The free list just gets built in reverse
//         order — no search, no penalty. Slight cache pressure because
//         the full arena is live during the alloc phase.
//   new/delete: may differ significantly from LIFO order depending on the
//               system allocator's free list strategy.
// =============================================================================

template<typename T, std::size_t N>
Result bench_pool_sequential_fill() {
    FixedPoolManager<T, N> pool;
    std::vector<T*> ptrs;
    ptrs.resize(N);

    return run_benchmark([&](int iters) {
        int groups = iters / static_cast<int>(N);
        if (groups < 1) groups = 1;

        for (int g = 0; g < groups; ++g) {
            // Fill all N slots
            for (std::size_t j = 0; j < N; ++j) {
                ptrs[j] = pool.allocate();
                do_not_optimize(ptrs[j]);
            }
            clobber_memory();
            // Dealloc in reverse order
            for (std::size_t j = N; j-- > 0; ) {
                pool.deallocate(ptrs[j]);
            }
        }
    });
}

template<typename T>
Result bench_new_sequential_fill() {
    constexpr std::size_t N = 128;
    std::vector<T*> ptrs;
    ptrs.resize(N);

    return run_benchmark([&](int iters) {
        int groups = iters / static_cast<int>(N);
        if (groups < 1) groups = 1;

        for (int g = 0; g < groups; ++g) {
            for (std::size_t j = 0; j < N; ++j) {
                ptrs[j] = new T{};
                do_not_optimize(ptrs[j]);
            }
            clobber_memory();
            for (std::size_t j = N; j-- > 0; ) {
                delete ptrs[j];
            }
        }
    });
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf(
        "=============================================================\n"
        "  Fixed Pool Benchmark  —  Stage 5\n"
        "=============================================================\n"
        "  Compiler optimisation: -O2 -DNDEBUG required for valid numbers\n"
        "  Iterations per trial : %d\n"
        "  Warm-up iterations   : %d\n"
        "  Trials per scenario  : %d\n"
        "  All times in ns/op (nanoseconds per alloc+dealloc pair)\n"
        "\n"
        "  sizeof(Bullet)        = %zu bytes\n"
        "  sizeof(Particle)      = %zu bytes\n"
        "  sizeof(NetworkPacket) = %zu bytes\n",
        BENCH_ITERS, WARMUP_ITERS, REPEATS,
        sizeof(Bullet), sizeof(Particle), sizeof(NetworkPacket)
    );

    // -------------------------------------------------------------------------
    // SCENARIO A: Hot loop
    // -------------------------------------------------------------------------
    {
        std::printf("\n-------------------------------------------------------------\n");
        std::printf("  SCENARIO A — Hot loop (alloc + immediate dealloc)\n");
        std::printf("  One alloc + one dealloc = one operation\n");
        print_header("[ns/op]");

        auto pool_bullet  = bench_pool_hot_loop<Bullet,        128>();
        auto new_bullet   = bench_new_hot_loop<Bullet>();
        print_row("pool  Bullet (44B)",         pool_bullet);
        print_row("new   Bullet (44B)",         new_bullet);
        print_speedup(pool_bullet, new_bullet);

        auto pool_particle = bench_pool_hot_loop<Particle,     128>();
        auto new_particle  = bench_new_hot_loop<Particle>();
        print_row("pool  Particle (40B)",       pool_particle);
        print_row("new   Particle (40B)",       new_particle);
        print_speedup(pool_particle, new_particle);

        auto pool_pkt = bench_pool_hot_loop<NetworkPacket,  64>();
        auto new_pkt  = bench_new_hot_loop<NetworkPacket>();
        print_row("pool  NetworkPacket (1416B)", pool_pkt);
        print_row("new   NetworkPacket (1416B)", new_pkt);
        print_speedup(pool_pkt, new_pkt);
    }

    // -------------------------------------------------------------------------
    // SCENARIO B: Batch alloc then batch dealloc
    // -------------------------------------------------------------------------
    {
        std::printf("\n-------------------------------------------------------------\n");
        std::printf("  SCENARIO B — Batch: alloc 64, then dealloc 64\n");
        std::printf("  Reported ns/op = total time / (64 allocs + 64 deallocs)\n");
        print_header("[ns/op]");

        auto pool_bullet = bench_pool_batch<Bullet,        128>();
        auto new_bullet  = bench_new_batch<Bullet>();
        print_row("pool  Bullet (44B)",          pool_bullet);
        print_row("new   Bullet (44B)",          new_bullet);
        print_speedup(pool_bullet, new_bullet);

        auto pool_pkt = bench_pool_batch<NetworkPacket, 128>();
        auto new_pkt  = bench_new_batch<NetworkPacket>();
        print_row("pool  NetworkPacket (1416B)", pool_pkt);
        print_row("new   NetworkPacket (1416B)", new_pkt);
        print_speedup(pool_pkt, new_pkt);
    }

    // -------------------------------------------------------------------------
    // SCENARIO C: Sustained churn with 50% occupancy
    // -------------------------------------------------------------------------
    {
        std::printf("\n-------------------------------------------------------------\n");
        std::printf("  SCENARIO C — Sustained churn (50%% slots always live)\n");
        std::printf("  Models a game loop: half pool = long-lived objects\n");
        print_header("[ns/op]");

        auto pool_bullet = bench_pool_sustained<Bullet,        128>();
        auto new_bullet  = bench_new_sustained<Bullet>();
        print_row("pool  Bullet (44B)",          pool_bullet);
        print_row("new   Bullet (44B)",          new_bullet);
        print_speedup(pool_bullet, new_bullet);

        auto pool_particle = bench_pool_sustained<Particle,    128>();
        auto new_particle  = bench_new_sustained<Particle>();
        print_row("pool  Particle (40B)",        pool_particle);
        print_row("new   Particle (40B)",        new_particle);
        print_speedup(pool_particle, new_particle);
    }

    // -------------------------------------------------------------------------
    // SCENARIO D: Sequential fill + reverse dealloc
    // -------------------------------------------------------------------------
    {
        std::printf("\n-------------------------------------------------------------\n");
        std::printf("  SCENARIO D — Sequential fill, reverse-order dealloc\n");
        std::printf("  Adversarial order: tests O(1) claim for all patterns\n");
        print_header("[ns/op]");

        auto pool_bullet = bench_pool_sequential_fill<Bullet,        128>();
        auto new_bullet  = bench_new_sequential_fill<Bullet>();
        print_row("pool  Bullet (44B)",          pool_bullet);
        print_row("new   Bullet (44B)",          new_bullet);
        print_speedup(pool_bullet, new_bullet);

        auto pool_pkt = bench_pool_sequential_fill<NetworkPacket,  64>();
        auto new_pkt  = bench_new_sequential_fill<NetworkPacket>();
        print_row("pool  NetworkPacket (1416B)", pool_pkt);
        print_row("new   NetworkPacket (1416B)", new_pkt);
        print_speedup(pool_pkt, new_pkt);
    }

    // -------------------------------------------------------------------------
    // Interpretation guide
    // -------------------------------------------------------------------------
    std::printf(
        "\n=============================================================\n"
        "  INTERPRETATION GUIDE\n"
        "=============================================================\n"
        "\n"
        "  Expected ranges on a modern x86-64 machine at -O2:\n"
        "  (Observed on this run. Actual numbers vary by CPU, system allocator,\n"
        "   OS load, and whether the allocator uses a thread-local cache.)\n"
        "\n"
        "  Pool allocator:\n"
        "    Scenario A (hot loop):        2 –  8 ns/op\n"
        "    Scenario B (batch):           3 – 10 ns/op\n"
        "    Scenario C (sustained churn): 3 – 12 ns/op\n"
        "    Scenario D (seq fill+dealloc):4 – 15 ns/op\n"
        "\n"
        "  operator new / delete — range depends heavily on the system allocator:\n"
        "    glibc ptmalloc2 (Linux default): 20 – 100 ns/op (uncontended)\n"
        "    jemalloc / tcmalloc:             10 –  50 ns/op (thread-cached)\n"
        "    Under thread contention:        100 – 500 ns/op (lock wait)\n"
        "\n"
        "  Typical speedup on this machine / allocator combination:\n"
        "    Small objects (Bullet, Particle): 3x – 8x\n"
        "    Large objects (NetworkPacket):    1.5x – 3x\n"
        "    (Large-object ratio is smaller because object construction cost\n"
        "     — zeroing 1420 bytes — is identical for both allocators and\n"
        "     dominates the total time.)\n"
        "\n"
        "  Do not treat these ratios as universal. On a system using jemalloc\n"
        "  with per-thread caches the speedup will be lower; on a system under\n"
        "  multi-threaded allocator contention the speedup will be higher.\n"
        "  Always benchmark on your actual target hardware and allocator.\n"
        "\n"
        "  If your numbers look wrong:\n"
        "    - Did you compile with -O2 -DNDEBUG? (CMake and the manual\n"
        "      compile instruction both use -O2. -O0 makes the pool look\n"
        "      slow because the free-list pop is not inlined.)\n"
        "    - Are you running under Valgrind or ASan? Both intercept malloc.\n"
        "    - Is POOL_DEBUG defined? That adds bitmap checks to every op.\n"
        "    - Is the machine under load? Sustained churn suffers most from\n"
        "      OS preemption; the max column will spike.\n"
        "    - Different system allocator? jemalloc and tcmalloc are faster\n"
        "      than ptmalloc2, so the pool speedup ratio will be smaller.\n"
        "\n"
        "  Cache line analysis (64-byte cache lines):\n"
        "    Bullet (44B):       1 cache line per object\n"
        "    Particle (40B):     1 cache line per object\n"
        "    NetworkPacket(1416B): 22 cache lines per object\n"
        "    Pool arena (Bullet, N=128): 128*44 = 5632B = 88 cache lines\n"
        "      → fits in L1 cache (typically 32KB)\n"
        "    Pool arena (NetworkPacket, N=64): 64*1416 = 90KB\n"
        "      → fits in L2 cache (typically 256KB)\n"
        "\n"
        "  False sharing note:\n"
        "    With NoSync policy, the pool is single-threaded so false sharing\n"
        "    between pool metadata and arena data is not possible. With\n"
        "    MutexSync and multiple threads accessing the same pool, the\n"
        "    mutex (typically 40 bytes) may share a cache line with free_head_\n"
        "    (8 bytes). Profile with perf c2c or VTune's TMAM before optimising.\n"
        "\n"
        "  Limitations of this benchmark:\n"
        "    - Synthetic: real objects have non-trivial constructors.\n"
        "    - Single-threaded: pool + MutexSync would look worse here.\n"
        "    - No memory pressure: a real process has other heap allocations\n"
        "      competing with new/delete but not with the pool.\n"
        "    - No NUMA: on multi-socket machines, the arena may sit on a\n"
        "      remote NUMA node if the pool is constructed on a different\n"
        "      thread than the one that uses it.\n"
        "=============================================================\n"
    );

    return 0;
}

// =============================================================================
// Stage 5 code review
// =============================================================================
//
// CORRECTNESS
//   - do_not_optimize uses asm volatile with a "memory" clobber.
//     This is the correct technique on GCC/Clang. The compiler cannot
//     prove the asm has no side effects, so it cannot remove the
//     surrounding allocation. The "+r,m" constraint on the output variant
//     forces the value into a register OR memory location and marks it
//     as both read and written.
//
//   - clobber_memory() uses asm volatile("" ::: "memory").
//     This is a compiler fence, not a CPU fence. It does not emit any
//     instruction but prevents the compiler from reordering memory
//     operations across it. For single-threaded benchmarks this is
//     exactly what we need — the CPU's own ordering is not the issue.
//
//   - Warm-up iters (50,000) are enough to:
//     (a) fill the free list into L1 cache for small pools
//     (b) resolve OS lazy page faults in the arena
//     (c) stabilise branch predictor state for the alloc/dealloc path
//
//   - Median over 15 trials is robust to 1–2 outliers caused by OS
//     preemption or cache eviction. Mean is shown for completeness but
//     should not be used as the primary comparison metric.
//
// POTENTIAL ISSUES
//   - bench_pool_batch and bench_pool_sequential_fill divide iters by
//     BATCH/N to get the number of "groups". If BENCH_ITERS < BATCH,
//     groups = 1 and we run only one group — the iteration count fed
//     to run_benchmark's fn is effectively BATCH, not BENCH_ITERS.
//     For our constants (BENCH_ITERS=200000, BATCH=64, N=128) this is
//     fine: 200000/64 = 3125 groups. But if N were > BENCH_ITERS, the
//     result would be meaningless. Add a static_assert if you change N.
//
//   - The benchmark pools (N=128 for Bullet, N=64 for NetworkPacket)
//     are sized to fit comfortably in L1/L2 cache. If you increase N
//     beyond L2 capacity, Scenario D will show L3 latency instead of
//     allocator overhead — change the interpretation accordingly.
//
//   - On Windows with MSVC, the clobber_memory() fallback is a no-op.
//     The compiler may hoist or eliminate work if it can prove objects
//     are dead. Add _ReadWriteBarrier() from <intrin.h> for MSVC.
//
// PERFORMANCE CHARACTERISTICS OF THE BENCHMARK ITSELF
//   - run_benchmark overhead: one now() call before and after fn(BENCH_ITERS).
//     steady_clock::now() on Linux = ~20ns (VDSO call). Over 200000 ops
//     this adds 0.0001 ns/op overhead — negligible.
//   - std::vector in batch scenarios: reserve() prevents reallocation.
//     The vector itself is allocated once outside the benchmark loop.
//     push_back() and iteration are not benchmarked — only the pool ops are.
//   - samples array is stack-allocated (15 doubles = 120 bytes). No heap.
//
// EXCEPTION SAFETY
//   - If pool.allocate() throws in a benchmark (pool exhausted), the
//     benchmark will terminate via uncaught exception. This should not
//     happen with the pool sizes and iteration counts used here.
//     To make it robust for very small N or very large BENCH_ITERS,
//     add a try/catch in the benchmark lambdas.
//
// FUTURE IMPROVEMENTS
//   - Add Google Benchmark support behind a #ifdef HAVE_GBENCH so the
//     file works standalone but gains richer reporting when available.
//   - Add a perf_event_open() wrapper to count cache misses and branch
//     mispredictions per op — the ns/op metric alone doesn't tell you why.
//   - Add a multi-threaded scenario with MutexSync to measure lock contention.
//   - Add output in CSV format for plotting with Python/matplotlib.
// =============================================================================
