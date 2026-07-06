// -----------------------------------------------------------------------------
// test_pool.cpp
//
// Tests for FixedPoolManager and PoolPtr.
// Written as standalone tests (no framework dependency for portability).
// Each test is a function returning bool. A test harness runs all and reports.
//
// To compile and run (debug build — enables POOL_DEBUG):
//
//   g++ -std=c++17 -DPOOL_DEBUG \
//       -I../include \
//       -Wall -Wextra -Wpedantic \
//       -fsanitize=address,undefined \
//       -g \
//       test_pool.cpp -o test_pool && ./test_pool
//
// To compile release (no debug checks, measures raw performance):
//
//   g++ -std=c++17 \
//       -I../include \
//       -O2 -DNDEBUG \
//       test_pool.cpp -o test_pool_release && ./test_pool_release
// -----------------------------------------------------------------------------

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "fixed_pool/pool_manager.hpp"
#include "fixed_pool/pool_ptr.hpp"
#include "fixed_pool/example_types.hpp"

using namespace fp;
using namespace fp::examples;

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::fprintf(stderr, "  FAIL: %s  (line %d)\n", #expr, __LINE__); \
            return false; \
        } \
    } while (0)

#define CHECK_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::fprintf(stderr, "  FAIL: %s == %s  (%zu != %zu)  (line %d)\n", \
                #a, #b, (std::size_t)(a), (std::size_t)(b), __LINE__); \
            return false; \
        } \
    } while (0)

static void run_test(const char* name, bool(*fn)()) {
    std::fprintf(stdout, "  %-55s ", name);
    if (fn()) {
        std::fprintf(stdout, "PASS\n");
        ++g_pass;
    } else {
        std::fprintf(stdout, "FAIL\n");
        ++g_fail;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Basic construction and initial state
// ---------------------------------------------------------------------------
static bool test_construction() {
    FixedPoolManager<Bullet, 8> pool;
    CHECK_EQ(pool.capacity(),  8u);
    CHECK_EQ(pool.in_use(),    0u);
    CHECK_EQ(pool.available(), 8u);
    CHECK(pool.empty());
    CHECK(!pool.full());
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Single allocate and deallocate
// ---------------------------------------------------------------------------
static bool test_single_alloc_dealloc() {
    FixedPoolManager<Bullet, 8> pool;

    Bullet* b = pool.allocate(1.f, 2.f, 3.f,  0.f, 0.f, 1.f,  10.f, 42u);

    CHECK(b != nullptr);
    CHECK_EQ(pool.in_use(),    1u);
    CHECK_EQ(pool.available(), 7u);
    CHECK_EQ(b->x,   1.f);
    CHECK_EQ(b->y,   2.f);
    CHECK_EQ(b->z,   3.f);
    CHECK_EQ(b->damage, 10.f);
    CHECK_EQ(b->owner_id, 42u);

    pool.deallocate(b);

    CHECK_EQ(pool.in_use(),    0u);
    CHECK_EQ(pool.available(), 8u);
    CHECK(pool.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: Fill the pool to capacity
// ---------------------------------------------------------------------------
static bool test_fill_to_capacity() {
    constexpr std::size_t N = 16;
    FixedPoolManager<Bullet, N> pool;

    std::vector<Bullet*> ptrs;
    ptrs.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        Bullet* b = pool.allocate();
        CHECK(b != nullptr);
        ptrs.push_back(b);
    }

    CHECK(pool.full());
    CHECK_EQ(pool.in_use(), N);
    CHECK_EQ(pool.available(), 0u);

    // Allocating when full must throw std::bad_alloc
    bool threw = false;
    try { (void)pool.allocate(); }
    catch (const std::bad_alloc&) { threw = true; }
    CHECK(threw);

    // Deallocate all
    for (auto* p : ptrs) pool.deallocate(p);

    CHECK(pool.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Block reuse — deallocate then reallocate
// ---------------------------------------------------------------------------
static bool test_block_reuse() {
    FixedPoolManager<Bullet, 4> pool;

    Bullet* b0 = pool.allocate();
    Bullet* b1 = pool.allocate();
    CHECK_EQ(pool.in_use(), 2u);

    void* addr_b1 = static_cast<void*>(b1);
    pool.deallocate(b1);
    CHECK_EQ(pool.in_use(), 1u);

    // The next allocation should reuse b1's block (LIFO free list)
    Bullet* b2 = pool.allocate();
    CHECK(static_cast<void*>(b2) == addr_b1);

    pool.deallocate(b0);
    pool.deallocate(b2);
    CHECK(pool.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: owns() correctness
// ---------------------------------------------------------------------------
static bool test_owns() {
    FixedPoolManager<Bullet, 8> pool;

    Bullet* b = pool.allocate();
    CHECK(pool.owns(b));

    // A stack-allocated Bullet should not be owned by the pool
    Bullet stack_bullet;
    CHECK(!pool.owns(&stack_bullet));

    // nullptr should not be owned
    CHECK(!pool.owns(nullptr));

    pool.deallocate(b);

    // After deallocation, owns() should still return true (it checks the
    // address range, not the allocation state)
    // Note: b is now dangling — we do NOT dereference it, only check address.
    CHECK(pool.owns(b));

    return true;
}

// ---------------------------------------------------------------------------
// Test 6: All pointers are within the arena and properly aligned
// ---------------------------------------------------------------------------
static bool test_alignment_and_bounds() {
    FixedPoolManager<Bullet, 16> pool;

    std::vector<Bullet*> ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        Bullet* b = pool.allocate();
        // Check alignment: pointer must be aligned to alignof(Bullet)
        CHECK((reinterpret_cast<std::uintptr_t>(b) % alignof(Bullet)) == 0);
        // Check ownership (also validates in-range)
        CHECK(pool.owns(b));
        ptrs.push_back(b);
    }

    // All pointers must be distinct
    for (std::size_t i = 0; i < ptrs.size(); ++i)
        for (std::size_t j = i + 1; j < ptrs.size(); ++j)
            CHECK(ptrs[i] != ptrs[j]);

    for (auto* p : ptrs) pool.deallocate(p);
    return true;
}

// ---------------------------------------------------------------------------
// Test 7: Destructor is called on deallocate
// ---------------------------------------------------------------------------
static bool test_destructor_called() {
    static int destructor_count = 0;
    destructor_count = 0;

    struct Tracked {
        // Use std::size_t to guarantee sizeof(Tracked) >= sizeof(void*).
        // int alone is 4 bytes on most platforms — not enough on 64-bit.
        std::size_t value;
        explicit Tracked(std::size_t v) noexcept : value{v} {}
        ~Tracked() noexcept { ++destructor_count; }
    };

    static_assert(sizeof(Tracked) >= sizeof(void*),
        "Tracked must satisfy pool constraint");

    FixedPoolManager<Tracked, 4> pool;

    auto* t1 = pool.allocate(std::size_t{10});
    auto* t2 = pool.allocate(std::size_t{20});
    CHECK_EQ(destructor_count, 0);

    pool.deallocate(t1);
    CHECK_EQ(destructor_count, 1);

    pool.deallocate(t2);
    CHECK_EQ(destructor_count, 2);

    return true;
}

// ---------------------------------------------------------------------------
// Test 8: Exception safety — constructor throws, block is returned
// ---------------------------------------------------------------------------
static bool test_exception_safety_constructor() {
    static int construct_count = 0;
    construct_count = 0;

    struct ThrowsOnThird {
        ThrowsOnThird() {
            ++construct_count;
            if (construct_count == 3) {
                --construct_count;  // don't count the failed one
                throw std::runtime_error("intentional");
            }
        }
        ~ThrowsOnThird() noexcept = default;
        // Padding to meet sizeof >= sizeof(void*)
        char padding[sizeof(void*)];
    };

    FixedPoolManager<ThrowsOnThird, 4> pool;
    CHECK_EQ(pool.available(), 4u);

    auto* t1 = pool.allocate();
    auto* t2 = pool.allocate();
    CHECK_EQ(pool.in_use(), 2u);

    // Third allocation should throw, block must be returned
    bool threw = false;
    try { (void)pool.allocate(); }
    catch (const std::runtime_error&) { threw = true; }

    CHECK(threw);
    // Pool must be back to 2 in use — the block was returned
    CHECK_EQ(pool.in_use(), 2u);
    CHECK_EQ(pool.available(), 2u);

    pool.deallocate(t1);
    pool.deallocate(t2);
    CHECK(pool.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Test 9: PoolPtr — RAII automatic deallocation
// ---------------------------------------------------------------------------
static bool test_pool_ptr_raii() {
    FixedPoolManager<Bullet, 8> pool;

    {
        auto b = make_pool_ptr(pool, 0.f, 0.f, 0.f,  1.f, 0.f, 0.f,  5.f, 1u);
        CHECK_EQ(pool.in_use(), 1u);
        CHECK(b != nullptr);
        CHECK_EQ(b->damage, 5.f);
    }
    // b went out of scope here; destructor called, block returned
    CHECK_EQ(pool.in_use(), 0u);

    return true;
}

// ---------------------------------------------------------------------------
// Test 10: PoolPtr — exception during scope, block still returned
// ---------------------------------------------------------------------------
static bool test_pool_ptr_exception_safety() {
    FixedPoolManager<Bullet, 8> pool;

    try {
        auto b = make_pool_ptr(pool);
        CHECK_EQ(pool.in_use(), 1u);
        throw std::runtime_error("simulated error");
        // b's destructor runs during stack unwinding
    }
    catch (const std::runtime_error&) {}

    CHECK_EQ(pool.in_use(), 0u);
    return true;
}

// ---------------------------------------------------------------------------
// Test 11: NetworkPacket pool (large object type)
// ---------------------------------------------------------------------------
static bool test_network_packet_pool() {
    FixedPoolManager<NetworkPacket, 32> pool;
    CHECK_EQ(pool.capacity(), 32u);

    auto* p = pool.allocate();
    p->source_ip   = 0xC0A80101;  // 192.168.1.1
    p->dest_ip     = 0xC0A80102;  // 192.168.1.2
    p->source_port = 12345;
    p->payload_len = 100;
    p->payload[0]  = 0xDE;
    p->payload[1]  = 0xAD;

    CHECK_EQ(p->source_ip, 0xC0A80101u);
    CHECK_EQ(p->payload[0], 0xDE);

    pool.deallocate(p);
    CHECK(pool.empty());
    return true;
}

// ---------------------------------------------------------------------------
// Test 12: Null deallocation is a no-op
// ---------------------------------------------------------------------------
static bool test_null_deallocate() {
    FixedPoolManager<Bullet, 4> pool;
    // Must not crash, must not change pool state
    pool.deallocate(nullptr);
    CHECK_EQ(pool.in_use(), 0u);
    return true;
}

// ---------------------------------------------------------------------------
// Test 13: Interleaved alloc/dealloc pattern
// ---------------------------------------------------------------------------
static bool test_interleaved_alloc_dealloc() {
    FixedPoolManager<Particle, 8> pool;

    auto* p0 = pool.allocate(0.f, 0.f, 0.f, 1.f);
    auto* p1 = pool.allocate(1.f, 1.f, 1.f, 2.f);
    auto* p2 = pool.allocate(2.f, 2.f, 2.f, 3.f);

    CHECK_EQ(pool.in_use(), 3u);

    pool.deallocate(p1);  // free middle block
    CHECK_EQ(pool.in_use(), 2u);

    auto* p3 = pool.allocate(3.f, 3.f, 3.f, 4.f);  // should reuse p1's slot
    CHECK_EQ(pool.in_use(), 3u);

    pool.deallocate(p0);
    pool.deallocate(p2);
    pool.deallocate(p3);
    CHECK(pool.empty());

    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::fprintf(stdout,
        "=== FixedPoolManager test suite ===\n"
        "  sizeof(Bullet)        = %zu\n"
        "  sizeof(NetworkPacket) = %zu\n"
        "  sizeof(Particle)      = %zu\n"
        "  sizeof(GameEntity)    = %zu\n"
        "  sizeof(void*)         = %zu\n\n",
        sizeof(Bullet),
        sizeof(NetworkPacket),
        sizeof(Particle),
        sizeof(GameEntity),
        sizeof(void*));

#ifdef POOL_DEBUG
    std::fprintf(stdout, "  Build: DEBUG (POOL_DEBUG defined)\n\n");
#else
    std::fprintf(stdout, "  Build: RELEASE\n\n");
#endif

    run_test("construction + initial state",            test_construction);
    run_test("single allocate + deallocate",            test_single_alloc_dealloc);
    run_test("fill to capacity + bad_alloc",            test_fill_to_capacity);
    run_test("block reuse (LIFO free list)",            test_block_reuse);
    run_test("owns() correctness",                      test_owns);
    run_test("alignment + bounds of all pointers",      test_alignment_and_bounds);
    run_test("destructor called on deallocate",         test_destructor_called);
    run_test("exception safety: constructor throws",    test_exception_safety_constructor);
    run_test("PoolPtr RAII auto-dealloc",               test_pool_ptr_raii);
    run_test("PoolPtr exception safety (unwinding)",    test_pool_ptr_exception_safety);
    run_test("NetworkPacket pool (large object)",       test_network_packet_pool);
    run_test("null deallocate is no-op",                test_null_deallocate);
    run_test("interleaved alloc/dealloc pattern",       test_interleaved_alloc_dealloc);

    std::fprintf(stdout,
        "\n  Results: %d passed, %d failed\n",
        g_pass, g_fail);

    return g_fail == 0 ? 0 : 1;
}
