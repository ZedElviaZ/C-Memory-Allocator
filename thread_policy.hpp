#pragma once

// -----------------------------------------------------------------------------
// thread_policy.hpp
//
// Policy types that control synchronisation behaviour of FixedPoolManager.
// Passed as the third template parameter.
//
// Design rationale:
//   Using policy-based design (Alexandrescu) rather than a runtime flag or
//   preprocessor toggle. This means:
//     - NoSync  compiles to zero overhead (empty base optimisation applies).
//     - MutexSync adds exactly one std::mutex, locked around alloc/dealloc.
//   The client chooses at the type level. No hidden runtime cost.
//
// Usage:
//   FixedPoolManager<Bullet, 1024, NoSync>    pool;  // single-threaded
//   FixedPoolManager<Bullet, 1024, MutexSync> pool;  // multi-threaded
// -----------------------------------------------------------------------------

#include <mutex>

namespace fp {

// ---------------------------------------------------------------------------
// NoSync
// Zero-overhead policy for single-threaded use.
// All methods are no-ops. The compiler eliminates them entirely.
// ---------------------------------------------------------------------------
struct NoSync {
protected:
    void lock()   const noexcept {}
    void unlock() const noexcept {}

    // RAII guard that does nothing. Used internally by the pool.
    struct Guard {
        explicit Guard(const NoSync&) noexcept {}
    };
};

// ---------------------------------------------------------------------------
// MutexSync
// Protects allocate() and deallocate() with a std::mutex.
// Suitable for pools shared across threads with moderate contention.
//
// Limitation: Under high contention, the mutex becomes a bottleneck.
// For high-contention multi-producer pools, a per-thread pool shard or
// a lock-free CAS list (Stage 6) is more appropriate.
// ---------------------------------------------------------------------------
struct MutexSync {
protected:
    mutable std::mutex mtx_;

    void lock()   const { mtx_.lock(); }
    void unlock() const { mtx_.unlock(); }

    struct Guard {
        const MutexSync& policy_;
        explicit Guard(const MutexSync& p) : policy_{p} { policy_.lock(); }
        ~Guard() { policy_.unlock(); }

        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
    };
};

} // namespace fp
