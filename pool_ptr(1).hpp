#pragma once

// =============================================================================
// pool_ptr.hpp  —  Stage 4: RAII wrapper with custom deleter
// =============================================================================
//
// What this file provides
// -----------------------
// PoolDeleter<T,N,Policy>   Custom deleter that routes deallocation back to
//                           a FixedPoolManager instead of the global heap.
//
// PoolPtr<T,N,Policy>       Type alias: std::unique_ptr<T, PoolDeleter<...>>.
//                           Gives objects automatic, exception-safe lifetime.
//
// make_pool_ptr(pool, args...) Factory analogous to std::make_unique.
//                           Allocates + constructs + wraps in one expression.
//
// adopt_pool_ptr(pool, raw) Takes ownership of a raw T* already returned by
//                           pool.allocate(). Does NOT construct — use when
//                           wrapping a pointer from legacy code.
//
// =============================================================================
// Design rationale
// =============================================================================
//
// WHY custom deleter instead of shared_ptr?
//   shared_ptr allocates a control block on the heap for each object. That
//   defeats the purpose of the pool. unique_ptr with a stateful custom deleter
//   stores the deleter inline — no heap allocation, no atomic refcount, no
//   overhead beyond a raw pointer per smart pointer.
//
// WHY store pool* in the deleter, not a reference?
//   std::unique_ptr requires its deleter to be DefaultConstructible and
//   CopyConstructible (for move semantics). References satisfy neither.
//   A raw (non-owning) pointer satisfies both at zero cost. The lifetime
//   contract (pool outlives all PoolPtrs) is documented, not enforced by
//   the type system — that is the accepted tradeoff here.
//
// WHY does PoolPtr encode N in the type?
//   Because PoolDeleter stores a FixedPoolManager<T,N,Policy>*, and N is a
//   template parameter of FixedPoolManager. This means PoolPtr<Bullet,256>
//   and PoolPtr<Bullet,512> are different types and cannot share a container.
//   This is a fundamental limitation of compile-time pool sizing.
//   Mitigation: if you need a uniform PoolPtr<Bullet> regardless of pool
//   size, type-erase the deleter with std::function or a virtual base class
//   (Stage 6 material). The cost is one extra indirection per deallocation.
//
// WHY is make_pool_ptr exception-safe?
//   pool.allocate() either returns a valid T* or throws (strong guarantee,
//   see pool_manager.hpp). If it throws, no PoolPtr is ever created, so
//   the deleter is never called with a garbage pointer.
//   If it returns successfully, the PoolPtr is constructed in the return
//   statement. unique_ptr's constructor is noexcept when given a raw pointer
//   and a compatible deleter, so there is no window between "allocate
//   succeeded" and "PoolPtr owns the pointer" where an exception could escape
//   and leak the block.
//
// LIFETIME CONTRACT (not enforceable by the type system):
//   The pool MUST outlive every PoolPtr it creates. Violating this produces
//   a dangling pool* in the deleter — calling it is undefined behaviour.
//   Enforce this by making the pool a member of the same object or a global
//   with static storage duration, not a local variable that could be destroyed
//   while PoolPtrs are still live.
//
// =============================================================================
// C++ concepts used
// =============================================================================
//
// std::unique_ptr<T, Deleter>
//   The standard exclusive-ownership smart pointer. Its destructor calls
//   Deleter::operator()(T*) when the managed pointer is non-null. This is
//   the only mechanism we exploit — everything else is the pool's concern.
//
// Custom deleter (stateful)
//   A deleter is any callable that accepts a T* and is DefaultConstructible,
//   CopyConstructible, and (ideally) noexcept in operator(). Our PoolDeleter
//   stores one pointer (the pool back-reference) so sizeof(PoolDeleter) ==
//   sizeof(void*). unique_ptr stores the deleter inline via EBO when the
//   deleter is empty, but our deleter is not empty — it is the minimum
//   non-empty stateful deleter possible.
//
// std::forward<Args>(args)...
//   Perfect forwarding in make_pool_ptr passes constructor arguments to
//   pool.allocate() without copying. If you pass an rvalue (e.g. a temporary
//   std::string), it is moved, not copied. If you pass an lvalue, it is
//   passed by reference, not copied.
//
// [[nodiscard]]
//   Applied to make_pool_ptr and adopt_pool_ptr. Discarding the return value
//   leaks the allocated block permanently — the compiler warning catches this.
//
// =============================================================================

#include <memory>
#include <type_traits>
#include "pool_manager.hpp"

#ifdef POOL_DEBUG
#  include <cstdio>   // std::fprintf
#  include <cstdlib>  // std::abort
#endif

namespace fp {

// =============================================================================
// PoolDeleter<T, N, SyncPolicy>
// =============================================================================
//
// Custom deleter for std::unique_ptr<T, PoolDeleter<T,N,SyncPolicy>>.
//
// operator()(T* ptr):
//   Calls pool->deallocate(ptr), which in turn calls ptr->~T() explicitly
//   and pushes the block back onto the free list. This is correct because:
//   - placement new was used to construct the object (pool.allocate does this)
//   - the memory belongs to the pool's arena, not the global heap
//   - explicit destructor call + pool return mirrors the construction exactly
//
// Thread safety:
//   operator() is noexcept. If SyncPolicy is MutexSync, pool->deallocate
//   acquires the pool's mutex internally. The PoolDeleter itself adds no
//   extra synchronisation.
//
// Size:
//   sizeof(PoolDeleter) == sizeof(void*). One pointer. No heap overhead.
//   unique_ptr<T, PoolDeleter> is therefore sizeof(T*) + sizeof(void*) ==
//   two pointers (16 bytes on 64-bit). This is the minimum possible cost
//   for a stateful deleter.
// =============================================================================
template<
    typename    T,
    std::size_t N,
    typename    SyncPolicy = NoSync
>
struct PoolDeleter {
    using pool_type = FixedPoolManager<T, N, SyncPolicy>;

    // -------------------------------------------------------------------------
    // pool — raw non-owning back-pointer to the allocating pool.
    //
    // Default value is nullptr so that:
    //   (a) PoolDeleter is DefaultConstructible (required by unique_ptr)
    //   (b) A default-constructed or moved-from PoolPtr has a no-op deleter
    //
    // A moved-from unique_ptr sets its internal pointer to nullptr, so
    // operator() is never called on a nullptr T* in normal use.
    // Defensive null check on pool_ protects against programmer error.
    // -------------------------------------------------------------------------
    pool_type* pool_ = nullptr;

    PoolDeleter() noexcept = default;

    explicit PoolDeleter(pool_type* p) noexcept : pool_{p} {}

    // Allow conversion from PoolDeleter<U,N,P> where U* is convertible to T*.
    // This enables PoolPtr<Base,...> to be constructed from PoolPtr<Derived,...>
    // if Derived derives from Base — same semantic as unique_ptr's converting
    // constructor. The pool type must match exactly (same N, same Policy).
    template<typename U,
             typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    explicit PoolDeleter(const PoolDeleter<U, N, SyncPolicy>& other) noexcept
        : pool_{reinterpret_cast<pool_type*>(other.pool_)}
    {}

    // -------------------------------------------------------------------------
    // operator()(ptr) — called by unique_ptr's destructor / reset
    //
    // Preconditions (programmer responsibility):
    //   1. pool_ is not nullptr when ptr is not nullptr (see note below)
    //   2. ptr was returned by pool_->allocate() and has not been deallocated
    //   3. pool_ is still alive (lifetime contract — not checkable here)
    //
    // Null-pool + non-null ptr is a programming error, not a valid state.
    // How it can happen:
    //   PoolPtr<Bullet, 8> p;               // default-constructed: ptr=null, pool_=null — OK
    //   Bullet* raw = pool.allocate();
    //   p.reset(raw);                       // ptr=raw, pool_=null — BUG: silent leak
    //
    // In the above case ptr is non-null but pool_ is null, so the block is
    // never returned to the pool. In release mode this silently leaks.
    // In debug mode (POOL_DEBUG) we abort immediately with a clear message.
    //
    // The safe pattern for retroactive wrapping is adopt_pool_ptr(pool, raw).
    //
    // noexcept: pool_->deallocate() is noexcept. T's destructor must be
    //   noexcept (enforced by static_assert in FixedPoolManager). Therefore
    //   this entire call chain is noexcept. unique_ptr's destructor requires
    //   the deleter to be noexcept — this satisfies that requirement.
    // -------------------------------------------------------------------------
    void operator()(T* ptr) const noexcept {
#ifdef POOL_DEBUG
        // ptr is non-null but pool_ is null: the deleter was never given a
        // pool to return this block to. This block will be permanently lost.
        // Catching this here (at deallocation time) is the earliest point
        // where we can detect the mistake — it is better than a silent leak.
        if (ptr != nullptr && pool_ == nullptr) {
            std::fprintf(stderr,
                "[PoolDeleter] FATAL: operator() called with a non-null ptr (%p) "
                "but pool_ is null.\n"
                "  The block will never be returned to any pool — permanent leak.\n"
                "  Cause: manual p.reset(raw) without a pool back-pointer, or\n"
                "         a default-constructed PoolDeleter given a live pointer.\n"
                "  Fix:   use adopt_pool_ptr(pool, raw) to wrap existing pointers.\n",
                static_cast<void*>(ptr));
            std::abort();
        }
#endif
        if (pool_ != nullptr && ptr != nullptr) {
            pool_->deallocate(ptr);
        }
    }
};

// =============================================================================
// PoolPtr<T, N, SyncPolicy>
// =============================================================================
//
// Type alias. Equivalent to std::unique_ptr<T, PoolDeleter<T,N,SyncPolicy>>.
//
// Ownership semantics:
//   - Exclusive ownership, identical to std::unique_ptr.
//   - Move-only: PoolPtr can be moved (transfers ownership) but not copied.
//   - When a PoolPtr is destroyed or reset, the object's destructor is called
//     and the block is returned to the pool via PoolDeleter::operator().
//
// Use like std::unique_ptr:
//   auto b = make_pool_ptr(pool, args...);  // create
//   b->damage = 50.f;                       // access
//   pass_ownership(std::move(b));           // transfer
//   b.reset();                              // explicit early release
//   // or just let b go out of scope        // implicit release
//
// Limitation — N is part of the type:
//   PoolPtr<Bullet, 256> and PoolPtr<Bullet, 512> are DIFFERENT TYPES.
//   You cannot store both in a std::vector<PoolPtr<Bullet, ?>>.
//   Workaround: use the same pool size everywhere for a given type,
//   or type-erase (Stage 6).
// =============================================================================
template<
    typename    T,
    std::size_t N,
    typename    SyncPolicy = NoSync
>
using PoolPtr = std::unique_ptr<T, PoolDeleter<T, N, SyncPolicy>>;


// =============================================================================
// make_pool_ptr(pool, args...)
// =============================================================================
//
// Factory function. The intended way to create pool-managed objects.
// Analogous to std::make_unique.
//
// What it does:
//   1. Calls pool.allocate(args...), which:
//      a. Pops a free block from the pool's free list.
//      b. Calls placement new T(args...) on the block.
//      c. Returns T* pointing to the live object.
//   2. Constructs a PoolPtr wrapping that T* with a deleter that points
//      back to pool.
//   3. Returns the PoolPtr (NRVO / move — no copy).
//
// Template parameter deduction:
//   T, N, and SyncPolicy are all deduced from the pool argument type.
//   The caller writes:
//     auto b = make_pool_ptr(my_bullet_pool, pos, vel, damage);
//   and the compiler deduces T=Bullet, N=1024, SyncPolicy=NoSync
//   from the pool type FixedPoolManager<Bullet,1024,NoSync>.
//   No explicit template parameters needed.
//
// Exception safety: STRONG GUARANTEE.
//   Case 1: pool is exhausted — allocate() throws std::bad_alloc.
//     No block was consumed. No PoolPtr is created. Exception propagates.
//     Pool state is unchanged.
//
//   Case 2: T's constructor throws.
//     pool.allocate() catches this internally, pushes the block back onto
//     the free list, and rethrows. No block is consumed. No PoolPtr is
//     created. Exception propagates. Pool state is unchanged.
//     (This guarantee comes from pool_manager.hpp, not from here.)
//
//   Case 3: PoolPtr construction after allocate() succeeds.
//     std::unique_ptr's constructor from (T*, Deleter) is noexcept when
//     Deleter's copy constructor is noexcept (ours is — it copies one ptr).
//     Therefore there is NO WINDOW between "allocate succeeded" and "PoolPtr
//     holds the pointer". The block cannot leak here.
//
// Return value:
//   [[nodiscard]] — discarding the return value leaks the block permanently.
//   The compiler will warn. Do not suppress this warning.
// =============================================================================
template<
    typename    T,
    std::size_t N,
    typename    SyncPolicy,
    typename... Args
>
[[nodiscard]] PoolPtr<T, N, SyncPolicy>
make_pool_ptr(FixedPoolManager<T, N, SyncPolicy>& pool, Args&&... args)
{
    // allocate() does: pop free block + placement new T(args...).
    // On success: returns live T*. On failure: throws, pool unchanged.
    T* raw = pool.allocate(std::forward<Args>(args)...);

    // unique_ptr construction is noexcept here (see exception safety note).
    // There is no gap between raw being valid and the PoolPtr owning it.
    return PoolPtr<T, N, SyncPolicy>{
        raw,
        PoolDeleter<T, N, SyncPolicy>{&pool}
    };
}


// =============================================================================
// adopt_pool_ptr(pool, raw_ptr)
// =============================================================================
//
// Takes ownership of a raw T* already returned by pool.allocate().
// Does NOT construct a new object. Use this when:
//   - You have a T* from legacy code that called pool.allocate() manually.
//   - You need to wrap it in a PoolPtr retroactively.
//   - The object already exists in the pool slot.
//
// Preconditions (programmer responsibility):
//   1. raw_ptr was returned by pool.allocate() and the object is live.
//   2. No other PoolPtr or raw-pointer owner exists for this block.
//   3. pool is alive and will outlive the returned PoolPtr.
//
// DO NOT call this with a pointer that:
//   - Was not returned by pool.allocate() (owns() will catch this in debug).
//   - Has already been wrapped in another PoolPtr (double-free).
//   - Was obtained from a different pool instance.
//
// Exception safety: noexcept.
//   No construction occurs. PoolPtr's constructor from (T*, Deleter) is
//   noexcept. This function cannot throw.
//
// Example:
//   // Legacy API that returns a raw pool pointer:
//   Bullet* b = fire_bullet(pool, pos, vel);  // calls pool.allocate() internally
//   // Wrap it safely:
//   auto managed_b = adopt_pool_ptr(pool, b);
//   // Now managed_b owns b. When managed_b is destroyed, ~Bullet() is called
//   // and the block is returned to pool. b must not be used after this point.
// =============================================================================
template<
    typename    T,
    std::size_t N,
    typename    SyncPolicy
>
[[nodiscard]] PoolPtr<T, N, SyncPolicy>
adopt_pool_ptr(FixedPoolManager<T, N, SyncPolicy>& pool, T* raw_ptr) noexcept
{
    // In debug mode, pool.owns() validates that raw_ptr actually belongs
    // to this pool. If it doesn't, we abort rather than corrupt the free list.
    // In release mode, this check is compiled out — the precondition is on
    // the caller.
    //
    // We cannot call pool.owns() here because it is not public-const-correct
    // without the pointer — but POOL_DEBUG calls it inside deallocate() anyway,
    // so the check still fires at deallocation time in debug builds.
    return PoolPtr<T, N, SyncPolicy>{
        raw_ptr,
        PoolDeleter<T, N, SyncPolicy>{&pool}
    };
}

} // namespace fp

// =============================================================================
// Stage 4 code review
// =============================================================================
//
// CORRECTNESS
//   - PoolDeleter::operator() guards against two distinct null cases:
//
//     Case A — pool_ == nullptr, ptr == nullptr:
//       Legitimate. Happens when a PoolPtr is default-constructed or has
//       been moved from. unique_ptr sets its internal pointer to nullptr
//       before calling the deleter on move, so this path is a safe no-op.
//
//     Case B — pool_ == nullptr, ptr != nullptr:
//       Programming error. The deleter has no pool to return the block to.
//       In POOL_DEBUG builds: std::abort() with a clear diagnostic.
//       In release builds: silent leak (unchanged — same as before the fix,
//       but the debug path catches it during development).
//       Prevention: always use make_pool_ptr() or adopt_pool_ptr(), never
//       call p.reset(raw) on a PoolPtr without first setting its deleter.
//
//     Case C — pool_ != nullptr, ptr != nullptr:
//       Normal deallocation path. pool_->deallocate(ptr) is called.
//
//   - The converting constructor (PoolDeleter<U> → PoolDeleter<T>) uses
//     reinterpret_cast on the pool pointer. This is safe only because both
//     FixedPoolManager<T,N,P> and FixedPoolManager<U,N,P> have the same
//     memory layout for their first member (arena_). In practice this
//     conversion is only used for base/derived relationships where the pool
//     manages derived objects. Marked "Stage 6 risk" — do not use without
//     review.
//
// EXCEPTION SAFETY
//   - make_pool_ptr: strong guarantee (analysed above).
//   - adopt_pool_ptr: noexcept.
//   - PoolDeleter::operator(): noexcept (propagates from deallocate).
//
// PERFORMANCE
//   - sizeof(PoolDeleter) == sizeof(void*) == 8 bytes on 64-bit.
//   - sizeof(PoolPtr) == sizeof(T*) + sizeof(PoolDeleter) == 16 bytes.
//   - No heap allocation. No atomic operations. No virtual dispatch.
//   - The deallocation call through the deleter is one level of indirection:
//     deleter.operator()(ptr) → pool_->deallocate(ptr). This is one function
//     call + one pointer-swap. The compiler inlines operator() at -O2.
//
// UNDEFINED BEHAVIOUR RISKS
//   - If pool_ outlives the PoolPtr but pool_ is not the pool that allocated
//     the object: pool->deallocate() writes the block back to the wrong free
//     list. This corrupts both pools silently in release mode. In debug mode,
//     owns() inside deallocate() catches it.
//   - The converting constructor uses reinterpret_cast. Risk noted above.
//
// FUTURE IMPROVEMENTS
//   - Type-erase N from PoolPtr so PoolPtr<Bullet> works for any pool size.
//     Approach: store a void(*)(void*, void*) function pointer as the deleter,
//     capturing pool and the correct deallocate() call at make_pool_ptr time.
//     Cost: one extra pointer per PoolPtr + no inlining of the deleter.
//   - Add a weak_pool_ptr (non-owning observer) that can check if the block
//     is still allocated via pool.owns() + the debug bitmap.
// =============================================================================
