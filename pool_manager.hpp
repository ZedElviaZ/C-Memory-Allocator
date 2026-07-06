#pragma once

// -----------------------------------------------------------------------------
// pool_manager.hpp
//
// FixedPoolManager<T, N, SyncPolicy>
//
// A fixed-size memory pool allocator backed by a single contiguous heap arena.
//
// Design decisions made (see design_decisions.md for full analysis):
//   1. Free list strategy : Intrusive singly-linked list.
//                           Zero overhead. O(1) alloc and dealloc.
//                           Debug bitmap compiled in under POOL_DEBUG.
//   2. Arena ownership    : Heap-allocated via ::operator new with align_val_t.
//                           Compile-time N encodes capacity in the type.
//   3. Thread safety      : Policy template parameter (NoSync / MutexSync).
//   4. Alignment          : std::align_val_t passed to ::operator new.
//                           Handles over-aligned types (SIMD, etc.) correctly.
//
// Constraints:
//   sizeof(T) >= sizeof(void*)   (static_assert enforced)
//   N >= 1                       (static_assert enforced)
//
// Usage example:
//   FixedPoolManager<Bullet, 1024> pool;
//   Bullet* b = pool.allocate(pos, vel);   // placement new, O(1)
//   pool.deallocate(b);                    // ~Bullet(), push free slot, O(1)
//
// Exception safety:
//   allocate() : strong guarantee. If placement new throws, the block is
//                returned to the free list before the exception propagates.
//   deallocate(): noexcept. Destructor must not throw (standard requirement).
//   Constructor : strong guarantee. On failure, already-built state is torn
//                 down correctly by the destructor delegation.
//
// Undefined behaviour risks (explicitly identified):
//   - Calling deallocate() with a pointer not owned by this pool.
//     Mitigation: owns(ptr) check available; enabled by default in debug.
//   - Calling deallocate() twice on the same pointer (double-free).
//     Mitigation: debug bitmap detects this in O(1).
//   - Using a T* after deallocate() (use-after-free).
//     Mitigation: POOL_DEBUG fills freed memory with 0xCD.
// -----------------------------------------------------------------------------

#include <cstddef>       // std::size_t, std::byte
#include <cstring>       // std::memset
#include <memory>        // std::unique_ptr
#include <new>           // ::operator new, std::align_val_t, std::bad_alloc
#include <stdexcept>     // std::logic_error (debug mode)
#include <type_traits>   // std::is_nothrow_destructible_v
#include <utility>       // std::forward

#ifdef POOL_DEBUG
#  include <bitset>
#  include <cassert>
#  include <cstdio>      // std::fprintf (debug diagnostics)
#endif

#include "thread_policy.hpp"

namespace fp {

// ---------------------------------------------------------------------------
// Implementation detail: FreeNode
//
// When a block is free, its first sizeof(void*) bytes are reinterpreted as
// a FreeNode. This is the intrusive part: no separate allocation, no separate
// memory. The object storage IS the free list node storage.
//
// This is well-defined in C++ because:
//   1. The memory was allocated by ::operator new — it has no declared type.
//   2. We access it only through std::byte* and then via reinterpret_cast
//      to FreeNode*, which has a scalar member. The effective type rule
//      (C++17 [basic.types]/10) permits this as long as we do not violate
//      the alignment requirement.
//   3. alignof(FreeNode) == alignof(void*). The static_assert below ensures
//      the block address satisfies this.
// ---------------------------------------------------------------------------
namespace detail {

struct FreeNode {
    FreeNode* next;
};

// ---------------------------------------------------------------------------
// round_up(value, alignment)
//
// Rounds 'value' up to the nearest multiple of 'alignment'.
// 'alignment' must be a power of two (enforced by ::operator new / alignof).
//
// Example: round_up(9, 8) == 16, round_up(8, 8) == 8, round_up(1, 4) == 4.
//
// This is required for kBlockSize so that block[i+1] starts at an address
// that satisfies alignof(T) AND alignof(FreeNode). Without this rounding,
// a type like:
//   struct Tiny9 { char data[9]; };  // sizeof=9, alignof=1 on most ABIs
// would place block 1 at arena+9 — not 8-byte aligned, so the FreeNode*
// stored there would be a misaligned pointer write. UBSan catches this.
// ---------------------------------------------------------------------------
inline constexpr std::size_t round_up(
    std::size_t value,
    std::size_t alignment
) noexcept {
    return (value + alignment - 1u) / alignment * alignment;
}

// Compute raw block size: must hold both T and a FreeNode pointer.
// If sizeof(T) < sizeof(void*), the intrusive pointer would overflow T's
// storage. The static_assert in the pool class body catches this at compile
// time, but the computation here must still be correct for the template
// to instantiate cleanly.
template<typename T>
inline constexpr std::size_t block_raw_size_v =
    (sizeof(T) >= sizeof(FreeNode)) ? sizeof(T) : sizeof(FreeNode);

// Compute block alignment: must satisfy both T and FreeNode.
template<typename T>
inline constexpr std::size_t block_align_v =
    (alignof(T) >= alignof(FreeNode)) ? alignof(T) : alignof(FreeNode);

// Compute the padded block size: round raw size UP to alignment boundary.
//
// This ensures every block starts at a correctly aligned address.
// Without this, a type with sizeof=9 and alignof=8 would place:
//   block[0] at arena+0   (aligned)
//   block[1] at arena+9   (NOT 8-byte aligned → UB for FreeNode* and T*)
//   block[2] at arena+18  (NOT 8-byte aligned)
// With round_up(9, 8) = 16:
//   block[0] at arena+0   (aligned)
//   block[1] at arena+16  (aligned)
//   block[2] at arena+32  (aligned)
template<typename T>
inline constexpr std::size_t block_size_v =
    round_up(block_raw_size_v<T>, block_align_v<T>);

} // namespace detail


// ---------------------------------------------------------------------------
// FixedPoolManager
// ---------------------------------------------------------------------------
template<
    typename    T,
    std::size_t N,
    typename    SyncPolicy = NoSync
>
class FixedPoolManager : private SyncPolicy {

    // -----------------------------------------------------------------------
    // Compile-time invariants
    // -----------------------------------------------------------------------
    static_assert(N >= 1,
        "FixedPoolManager: N must be at least 1.");

    static_assert(sizeof(T) >= sizeof(void*),
        "FixedPoolManager: sizeof(T) must be >= sizeof(void*). "
        "The intrusive free list stores a pointer inside each free block. "
        "Use an external index list for types smaller than a pointer.");

    static_assert(std::is_nothrow_destructible_v<T>,
        "FixedPoolManager: T's destructor must be noexcept. "
        "deallocate() is noexcept and calls ptr->~T(). "
        "A throwing destructor would terminate the program.");

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    static constexpr std::size_t kBlockSize  = detail::block_size_v<T>;
    static constexpr std::size_t kBlockAlign = detail::block_align_v<T>;
    static constexpr std::size_t kArenaSize  = kBlockSize * N;

public:
    // -----------------------------------------------------------------------
    // Public type aliases
    // -----------------------------------------------------------------------
    using value_type = T;
    using size_type  = std::size_t;

    static constexpr size_type capacity() noexcept { return N; }

    // -----------------------------------------------------------------------
    // Constructor
    //
    // Allocates the arena with proper alignment, then walks every block to
    // build the initial free list. All N blocks start on the free list.
    //
    // Why walk the arena at construction rather than lazily?
    //   Lazy initialisation (maintain a "high-water mark" bump pointer until
    //   the first free-then-reuse) is a valid optimisation for large N where
    //   only a fraction of blocks will ever be used. It would be Stage 6
    //   material. For now, eager initialisation keeps the invariants simple:
    //   after construction, the free list is always the complete source of
    //   truth about which blocks are available.
    // -----------------------------------------------------------------------
    FixedPoolManager()
        : arena_{
            static_cast<std::byte*>(
                ::operator new(kArenaSize,
                    std::align_val_t{kBlockAlign})
            )
          }
        , free_head_{nullptr}
        , in_use_{0}
    {
        // Build the free list by linking every block to the next.
        // We iterate in reverse so that block 0 ends up at the head,
        // meaning the first allocation returns block 0 — predictable
        // for testing and cache-friendly for sequential use patterns.
        //
        // We use std::memcpy to write the next pointer into raw bytes.
        // This is the standard-compliant way to write a typed value into
        // untyped storage without triggering UBSan's type-tracking checks.
        // The memory from ::operator new has no declared type (it is raw
        // storage), so reinterpret_cast + member write may trigger UBSan's
        // "member access within untyped storage" diagnostic on some toolchains.
        // memcpy avoids this entirely and produces identical machine code.
        for (std::size_t i = N; i-- > 0; ) {
            std::byte* block_ptr = arena_ + i * kBlockSize;
            detail::FreeNode* next_val = free_head_;
            std::memcpy(block_ptr, &next_val, sizeof(detail::FreeNode*));
            free_head_ = reinterpret_cast<detail::FreeNode*>(block_ptr);
        }

#ifdef POOL_DEBUG
        // In debug mode, poison the entire arena so accidental reads of
        // uninitialised or freed memory produce 0xCD values, not plausible
        // stale data. This is the same pattern used by MSVC's debug heap.
        std::memset(arena_, 0xCD, kArenaSize);

        // Re-build the free list after poisoning.
        // The memset wrote 0xCD over every byte including the next pointers
        // we wrote in the first pass. Use std::memcpy to write the pointer
        // value — this is the standard-compliant way to write through raw
        // bytes without UBSan's type-aliasing checks triggering on the
        // 0xCD-filled memory.
        free_head_ = nullptr;
        for (std::size_t i = N; i-- > 0; ) {
            std::byte* block_ptr = arena_ + i * kBlockSize;
            detail::FreeNode* next_val = free_head_;
            std::memcpy(block_ptr, &next_val, sizeof(detail::FreeNode*));
            free_head_ = reinterpret_cast<detail::FreeNode*>(block_ptr);
        }
        // allocated_ bitset is default-constructed to all zeros (all free).
#endif
    }

    // -----------------------------------------------------------------------
    // Destructor
    //
    // Precondition: all allocated objects must have been deallocated before
    // the pool is destroyed. We do NOT destroy live objects here — that would
    // require tracking which blocks are live (feasible with the debug bitmap)
    // but would silently mask lifetime bugs in client code.
    //
    // In debug mode, we assert that in_use_ == 0 and print diagnostics.
    // In release mode, the arena is freed regardless — leaked objects'
    // destructors are not called. This is intentional: the pool is the
    // allocator, not the owner. Ownership is the client's responsibility.
    // -----------------------------------------------------------------------
    ~FixedPoolManager() noexcept {
#ifdef POOL_DEBUG
        if (in_use_ != 0) {
            std::fprintf(stderr,
                "[FixedPoolManager] LEAK DETECTED: %zu block(s) still "
                "allocated at pool destruction. Destructors NOT called.\n",
                in_use_);
            // Do not abort here — the arena must still be freed.
        }
#endif
        ::operator delete(
            static_cast<void*>(arena_),
            std::align_val_t{kBlockAlign}
        );
    }

    // -----------------------------------------------------------------------
    // Rule of Five
    //
    // Copy: deleted. An arena cannot be copied — the free list pointers inside
    //   free blocks point into the original arena. Copying the arena bytes
    //   would produce dangling internal pointers.
    //
    // Move: deleted for now (Stage 6 will evaluate this properly).
    //   Move could be correct if we transfer arena ownership and null the
    //   source's arena pointer. However, the free list consists of pointers
    //   INTO the arena, so moving the arena pointer is safe — the free list
    //   nodes move with the arena bytes. This is actually safe to implement.
    //   Deferred to Stage 6 to keep this stage focused and reviewable.
    // -----------------------------------------------------------------------
    FixedPoolManager(const FixedPoolManager&)            = delete;
    FixedPoolManager& operator=(const FixedPoolManager&) = delete;
    FixedPoolManager(FixedPoolManager&&)                 = delete;
    FixedPoolManager& operator=(FixedPoolManager&&)      = delete;

    // -----------------------------------------------------------------------
    // allocate<Args...>(args...)
    //
    // Pops the free list head and constructs T in-place via placement new.
    // Forwards all arguments to T's constructor.
    //
    // Returns: T* pointing to the newly constructed object.
    // Throws : std::bad_alloc if the pool is exhausted.
    //          Any exception thrown by T's constructor (strong guarantee).
    //
    // Exception safety analysis:
    //   1. We pop the block from the free list BEFORE calling placement new.
    //   2. If placement new throws, we catch the exception, push the block
    //      back onto the free list, and rethrow. The pool is left in the
    //      exact state it was in before allocate() was called.
    //   3. This is the strong exception guarantee.
    //
    // Why not pop AFTER placement new?
    //   Placement new must have a valid block before it can construct.
    //   We have no choice — we must pop first.
    // -----------------------------------------------------------------------
    template<typename... Args>
    [[nodiscard]] T* allocate(Args&&... args) {
        typename SyncPolicy::Guard guard{*this};

        if (free_head_ == nullptr) {
            throw std::bad_alloc{};
        }

        // Pop the head block off the free list.
        // Use std::memcpy to read the 'next' pointer out of raw storage.
        // Direct member access (node->next) on memory that has no declared
        // C++ type triggers UBSan's "member access within object of type
        // 'FreeNode' whose dynamic type is [untyped]" on strict toolchains.
        // memcpy reads the bytes without any type assumption — identical
        // machine code, fully defined behaviour.
        detail::FreeNode* node = free_head_;
        detail::FreeNode* next_head = nullptr;
        std::memcpy(&next_head, node, sizeof(detail::FreeNode*));
        free_head_ = next_head;

        // Get the raw block pointer. The FreeNode aliased the first bytes;
        // now we will write a T object there via placement new.
        void* block = static_cast<void*>(node);

        // Attempt in-place construction. If T's constructor throws,
        // we must restore the free list before propagating the exception.
        try {
            T* obj = ::new(block) T(std::forward<Args>(args)...);
            ++in_use_;

#ifdef POOL_DEBUG
            // Mark block as allocated in the bitmap.
            std::size_t idx = block_index(obj);
            allocated_.set(idx);
#endif
            return obj;
        }
        catch (...) {
            // Construction failed. Restore the free list.
            // Use std::memcpy to write the next pointer — same rationale
            // as the pop above: avoids UBSan complaints on untyped storage.
            detail::FreeNode* old_head = free_head_;
            std::memcpy(node, &old_head, sizeof(detail::FreeNode*));
            free_head_ = node;
            throw;  // propagate the original exception
        }
    }

    // -----------------------------------------------------------------------
    // deallocate(ptr)
    //
    // Calls ptr->~T() explicitly, then pushes the block back onto the
    // free list.
    //
    // Preconditions (asserted in debug mode):
    //   1. ptr != nullptr
    //   2. ptr was returned by this pool's allocate()
    //   3. ptr has not already been deallocated (no double-free)
    //
    // noexcept: T's destructor must be noexcept (static_assert above).
    //   If it were not, and it threw, we would have partially-destroyed
    //   object storage on the free list — undefined behaviour.
    // -----------------------------------------------------------------------
    void deallocate(T* ptr) noexcept {
        typename SyncPolicy::Guard guard{*this};

        if (ptr == nullptr) return;  // null dealloc is a no-op, like delete

#ifdef POOL_DEBUG
        // Validate ownership: ptr must point into our arena and be
        // block-aligned. An invalid pointer here is a programming error —
        // we abort rather than corrupt the free list.
        if (!owns(ptr)) {
            std::fprintf(stderr,
                "[FixedPoolManager] FATAL: deallocate(%p) — pointer not "
                "owned by this pool. Arena: [%p, %p).\n",
                static_cast<void*>(ptr),
                static_cast<void*>(arena_),
                static_cast<void*>(arena_ + kArenaSize));
            std::abort();
        }

        std::size_t idx = block_index(ptr);

        // Double-free check: if the bit is already clear, the block is
        // already on the free list. Double-freeing corrupts the free list
        // silently in release mode — in debug mode we abort immediately.
        if (!allocated_.test(idx)) {
            std::fprintf(stderr,
                "[FixedPoolManager] FATAL: double-free detected at %p "
                "(block index %zu).\n",
                static_cast<void*>(ptr), idx);
            std::abort();
        }

        allocated_.reset(idx);
#endif

        // Call the destructor explicitly.
        // This is correct: placement new constructed the object,
        // so we are responsible for its destruction. We must NOT call
        // 'delete ptr' — that would call the destructor AND then try to
        // free 'ptr' via the global allocator, which does not own this memory.
        ptr->~T();

#ifdef POOL_DEBUG
        // Poison the freed block so use-after-free reads 0xCD, not
        // stale data that might look valid.
        std::memset(static_cast<void*>(ptr), 0xCD, kBlockSize);
#endif

        // Reinterpret the raw block as a FreeNode and push onto free list.
        // Use std::memcpy to write the next pointer into the block's raw bytes.
        // This is the same pattern as the constructor: avoids member-write UB
        // on untyped storage that UBSan tracks on strict configurations.
        auto* node = reinterpret_cast<detail::FreeNode*>(ptr);
        detail::FreeNode* old_head = free_head_;
        std::memcpy(node, &old_head, sizeof(detail::FreeNode*));
        free_head_ = node;

        --in_use_;
    }

    // -----------------------------------------------------------------------
    // owns(ptr)
    //
    // Returns true iff ptr points to the start of a block inside this pool's
    // arena. This checks:
    //   1. ptr is within [arena_, arena_ + kArenaSize)
    //   2. (ptr - arena_) is a multiple of kBlockSize
    //
    // This does NOT check whether the block is currently allocated or free.
    // An owns() == true pointer may have been deallocated already.
    //
    // Complexity: O(1).
    // Use case: assertion in deallocate(), and external validation.
    // -----------------------------------------------------------------------
    [[nodiscard]] bool owns(const T* ptr) const noexcept {
        const auto* p    = reinterpret_cast<const std::byte*>(ptr);
        const auto  diff = p - arena_;
        if (diff < 0) return false;
        const auto udiff = static_cast<std::size_t>(diff);
        return udiff < kArenaSize && (udiff % kBlockSize == 0);
    }

    // -----------------------------------------------------------------------
    // Observers
    // -----------------------------------------------------------------------
    [[nodiscard]] std::size_t in_use()    const noexcept { return in_use_; }
    [[nodiscard]] std::size_t available() const noexcept { return N - in_use_; }
    [[nodiscard]] bool        full()      const noexcept { return in_use_ == N; }
    [[nodiscard]] bool        empty()     const noexcept { return in_use_ == 0; }

    // -----------------------------------------------------------------------
    // debug_dump()
    //
    // Prints a human-readable summary of pool state to stderr.
    // Only meaningful in POOL_DEBUG builds.
    // -----------------------------------------------------------------------
    void debug_dump() const noexcept {
#ifdef POOL_DEBUG
        std::fprintf(stderr,
            "[FixedPoolManager] N=%zu  block_size=%zu  align=%zu\n"
            "  in_use=%zu  available=%zu\n"
            "  arena=[%p, %p)\n",
            N, kBlockSize, kBlockAlign,
            in_use_, available(),
            static_cast<void*>(arena_),
            static_cast<void*>(arena_ + kArenaSize));

        std::fprintf(stderr, "  allocated bitmap: ");
        for (std::size_t i = 0; i < N; ++i) {
            std::fputc(allocated_.test(i) ? '1' : '0', stderr);
        }
        std::fputc('\n', stderr);
#else
        std::fprintf(stderr,
            "[FixedPoolManager] debug_dump() requires -DPOOL_DEBUG.\n");
#endif
    }

private:
    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

#ifdef POOL_DEBUG
    // Returns the block index for a pointer known to be in-range.
    // Precondition: owns(ptr) == true.
    std::size_t block_index(const T* ptr) const noexcept {
        const auto* p = reinterpret_cast<const std::byte*>(ptr);
        return static_cast<std::size_t>(p - arena_) / kBlockSize;
    }
#endif

    // -----------------------------------------------------------------------
    // Data members
    //
    // Layout rationale:
    //   arena_      — the hot path only reads free_head_, not the arena ptr.
    //                 Placing arena_ first ensures it does not share a cache
    //                 line with free_head_ for very large arenas. For small
    //                 pools this is irrelevant but costs nothing.
    //   free_head_  — accessed on every allocate/deallocate. Keep it in the
    //                 first cache line of the pool object.
    //   in_use_     — updated every alloc/dealloc. Same cache line as head.
    //
    // With NoSync (empty base), SyncPolicy contributes 0 bytes.
    // With MutexSync, the mutex is in the base subobject.
    // -----------------------------------------------------------------------
    std::byte*          arena_;
    detail::FreeNode*   free_head_;
    std::size_t         in_use_;

#ifdef POOL_DEBUG
    std::bitset<N>      allocated_;
#endif
};

} // namespace fp
