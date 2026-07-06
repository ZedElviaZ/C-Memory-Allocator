## Fixed-Size Memory Pool Allocator

Deterministic Memory Allocation with 2–6× Faster Performance Than `new/delete`
To explore low-level systems programming and push my understanding of modern C++, I designed and implemented a fixed-size memory pool allocator from scratch. The allocator delivers deterministic O(1) allocation/deallocation, eliminates heap allocations on the hot path after initialization, and benchmarks between **2× and nearly 6× faster** than the standard allocator depending on workload.
Rather than treating this as a coding exercise, I approached it like i was working on a product-level project: prioritizing correctness, exception safety, alignment guarantees, debugging support and measurable performance.

---

## The Problem:

General-purpose allocators are designed for flexibility, not predictability. They can introduce fragmentation, variable allocation latency, and unnecessary overhead when repeatedly allocating large numbers of objects of identical size.
For applications such as game engines, simulation systems, and real-time software, deterministic allocation speed is often more valuable than allocator flexibility. The goal was to build an allocator that could provide consistent performance while remaining safe and ergonomic to use.

---

## Design Decisions

The project uses a fixed-size pool allocator backed by a preallocated block of memory and an intrusive free list. This approach guarantees constant-time allocation and deallocation while avoiding repeated calls to the operating system or the general-purpose heap.

Several design decisions shaped the implementation:
Memory is allocated only once during pool construction, ensuring zero heap allocations during normal operation.
Blocks are aligned correctly to support over-aligned object types.
Construction uses placement new, while exception safety ensures failed constructors immediately return memory to the pool.
A lightweight RAII smart pointer (`PoolPtr`) integrates the allocator naturally into modern C++ code.
Optional debug mode adds allocation tracking and validation without affecting release performance.
Threading behavior is policy-based, allowing either single-threaded or mutex-protected pools without changing user code.

The emphasis throughout was on balancing performance with correctness rather than optimizing for benchmark numbers alone.

---

## Results:

The finished allocator achieved the intended design goals:

| Metric                                | Result                            |
| ------------------------------------- | --------------------------------- |
| Allocation Complexity                 | O(1)                              |
| Heap Allocations After Initialization | 0                                 |
| Exception Safety                      | Strong Guarantee                  |
| Alignment Support                     | Yes                               |
| RAII Integration                      | Custom `PoolPtr`                  |
| Thread Safety                         | Policy-Based                      |
| Benchmark Performance                 | ~2×–5.8× faster than `new/delete` |

Debug and release configurations were validated using AddressSanitizer, UndefinedBehaviorSanitizer, and a dedicated test suite.

---

## One Difficult Problem:

One subtle challenge was supporting correct alignment while maintaining a compact free list.
Simply dividing the memory buffer into equal chunks was insufficient as object alignment requirements can exceed the size of a pointer, misaligned storage would result in undefined behavior for certain types.
The solution was to compute block sizes at compile time using alignment-aware calculations, ensuring every object begins at a correctly aligned address while preserving constant time indexing throughout the allocator.

---

## Tradeoffs:

This allocator intentionally optimizes for predictability over flexibility.
It is ideal when object sizes are known in advance but it is not intended to replace a general-purpose allocator. Each pool manages a single object size, and memory remains reserved for the pool's lifetime, even when objects are freed.
If I were extending the project further, I would explore:
lock-free free lists for high-concurrency workloads,
support for dynamically growing pools,
allocator-aware STL container integration,
and richer runtime diagnostics for release builds.

---

## Technical Overview of this Project:

# Language i used: C++17

# Concepts that i used in this project were:

Custom memory management
Placement new
Intrusive free lists
RAII
Policy-based design
Template metaprogramming
Alignment handling
Exception safety
AddressSanitizer & UndefinedBehaviorSanitizer
Benchmarking and profiling

---

## Representative Implementation:

template<typename T, typename... Args>
PoolPtr<T> make(Args&&... args)
{
void\* block = allocate();

    try
    {
        return PoolPtr<T>(
            new (block) T(std::forward<Args>(args)...),
            this
        );
    }
    catch (...)
    {
        deallocate(block);
        throw;
    }

}

```

The allocator guarantees that if object construction fails, the reserved block is immediately returned to the pool, preserving allocator consistency while maintaining the strong exception guarantee.

---

## Project Context

This project actually at first began as a personal challenge to test whether I could design and implement a low-level memory allocator from first principles.
i used AI as a assistant for discussion, tweaks, review, and refinement, but the project was not generated through vibe coding.
Every design decision, implementation detail, debugging step, and architectural tradeoff that was done in this project was understood, evaluated, and intentionally integrated into the final system.
```
