# Custom Memory Allocator / Data Structure Library

Core data structures rebuilt from scratch in C++17 — no `std::vector`,
`std::list`, `std::map`, or `std::allocator` — to work directly with
pointers, manual dynamic memory (`new`/`delete`, placement `new`), and
templates.

## Contents

| Header | What it is |
|---|---|
| `include/cds/allocator.hpp` | `DefaultAllocator<T>` (thin wrapper over global `operator new`/`delete`) and `PoolAllocator<T>` (fixed-size free-list pool that recycles freed slots) |
| `include/cds/vector.hpp` | `Vector<T>` — dynamic array with manual capacity growth, placement-new construction, Rule of Five |
| `include/cds/linked_list.hpp` | `LinkedList<T>` — doubly linked list, node allocation via a pluggable allocator |
| `include/cds/bst.hpp` | `BST<T>` — unbalanced binary search tree with insert/erase/inorder traversal |
| `include/cds/grid2d.hpp` | `Grid2D<T>` — simple 2D grid backed by one contiguous allocation (row-major) |

All containers are header-only templates and accept an allocator type
(`Vector`, `Grid2D`) or allocator template (`LinkedList`, `BST`, since they
allocate an internal `Node` type rather than `T` directly). `LinkedList` and
`BST` default to `PoolAllocator` since node-based structures churn many
same-sized allocations; `Vector` and `Grid2D` default to `DefaultAllocator`
since they need variably-sized contiguous buffers.

## Build & run

```sh
cmake -S . -B build
cmake --build build
./build/demo         # runs the demo program
ctest --test-dir build --output-on-failure   # or ./build/run_tests
```

Requires a C++17 compiler and CMake 3.14+.

## Design notes

- **Rule of Five everywhere**: every container implements copy ctor/assign,
  move ctor/assign, and destructor explicitly, since each owns raw memory.
- **Placement `new`**: containers allocate raw, uninitialized memory via the
  allocator and construct objects into it with placement `new`, then call
  destructors explicitly before deallocating — the same separation
  `std::allocator_traits` enforces internally.
- **`PoolAllocator<T>`**: pre-allocates chunks of `ChunkSlots` fixed-size
  slots and threads freed slots into an intrusive free list (the freed
  memory itself stores the "next" pointer), so repeated single-object
  alloc/free cycles (e.g. `LinkedList` push/pop, `BST` insert/erase) avoid
  hitting the global allocator on every call.
