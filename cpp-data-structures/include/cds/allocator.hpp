#pragma once

#include <cstddef>
#include <new>
#include <utility>

namespace cds {

// Thin wrapper around global operator new/delete. Conforms to the minimal
// allocator interface used throughout this library: allocate(n)/deallocate(p, n).
template <typename T>
class DefaultAllocator {
public:
    using value_type = T;

    DefaultAllocator() noexcept = default;
    template <typename U>
    DefaultAllocator(const DefaultAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p);
    }
};

// Fixed-size free-list pool allocator. Efficient for node-based containers
// (LinkedList, BST) that only ever allocate/deallocate one object at a time:
// freed slots are recycled instead of going back to the OS. Falls back to
// the global allocator for batch requests (n != 1), which single-object
// containers never issue.
template <typename T, std::size_t ChunkSlots = 512>
class PoolAllocator {
    union Slot {
        alignas(T) unsigned char storage[sizeof(T)];
        Slot* next;
    };

    struct Chunk {
        Slot slots[ChunkSlots];
        Chunk* next;
    };

public:
    using value_type = T;

    PoolAllocator() noexcept = default;
    PoolAllocator(const PoolAllocator&) noexcept {}
    template <typename U>
    PoolAllocator(const PoolAllocator<U, ChunkSlots>&) noexcept {}

    PoolAllocator(PoolAllocator&& other) noexcept
        : freeList_(other.freeList_), chunks_(other.chunks_) {
        other.freeList_ = nullptr;
        other.chunks_ = nullptr;
    }

    PoolAllocator& operator=(PoolAllocator&& other) noexcept {
        if (this != &other) {
            releaseAll();
            freeList_ = other.freeList_;
            chunks_ = other.chunks_;
            other.freeList_ = nullptr;
            other.chunks_ = nullptr;
        }
        return *this;
    }

    ~PoolAllocator() { releaseAll(); }

    T* allocate(std::size_t n) {
        if (n != 1) {
            // Batch requests aren't pooled; hand them to the global allocator.
            return static_cast<T*>(::operator new(n * sizeof(T)));
        }
        if (!freeList_) refill();
        Slot* slot = freeList_;
        freeList_ = freeList_->next;
        return reinterpret_cast<T*>(slot);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (!p) return;
        if (n != 1) {
            ::operator delete(p);
            return;
        }
        Slot* slot = reinterpret_cast<Slot*>(p);
        slot->next = freeList_;
        freeList_ = slot;
    }

private:
    void refill() {
        Chunk* chunk = new Chunk;
        chunk->next = chunks_;
        chunks_ = chunk;
        for (std::size_t i = 0; i < ChunkSlots; ++i) {
            chunk->slots[i].next = freeList_;
            freeList_ = &chunk->slots[i];
        }
    }

    void releaseAll() noexcept {
        Chunk* chunk = chunks_;
        while (chunk) {
            Chunk* next = chunk->next;
            delete chunk;
            chunk = next;
        }
        chunks_ = nullptr;
        freeList_ = nullptr;
    }

    Slot* freeList_ = nullptr;
    Chunk* chunks_ = nullptr;
};

// Constructs/destroys objects in raw memory obtained from an allocator,
// keeping the placement-new bookkeeping in one place.
template <typename Alloc, typename... Args>
typename Alloc::value_type* allocateAndConstruct(Alloc& alloc, Args&&... args) {
    using T = typename Alloc::value_type;
    T* p = alloc.allocate(1);
    ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    return p;
}

template <typename Alloc>
void destroyAndDeallocate(Alloc& alloc, typename Alloc::value_type* p) noexcept {
    using T = typename Alloc::value_type;
    if (!p) return;
    p->~T();
    alloc.deallocate(p, 1);
}

}  // namespace cds
