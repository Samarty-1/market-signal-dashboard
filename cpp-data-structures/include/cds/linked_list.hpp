#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "cds/allocator.hpp"

namespace cds {

// Doubly linked list with node allocation delegated to Alloc<Node>. Defaults
// to PoolAllocator so repeated push/pop churn reuses freed node slots instead
// of round-tripping through the global allocator on every operation.
template <typename T, template <typename> class Alloc = PoolAllocator>
class LinkedList {
    struct Node {
        T data;
        Node* prev = nullptr;
        Node* next = nullptr;
        template <typename... Args>
        explicit Node(Args&&... args) : data(std::forward<Args>(args)...) {}
    };

    using NodeAllocator = Alloc<Node>;

public:
    class Iterator {
    public:
        explicit Iterator(Node* node) : node_(node) {}
        T& operator*() const { return node_->data; }
        T* operator->() const { return &node_->data; }
        Iterator& operator++() { node_ = node_->next; return *this; }
        Iterator& operator--() { node_ = node_->prev; return *this; }
        bool operator==(const Iterator& other) const { return node_ == other.node_; }
        bool operator!=(const Iterator& other) const { return node_ != other.node_; }

    private:
        friend class LinkedList;
        Node* node_;
    };

    LinkedList() noexcept = default;

    LinkedList(const LinkedList& other) {
        for (Node* n = other.head_; n; n = n->next) pushBackImpl(n->data);
    }

    LinkedList(LinkedList&& other) noexcept
        : head_(other.head_), tail_(other.tail_), size_(other.size_) {
        other.head_ = other.tail_ = nullptr;
        other.size_ = 0;
    }

    LinkedList& operator=(const LinkedList& other) {
        if (this != &other) {
            LinkedList tmp(other);
            swap(tmp);
        }
        return *this;
    }

    LinkedList& operator=(LinkedList&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = other.head_;
            tail_ = other.tail_;
            size_ = other.size_;
            other.head_ = other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~LinkedList() { clear(); }

    void swap(LinkedList& other) noexcept {
        std::swap(head_, other.head_);
        std::swap(tail_, other.tail_);
        std::swap(size_, other.size_);
    }

    void push_back(const T& value) { pushBackImpl(value); }
    void push_back(T&& value) { pushBackImpl(std::move(value)); }
    void push_front(const T& value) { pushFrontImpl(value); }
    void push_front(T&& value) { pushFrontImpl(std::move(value)); }

    void pop_back() {
        if (!tail_) throw std::out_of_range("LinkedList::pop_back on empty list");
        Node* doomed = tail_;
        tail_ = tail_->prev;
        if (tail_) tail_->next = nullptr; else head_ = nullptr;
        destroyNode(doomed);
        --size_;
    }

    void pop_front() {
        if (!head_) throw std::out_of_range("LinkedList::pop_front on empty list");
        Node* doomed = head_;
        head_ = head_->next;
        if (head_) head_->prev = nullptr; else tail_ = nullptr;
        destroyNode(doomed);
        --size_;
    }

    Iterator erase(Iterator it) {
        Node* node = it.node_;
        Node* next = node->next;
        if (node->prev) node->prev->next = node->next; else head_ = node->next;
        if (node->next) node->next->prev = node->prev; else tail_ = node->prev;
        destroyNode(node);
        --size_;
        return Iterator(next);
    }

    void clear() noexcept {
        Node* n = head_;
        while (n) {
            Node* next = n->next;
            destroyNode(n);
            n = next;
        }
        head_ = tail_ = nullptr;
        size_ = 0;
    }

    Iterator begin() noexcept { return Iterator(head_); }
    Iterator end() noexcept { return Iterator(nullptr); }

    T& front() { return head_->data; }
    T& back() { return tail_->data; }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    template <typename U>
    void pushBackImpl(U&& value) {
        Node* node = allocateAndConstruct(alloc_, std::forward<U>(value));
        node->prev = tail_;
        node->next = nullptr;
        if (tail_) tail_->next = node; else head_ = node;
        tail_ = node;
        ++size_;
    }

    template <typename U>
    void pushFrontImpl(U&& value) {
        Node* node = allocateAndConstruct(alloc_, std::forward<U>(value));
        node->next = head_;
        node->prev = nullptr;
        if (head_) head_->prev = node; else tail_ = node;
        head_ = node;
        ++size_;
    }

    void destroyNode(Node* node) noexcept { destroyAndDeallocate(alloc_, node); }

    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    std::size_t size_ = 0;
    NodeAllocator alloc_{};
};

}  // namespace cds
