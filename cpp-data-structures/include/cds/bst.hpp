#pragma once

#include <cstddef>
#include <utility>

#include "cds/allocator.hpp"
#include "cds/vector.hpp"

namespace cds {

// Unbalanced binary search tree. Node allocation goes through Alloc<Node>
// (PoolAllocator by default) so insert/erase-heavy workloads recycle freed
// node memory instead of hitting the global allocator each time.
template <typename T, template <typename> class Alloc = PoolAllocator>
class BST {
    struct Node {
        T data;
        Node* left = nullptr;
        Node* right = nullptr;
        explicit Node(const T& value) : data(value) {}
        explicit Node(T&& value) : data(std::move(value)) {}
    };

    using NodeAllocator = Alloc<Node>;

public:
    BST() noexcept = default;

    BST(const BST& other) { root_ = cloneSubtree(other.root_); size_ = other.size_; }

    BST(BST&& other) noexcept : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    BST& operator=(const BST& other) {
        if (this != &other) {
            BST tmp(other);
            swap(tmp);
        }
        return *this;
    }

    BST& operator=(BST&& other) noexcept {
        if (this != &other) {
            destroySubtree(root_);
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~BST() { destroySubtree(root_); }

    void swap(BST& other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
    }

    void insert(const T& value) { root_ = insertImpl(root_, value); }
    void insert(T&& value) { root_ = insertImpl(root_, std::move(value)); }

    bool contains(const T& value) const {
        Node* n = root_;
        while (n) {
            if (value < n->data) n = n->left;
            else if (n->data < value) n = n->right;
            else return true;
        }
        return false;
    }

    void erase(const T& value) { root_ = eraseImpl(root_, value); }

    std::size_t height() const { return heightImpl(root_); }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // In-order traversal yields values in sorted order.
    Vector<T> inorder() const {
        Vector<T> out;
        inorderImpl(root_, out);
        return out;
    }

private:
    template <typename U>
    Node* insertImpl(Node* node, U&& value) {
        if (!node) {
            ++size_;
            return allocateAndConstruct(alloc_, std::forward<U>(value));
        }
        if (value < node->data) node->left = insertImpl(node->left, std::forward<U>(value));
        else if (node->data < value) node->right = insertImpl(node->right, std::forward<U>(value));
        return node;
    }

    Node* eraseImpl(Node* node, const T& value) {
        if (!node) return nullptr;
        if (value < node->data) {
            node->left = eraseImpl(node->left, value);
        } else if (node->data < value) {
            node->right = eraseImpl(node->right, value);
        } else {
            if (!node->left) {
                Node* right = node->right;
                destroyNode(node);
                --size_;
                return right;
            }
            if (!node->right) {
                Node* left = node->left;
                destroyNode(node);
                --size_;
                return left;
            }
            Node* successor = node->right;
            while (successor->left) successor = successor->left;
            node->data = successor->data;
            node->right = eraseSmallest(node->right);
        }
        return node;
    }

    // Removes the minimum node of the subtree rooted at `node`; used after
    // copying a successor's value up during two-child erase.
    Node* eraseSmallest(Node* node) {
        if (!node->left) {
            Node* right = node->right;
            destroyNode(node);
            --size_;
            return right;
        }
        node->left = eraseSmallest(node->left);
        return node;
    }

    static std::size_t heightImpl(Node* node) {
        if (!node) return 0;
        std::size_t l = heightImpl(node->left);
        std::size_t r = heightImpl(node->right);
        return 1 + (l > r ? l : r);
    }

    static void inorderImpl(Node* node, Vector<T>& out) {
        if (!node) return;
        inorderImpl(node->left, out);
        out.push_back(node->data);
        inorderImpl(node->right, out);
    }

    Node* cloneSubtree(Node* node) {
        if (!node) return nullptr;
        Node* copy = allocateAndConstruct(alloc_, node->data);
        copy->left = cloneSubtree(node->left);
        copy->right = cloneSubtree(node->right);
        return copy;
    }

    void destroySubtree(Node* node) noexcept {
        if (!node) return;
        destroySubtree(node->left);
        destroySubtree(node->right);
        destroyNode(node);
    }

    void destroyNode(Node* node) noexcept { destroyAndDeallocate(alloc_, node); }

    Node* root_ = nullptr;
    std::size_t size_ = 0;
    NodeAllocator alloc_{};
};

}  // namespace cds
