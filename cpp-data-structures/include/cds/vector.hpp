#pragma once

#include <cstddef>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <utility>

#include "cds/allocator.hpp"

namespace cds {

// A minimal std::vector-alike built directly on raw storage: a buffer of
// uninitialized memory obtained from Allocator, with elements placement-new
// constructed into it one at a time. Demonstrates the buffer/size/capacity
// growth strategy and the Rule of Five without leaning on any STL container.
template <typename T, typename Allocator = DefaultAllocator<T>>
class Vector {
public:
    Vector() noexcept = default;

    explicit Vector(std::size_t count, const T& value = T()) {
        reserve(count);
        for (std::size_t i = 0; i < count; ++i) emplace_back(value);
    }

    Vector(std::initializer_list<T> init) {
        reserve(init.size());
        for (const auto& v : init) emplace_back(v);
    }

    Vector(const Vector& other) {
        reserve(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i) emplace_back(other.data_[i]);
    }

    Vector(Vector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            Vector tmp(other);
            swap(tmp);
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            destroyAll();
            deallocate();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~Vector() {
        destroyAll();
        deallocate();
    }

    void swap(Vector& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    void reserve(std::size_t newCapacity) {
        if (newCapacity <= capacity_) return;
        T* newData = alloc_.allocate(newCapacity);
        for (std::size_t i = 0; i < size_; ++i) {
            ::new (static_cast<void*>(newData + i)) T(std::move(data_[i]));
            data_[i].~T();
        }
        alloc_.deallocate(data_, capacity_);
        data_ = newData;
        capacity_ = newCapacity;
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) grow();
        ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
        return data_[size_++];
    }

    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }

    void pop_back() {
        if (size_ == 0) throw std::out_of_range("Vector::pop_back on empty vector");
        --size_;
        data_[size_].~T();
    }

    void clear() noexcept {
        destroyAll();
        size_ = 0;
    }

    T& operator[](std::size_t index) noexcept { return data_[index]; }
    const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    T& at(std::size_t index) {
        if (index >= size_) throw std::out_of_range("Vector::at index out of range");
        return data_[index];
    }
    const T& at(std::size_t index) const {
        if (index >= size_) throw std::out_of_range("Vector::at index out of range");
        return data_[index];
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }

    T& front() noexcept { return data_[0]; }
    T& back() noexcept { return data_[size_ - 1]; }

private:
    void grow() {
        std::size_t newCapacity = capacity_ == 0 ? 4 : capacity_ * 2;
        reserve(newCapacity);
    }

    void destroyAll() noexcept {
        for (std::size_t i = 0; i < size_; ++i) data_[i].~T();
    }

    void deallocate() noexcept {
        alloc_.deallocate(data_, capacity_);
        data_ = nullptr;
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
    Allocator alloc_{};
};

}  // namespace cds
