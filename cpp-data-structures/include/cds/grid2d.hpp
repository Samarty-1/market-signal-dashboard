#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "cds/allocator.hpp"

namespace cds {

// A simple 2D grid/matrix backed by one contiguous allocation, indexed as
// row-major (row * cols_ + col). Demonstrates manually managing a 2D
// structure without nested containers.
template <typename T, typename Allocator = DefaultAllocator<T>>
class Grid2D {
public:
    Grid2D() noexcept = default;

    Grid2D(std::size_t rows, std::size_t cols, const T& initial = T())
        : rows_(rows), cols_(cols) {
        std::size_t count = rows_ * cols_;
        data_ = alloc_.allocate(count);
        for (std::size_t i = 0; i < count; ++i) {
            ::new (static_cast<void*>(data_ + i)) T(initial);
        }
    }

    Grid2D(const Grid2D& other) : rows_(other.rows_), cols_(other.cols_) {
        std::size_t count = rows_ * cols_;
        data_ = alloc_.allocate(count);
        for (std::size_t i = 0; i < count; ++i) {
            ::new (static_cast<void*>(data_ + i)) T(other.data_[i]);
        }
    }

    Grid2D(Grid2D&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;
        other.rows_ = other.cols_ = 0;
    }

    Grid2D& operator=(const Grid2D& other) {
        if (this != &other) {
            Grid2D tmp(other);
            swap(tmp);
        }
        return *this;
    }

    Grid2D& operator=(Grid2D&& other) noexcept {
        if (this != &other) {
            destroyAll();
            alloc_.deallocate(data_, rows_ * cols_);
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.data_ = nullptr;
            other.rows_ = other.cols_ = 0;
        }
        return *this;
    }

    ~Grid2D() {
        destroyAll();
        alloc_.deallocate(data_, rows_ * cols_);
    }

    void swap(Grid2D& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
    }

    T& operator()(std::size_t row, std::size_t col) noexcept {
        return data_[row * cols_ + col];
    }
    const T& operator()(std::size_t row, std::size_t col) const noexcept {
        return data_[row * cols_ + col];
    }

    T& at(std::size_t row, std::size_t col) {
        if (row >= rows_ || col >= cols_) throw std::out_of_range("Grid2D::at index out of range");
        return data_[row * cols_ + col];
    }
    const T& at(std::size_t row, std::size_t col) const {
        if (row >= rows_ || col >= cols_) throw std::out_of_range("Grid2D::at index out of range");
        return data_[row * cols_ + col];
    }

    void fill(const T& value) {
        std::size_t count = rows_ * cols_;
        for (std::size_t i = 0; i < count; ++i) data_[i] = value;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_ * cols_; }
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

private:
    void destroyAll() noexcept {
        std::size_t count = rows_ * cols_;
        for (std::size_t i = 0; i < count; ++i) data_[i].~T();
    }

    T* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    Allocator alloc_{};
};

}  // namespace cds
