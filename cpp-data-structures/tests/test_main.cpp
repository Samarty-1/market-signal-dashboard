// Lightweight assert-based test harness (no external framework dependency).
#include <cstdlib>
#include <iostream>
#include <string>

#include "cds/allocator.hpp"
#include "cds/bst.hpp"
#include "cds/grid2d.hpp"
#include "cds/linked_list.hpp"
#include "cds/vector.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "FAILED: " << expr << " (" << file << ":" << line << ")\n";
    }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

void test_vector_basic() {
    cds::Vector<int> v;
    CHECK(v.empty());
    for (int i = 0; i < 100; ++i) v.push_back(i);
    CHECK(v.size() == 100);
    for (int i = 0; i < 100; ++i) CHECK(v[i] == i);
    v.pop_back();
    CHECK(v.size() == 99);
    v.clear();
    CHECK(v.empty());
}

void test_vector_copy_move() {
    cds::Vector<std::string> a;
    a.push_back("hello");
    a.push_back("world");
    cds::Vector<std::string> b(a);
    CHECK(b.size() == 2);
    CHECK(b[0] == "hello");
    a[0] = "changed";
    CHECK(b[0] == "hello");  // deep copy, not aliased

    cds::Vector<std::string> c(std::move(b));
    CHECK(c.size() == 2);
    CHECK(b.empty());  // moved-from
}

void test_vector_out_of_range() {
    cds::Vector<int> v;
    v.push_back(1);
    bool threw = false;
    try {
        v.at(5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void test_linked_list_basic() {
    cds::LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_front(0);
    CHECK(list.size() == 3);
    int expected = 0;
    for (int x : list) CHECK(x == expected++);
    list.pop_front();
    CHECK(list.front() == 1);
    list.pop_back();
    CHECK(list.back() == 1);
    CHECK(list.size() == 1);
}

void test_linked_list_erase() {
    cds::LinkedList<int> list;
    for (int i = 0; i < 5; ++i) list.push_back(i);
    auto it = list.begin();
    ++it;  // points at 1
    it = list.erase(it);  // removes 1, it now points at 2
    CHECK(*it == 2);
    CHECK(list.size() == 4);
}

void test_linked_list_pool_reuse() {
    // Push/pop churn should not grow unbounded memory; correctness check only.
    cds::LinkedList<int> list;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 50; ++i) list.push_back(i);
        while (!list.empty()) list.pop_back();
    }
    CHECK(list.empty());
    CHECK(list.size() == 0);
}

void test_bst_basic() {
    cds::BST<int> tree;
    for (int x : {5, 3, 8, 1, 4, 7, 9, 2, 6}) tree.insert(x);
    CHECK(tree.size() == 9);
    CHECK(tree.contains(7));
    CHECK(!tree.contains(42));

    auto sorted = tree.inorder();
    CHECK(sorted.size() == 9);
    for (std::size_t i = 1; i < sorted.size(); ++i) CHECK(sorted[i - 1] < sorted[i]);
}

void test_bst_erase() {
    cds::BST<int> tree;
    for (int x : {5, 3, 8, 1, 4, 7, 9}) tree.insert(x);
    tree.erase(3);  // two-child node
    CHECK(!tree.contains(3));
    CHECK(tree.size() == 6);
    tree.erase(9);  // leaf
    CHECK(!tree.contains(9));
    tree.erase(8);  // one-child node
    CHECK(!tree.contains(8));
    auto sorted = tree.inorder();
    for (std::size_t i = 1; i < sorted.size(); ++i) CHECK(sorted[i - 1] < sorted[i]);
}

void test_bst_copy() {
    cds::BST<int> a;
    for (int x : {5, 2, 8}) a.insert(x);
    cds::BST<int> b(a);
    b.insert(100);
    CHECK(!a.contains(100));
    CHECK(b.contains(100));
}

void test_grid2d_basic() {
    cds::Grid2D<int> grid(3, 4, 7);
    CHECK(grid.rows() == 3);
    CHECK(grid.cols() == 4);
    for (std::size_t r = 0; r < grid.rows(); ++r)
        for (std::size_t c = 0; c < grid.cols(); ++c) CHECK(grid(r, c) == 7);

    grid(1, 2) = 42;
    CHECK(grid.at(1, 2) == 42);

    bool threw = false;
    try {
        grid.at(10, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void test_grid2d_copy() {
    cds::Grid2D<int> a(2, 2, 1);
    cds::Grid2D<int> b(a);
    a(0, 0) = 99;
    CHECK(b(0, 0) == 1);  // deep copy
}

void test_pool_allocator_reuse() {
    cds::PoolAllocator<int> alloc;
    int* p1 = alloc.allocate(1);
    *p1 = 123;
    alloc.deallocate(p1, 1);
    int* p2 = alloc.allocate(1);
    CHECK(p1 == p2);  // freed slot recycled
    alloc.deallocate(p2, 1);
}

}  // namespace

int main() {
    test_vector_basic();
    test_vector_copy_move();
    test_vector_out_of_range();
    test_linked_list_basic();
    test_linked_list_erase();
    test_linked_list_pool_reuse();
    test_bst_basic();
    test_bst_erase();
    test_bst_copy();
    test_grid2d_basic();
    test_grid2d_copy();
    test_pool_allocator_reuse();

    std::cout << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
