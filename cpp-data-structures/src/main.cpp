#include <iostream>
#include <string>

#include "cds/allocator.hpp"
#include "cds/bst.hpp"
#include "cds/grid2d.hpp"
#include "cds/linked_list.hpp"
#include "cds/vector.hpp"

int main() {
    std::cout << "== Vector<int> ==\n";
    cds::Vector<int> v;
    for (int i = 1; i <= 5; ++i) v.push_back(i * i);
    for (int x : v) std::cout << x << ' ';
    std::cout << "\nsize=" << v.size() << " capacity=" << v.capacity() << "\n\n";

    std::cout << "== LinkedList<std::string> (PoolAllocator-backed nodes) ==\n";
    cds::LinkedList<std::string> list;
    list.push_back("alpha");
    list.push_back("beta");
    list.push_front("start");
    for (const auto& s : list) std::cout << s << ' ';
    std::cout << "\nsize=" << list.size() << "\n\n";

    std::cout << "== BST<int> ==\n";
    cds::BST<int> tree;
    for (int x : {5, 3, 8, 1, 4, 7, 9}) tree.insert(x);
    std::cout << "inorder: ";
    for (int x : tree.inorder()) std::cout << x << ' ';
    std::cout << "\nheight=" << tree.height() << " contains(7)=" << tree.contains(7)
              << " contains(6)=" << tree.contains(6) << "\n";
    tree.erase(3);
    std::cout << "after erase(3): ";
    for (int x : tree.inorder()) std::cout << x << ' ';
    std::cout << "\n\n";

    std::cout << "== Grid2D<int> (Simple 2D) ==\n";
    cds::Grid2D<int> grid(3, 4, 0);
    for (std::size_t r = 0; r < grid.rows(); ++r)
        for (std::size_t c = 0; c < grid.cols(); ++c)
            grid(r, c) = static_cast<int>(r * grid.cols() + c);

    for (std::size_t r = 0; r < grid.rows(); ++r) {
        for (std::size_t c = 0; c < grid.cols(); ++c) std::cout << grid(r, c) << '\t';
        std::cout << '\n';
    }

    return 0;
}
