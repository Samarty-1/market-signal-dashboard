cmake_minimum_required(VERSION 3.14)
project(CustomDataStructures CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_library(cds INTERFACE)
target_include_directories(cds INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(demo src/main.cpp)
target_link_libraries(demo PRIVATE cds)

enable_testing()
add_executable(run_tests tests/test_main.cpp)
target_link_libraries(run_tests PRIVATE cds)
add_test(NAME AllTests COMMAND run_tests)
