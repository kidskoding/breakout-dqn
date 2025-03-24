cmake_minimum_required(VERSION 3.2)
project(breakout-dqn)

set(CMAKE_PREFIX_PATH ~/libtorch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(breakout-dqn src/main.cpp)
target_link_libraries(breakout-dqn ${TORCH_LIBRARIES})
set_property(TARGET breakout-dqn PROPERTY CXX_STANDARD 14)
