cmake_minimum_required(VERSION 3.5)
project(openpose)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(demo demo.cpp)

# Link OpenCV libraries
target_link_libraries(demo ${OpenCV_LIBS})
