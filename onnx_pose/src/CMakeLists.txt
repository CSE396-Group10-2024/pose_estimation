cmake_minimum_required(VERSION 3.0)

set(LibraryName "src")

# Create library
add_library (${LibraryName} MPPose.h MPPose.cpp MPPersonDet.h MPPersonDet.cpp processFrame.h processFrame.cpp)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})


# Link Common Helper module

# For InferenceHelper


