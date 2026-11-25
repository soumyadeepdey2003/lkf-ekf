cmake_minimum_required(VERSION 3.20)

project(KF_Research LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Locate the existing OpenCV installation
set(OpenCV_DIR
    "E:/KF_Research/third_party/opencv_install/x64/mingw/lib"
)

# Find required OpenCV components
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    video
    videoio
    highgui
    imgcodecs
)

# Measurement generator library
add_library(measurement_generator STATIC
    detection/measurement_generator.cpp
)

target_include_directories(measurement_generator
    PUBLIC
        detection
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(measurement_generator
    PUBLIC
        ${OpenCV_LIBS}
)

# Display configuration information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV directory: ${OpenCV_DIR}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

message(STATUS "Research project configured successfully")

# CPU measurement generator executable

add_executable(measurement_generator_cpu
    detection/main.cpp
)

target_link_libraries(measurement_generator_cpu
    PRIVATE
        measurement_generator
)

set_target_properties(measurement_generator_cpu
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
