cmake_minimum_required(VERSION 3.16)
project(VehicleDistanceMeasurement LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Apple Silicon (M1/M2/M3/M4) optimizations
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    add_compile_options(-O3 -ffast-math)
endif()

# Find OpenCV (required components)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

# Optional: ONNX Runtime for better YOLO performance
# Install eg: brew install onnxruntime
# Or build from source with CoreML EP
option(USE_ONNXRUNTIME "Use ONNX Runtime instead of OpenCV DNN" OFF)

if(USE_ONNXRUNTIME)
    find_package(onnxruntime REQUIRED)
    message(STATUS "Building with ONNX Runtime")
else()
    message(STATUS "Building with OpenCV DNN (export models to ONNX first)")
endif()

add_executable(vehicle_distance
    src/main.cpp
    src/roi_utils.cpp
    src/distance_calculator.cpp
    src/visualization.cpp
    src/yolo_detector.cpp
)

target_include_directories(vehicle_distance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vehicle_distance PRIVATE
    ${OpenCV_LIBS}
)

if(USE_ONNXRUNTIME)
    target_link_libraries(vehicle_distance PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(vehicle_distance PRIVATE USE_ONNXRUNTIME)
endif()

# Install
install(TARGETS vehicle_distance RUNTIME DESTINATION bin)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
