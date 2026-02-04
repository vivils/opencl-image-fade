cmake_minimum_required(VERSION 3.10)

project(soclp-1 LANGUAGES CXX)

#set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        PNG::PNG
        opencv_core
        opencv_imgcodecs
        opencv_highgui
        ${OpenCL_LIBRARIES}
)


