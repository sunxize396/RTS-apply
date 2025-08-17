
cmake_minimum_required(VERSION 3.1)
 
project(imgdisplay)
 
find_package(OpenCV REQUIRED)
 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
 

add_executable(imgdisplay main.cpp)
 

target_link_libraries(imgdisplay PRIVATE ${OpenCV_LIBS})
