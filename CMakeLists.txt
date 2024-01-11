cmake_minimum_required(VERSION 3.10)
project(PanaromicImage CXX)
set(CMAKE_CXX_STANDARD 20)

#find OpenCV
find_package(OpenCV REQUIRED)
include_directories(PanaromicImage ${OPENCV_INCLUDE_DIRS})

# Set compiler-specific warning flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC
    target_compile_options(PanaromicImage PRIVATE -Wall -Wextra -pedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang
    target_compile_options(PanaromicImage PRIVATE -Wall -Wextra -pedantic)
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(PanaromicImage ${SOURCES})
target_include_directories(PanaromicImage PRIVATE src/main.cpp)
target_link_libraries(PanaromicImage ${OpenCV_LIBS})