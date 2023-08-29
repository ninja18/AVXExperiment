cmake_minimum_required(VERSION 3.25.1)

project(avx
	VERSION 1.0.0
	LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# add_library(fill_this)
# link_directories(fill_this)
# include_directories(fill_this)
# target_include_directories(fill_this)

file(GLOB SOURCES "*.cpp" "*.c")

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -pedantic -march=native)
# target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS})
