cmake_minimum_required(VERSION 3.15)
project(torchtry)

message(STATUS "CMAKE_BUILD_TYPE `${CMAKE_BUILD_TYPE}`")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR `${CMAKE_CURRENT_SOURCE_DIR}`")

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED CONFIG)

if(TORCH_CXX_FLAGS)
    message(FATAL_ERROR "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
endif()

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main
  main.cpp)

add_executable(main_kaggle
  main_kaggle.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main_kaggle "${TORCH_LIBRARIES}")
