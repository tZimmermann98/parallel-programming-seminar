cmake_minimum_required(VERSION 3.0.0)
project(ppseminar_sequential VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_executable(main main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
