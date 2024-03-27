cmake_minimum_required(VERSION 3.10)

# set the project name
project(othello)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

SET(CURSES_USE_NCURSES TRUE)
find_package(Curses)
include_directories(${CURSES_INCLUDE_DIR})

include_directories(${Python3_INCLUDE_DIRS})

pybind11_add_module(othello bindings.cpp  engine.h       io.cpp      tests.cpp
ai.cpp      ai.h          constants.h     io.h       tests.h
bindings.cpp  engine.cpp  main.cpp objects.cpp objects.h)

target_compile_definitions(othello PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# add the executable
add_executable(othello_demo engine.h       io.cpp      tests.cpp
ai.cpp      ai.h            constants.h   interface.cpp  io.h       tests.h
engine.cpp      interface.h   objects.cpp objects.h main.cpp)

target_link_libraries(othello_demo ${CURSES_LIBRARIES})
