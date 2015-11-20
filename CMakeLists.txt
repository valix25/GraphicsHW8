cmake_minimum_required(VERSION 2.8)

project(GlobalGL)

#################
# Safety Checks #
#################

# Avoid in-source builds
if ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "Warning: In-source build! Aborting. Call cmake .. from a build folder.")
endif()

#######################
# CMake path packages #
#######################

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#####################
# Compiler Settings #
#####################

message("Setting compiler settings:")
if( WIN32 )
    message("-- Windows detected")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    message("-- Linux/UNIX detected")
    # Set C++11
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Enable all warnings
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

endif()
message("-- cxx Flags: " ${CMAKE_CXX_FLAGS})
message("-- Finished setting compiler settings.")

###############
# Directories #
###############

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

#########
# Files #
#########
set(SOURCES_CPP 
	${SOURCE_DIR}/Dummy.cpp
    ${SOURCE_DIR}/lodepng.cpp
    ${SOURCE_DIR}/Polynomial.cpp
    ${SOURCE_DIR}/PolynomialRootFinder.cpp
    ${SOURCE_DIR}/plyloader.cpp)
set(INCLUDES_HPP
	${INCLUDE_DIR}/Dummy.hpp
    ${INCLUDE_DIR}/lodepng.h
    ${INCLUDE_DIR}/Polynomial.hpp
    ${INCLUDE_DIR}/PolynomialRootFinder.hpp
    ${INCLUDE_DIR}/plyloader.h)

# all files
set(SOURCES_ALL ${SOURCES_CPP} ${INCLUDES_DIR})

add_executable(GlobalGL main.cpp ${SOURCES_ALL})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("-- OpenCV found")
endif()

#find_package(SDL2 REQUIRED)
#add_definitions(-lSDL2main -lSDL2)
add_definitions( -fopenmp )

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_gpu
    opencv_surface_matching opencv_viz)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${SDL2_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
include_directories(${OpenCV_INCLUDE_DIRS})

find_file(MYLIB libply.so.2 /lib/x86_64-linux-gnu)
if(NOT MYLIB)
  MESSAGE(SEND_ERROR "Cannot find library libply.so.2")
ENDIF(NOT MYLIB)

find_library(LIBPLY libply.so
    /usr/local/lib
    /lib/x86_64-linux-gnu
    /usr/local/lib64
    /usr/lib
    /usr/lib64)
message("lib is at: ${LIBPLY}")

#add_definitions(`pkg-config --cflags opencv`)
#add_definitions(`pkg-config --libs opencv`)

target_link_libraries(GlobalGL ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} 
	${SDL2_LIBRARY} ${GLEW_LIBRARIES} ${LIBPLY} ${OpenCV_LIBS})
