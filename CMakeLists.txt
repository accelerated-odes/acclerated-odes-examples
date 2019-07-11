cmake_minimum_required(VERSION 3.10)

# Prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

project(ACCELERATED-ODES-EXAMPLES CXX)

# build options
option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_CVODE "Enable CVODE" OFF)

# check for CUDA
if(ENABLE_CUDA)
  enable_language(CUDA)
endif()

# check for AMREX location
if(NOT AMREX_DIR)
  set(AMREX_DIR "" CACHE FILEPATH "AMReX install directory")
  message(FATAL_ERROR "AMREX_DIR is not set")
else()
  set(AMREX_DIR "${AMREX_DIR}" CACHE FILEPATH "AMReX install directory")
endif()

# find AMReX
find_package(AMREX
  NAMES AMREX AMReX
  PATHS ${AMREX_DIR} ${AMREX_DIR}/lib/cmake/AMReX
  NO_DEFAULT_PATH)

if(AMREX_FOUND)
  message(STATUS "Looking for AMReX... success")
else()
  message(FATAL_ERROR "Looking for AMReX... failed")
endif()

# check for SDC location
if(NOT SDC_DIR)
  set(SDC_DIR "" CACHE FILEPATH "SDC install directory")
  message(FATAL_ERROR "SDC_DIR is not set")
else()
  set(SDC_DIR "${SDC_DIR}" CACHE FILEPATH "SDC install directory")
endif()

# create SDC target (header only library)
if(NOT TARGET SDC)
  add_library(SDC::SDC IMPORTED INTERFACE)
  target_include_directories(SDC::SDC INTERFACE "${SDC_DIR}/Source")
endif()

if(ENABLE_CVODE)
  # check for SUNDIALS location
  if(NOT SUNDIALS_DIR)
    set(SUNDIALS_DIR "" CACHE FILEPATH "SUNDIALS install directory")
    message(FATAL_ERROR "SUNDIALS_DIR is not set")
  else()
    set(SUNDIALS_DIR "${SUNDIALS_DIR}" CACHE FILEPATH "SUNDIALS install directory")
  endif()

  # find CVODE library
  find_library(CVODE_LIBS sundials_cvode
    PATHS ${SUNDIALS_DIR}/lib ${SUNDIALS_DIR}/lib64
    NO_DEFAULT_PATH)

  if(CVODE_LIBS)
    message(STATUS "Looking for CVODE... success")
  else()
    message(FATAL_ERROR "Looking for CVODE... failed")
  endif()

  # Create target for SUNDIALS
  if(NOT TARGET SUNDIALS::cvode)
    add_library(SUNDIALS::cvode UNKNOWN IMPORTED)
    set_property(TARGET SUNDIALS::cvode PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SUNDIALS_DIR}/include)
    set_property(TARGET SUNDIALS::cvode PROPERTY IMPORTED_LOCATION ${CVODE_LIBS})
  endif()
endif()

# extra link library
set(EXTRA_LINK_LIBS gfortran)

# add source
add_subdirectory(kinetics)
