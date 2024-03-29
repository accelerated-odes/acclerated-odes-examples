# compile example for CPU
add_executable(kinetics1D main.cpp)
target_link_libraries(kinetics1D PUBLIC AMReX::amrex SDC::SDC)
if(ENABLE_CVODE)
  target_link_libraries(kinetics1D PUBLIC SUNDIALS::cvode)
  target_compile_definitions(kinetics1D PUBLIC USE_CVODE)
  if(ENABLE_KLU)
    target_link_libraries(kinetics1D PUBLIC SUNDIALS::sunls_klu SUITESPARSE::klu)
    target_compile_definitions(kinetics1D PUBLIC USE_KLU)
  endif()
  if(ENABLE_OpenMP AND OpenMP_CXX_FOUND)
    target_link_libraries(kinetics1D PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(kinetics1D PUBLIC USE_OPENMP)
  endif()
endif()

# compile example for GPU
if(ENABLE_CUDA)
  add_executable(kinetics1D_cuda main.cu)
  target_include_directories(kinetics1D_cuda PUBLIC)
  target_link_libraries(kinetics1D_cuda PUBLIC AMReX::amrex SDC::SDC)
  target_compile_definitions(kinetics1D_cuda PUBLIC AMREX_USE_CUDA)
  # if(ENABLE_CVODE)
  #   target_link_libraries(kinetics1D PUBLIC SUNDIALS::cuvode)
  # endif()
endif()