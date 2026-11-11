set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})

# python module.
# pip install "pybind11[global]"
find_package(pybind11 QUIET)
if(pybind11_FOUND)
  file(GLOB PYBIND_CPP python/*.cpp)
  pybind11_add_module(C ${PYBIND_CPP})
  target_link_libraries(C PRIVATE pipeann)
else()
  message(STATUS "pybind11 not found; skipping PipeANN Python module.")
endif()
