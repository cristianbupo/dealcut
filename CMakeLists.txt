cmake_minimum_required(VERSION 3.16)
project(dealcut LANGUAGES C CXX)

# ===========================================================
# Target 1: step-85 (deal.II tutorial)
# ===========================================================
find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        $ENV{HOME}/dealii
)

if(${deal.II_FOUND})
  if(NOT DEAL_II_WITH_LAPACK)
    message(WARNING "deal.II was not configured with DEAL_II_WITH_LAPACK=ON. step-85 requires LAPACK.")
  else()
    deal_ii_initialize_cached_variables()
    add_executable(step-85 step-85.cc)
    deal_ii_setup_target(step-85)
  endif()
else()
  message(STATUS "deal.II not found — skipping step-85")
endif()

# ===========================================================
# Target 2: SOC (deal.II + VTK + Gmsh)
# ===========================================================
if(${deal.II_FOUND} AND DEAL_II_WITH_LAPACK)
  # Request only the VTK modules used by SOC/growth to avoid optional Qt GUI deps.
  find_package(VTK QUIET COMPONENTS CommonCore CommonDataModel IOXML)
  find_path(GMSH_INCLUDE_DIR gmsh.h HINTS /usr/include)
  find_library(GMSH_LIBRARY gmsh HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)

  if(VTK_FOUND AND GMSH_INCLUDE_DIR AND GMSH_LIBRARY)
    add_executable(SOC SOC.cc)
    deal_ii_setup_target(SOC)
    target_link_libraries(SOC ${VTK_LIBRARIES})
    if(COMMAND vtk_module_autoinit)
      vtk_module_autoinit(TARGETS SOC MODULES ${VTK_LIBRARIES})
    endif()
    target_include_directories(SOC PRIVATE ${GMSH_INCLUDE_DIR})
    target_link_libraries(SOC ${GMSH_LIBRARY})

    add_executable(growth growth.cc)
    deal_ii_setup_target(growth)
    target_link_libraries(growth ${VTK_LIBRARIES})
    if(COMMAND vtk_module_autoinit)
      vtk_module_autoinit(TARGETS growth MODULES ${VTK_LIBRARIES})
    endif()
    target_include_directories(growth PRIVATE ${GMSH_INCLUDE_DIR})
    target_link_libraries(growth ${GMSH_LIBRARY})
  else()
    message(STATUS "VTK or Gmsh not found — skipping SOC")
  endif()
else()
  message(STATUS "deal.II not found — skipping SOC")
endif()

# ===========================================================
# Target 3: bulk (CutFEM-Library example)
# ===========================================================
set(CUTFEM_DIR "$ENV{HOME}/git/CutFEM-Library" CACHE PATH "Path to CutFEM-Library source tree")
set(CUTFEM_BUILD_DIR "${CUTFEM_DIR}/build" CACHE PATH "Path to CutFEM-Library build directory")

if(EXISTS "${CUTFEM_DIR}/cpp/cutfem.hpp" AND EXISTS "${CUTFEM_BUILD_DIR}/lib")
  add_executable(bulk bulk.cpp)
  set_target_properties(bulk PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  # Include directories replicating the original CutFEM-Library build:
  #  - cpp/problem/ (INTERFACE) makes ../cutfem.hpp resolve to cpp/cutfem.hpp
  #  - CMAKE_SOURCE_DIR and cpp/ are added for transitive headers
  target_include_directories(bulk PRIVATE
    ${CUTFEM_DIR}                # original CMAKE_SOURCE_DIR
    ${CUTFEM_DIR}/cpp            # for cutfem.hpp and sub-headers
    ${CUTFEM_DIR}/cpp/problem    # INTERFACE include from problem lib
    ${CUTFEM_DIR}/cpp/FESpace    # PUBLIC include from FESpace lib
    ${CUTFEM_DIR}/cpp/common
    ${CUTFEM_DIR}/cpp/solver
    ${CUTFEM_DIR}/cpp/parallel
    ${CUTFEM_DIR}/cpp/num
    ${CUTFEM_DIR}/cpp/algoim
    ${CUTFEM_DIR}/cpp/concept
    ${CUTFEM_BUILD_DIR}          # for generated cutFEMConfig.h
  )

  target_link_directories(bulk PRIVATE ${CUTFEM_BUILD_DIR}/lib)
  target_link_libraries(bulk PRIVATE parallel common FESpace solver problem)

  # step85_cutfem: Step-85 Poisson problem reimplemented with CutFEM-Library
  add_executable(step85_cutfem step85_cutfem.cpp)
  set_target_properties(step85_cutfem PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
  target_include_directories(step85_cutfem PRIVATE
    ${CUTFEM_DIR}
    ${CUTFEM_DIR}/cpp
    ${CUTFEM_DIR}/cpp/problem
    ${CUTFEM_DIR}/cpp/FESpace
    ${CUTFEM_DIR}/cpp/common
    ${CUTFEM_DIR}/cpp/solver
    ${CUTFEM_DIR}/cpp/parallel
    ${CUTFEM_DIR}/cpp/num
    ${CUTFEM_DIR}/cpp/algoim
    ${CUTFEM_DIR}/cpp/concept
    ${CUTFEM_BUILD_DIR}
  )
  target_link_directories(step85_cutfem PRIVATE ${CUTFEM_BUILD_DIR}/lib)
  target_link_libraries(step85_cutfem PRIVATE parallel common FESpace solver problem)

  # SOC_cutfem: SOC elasticity problem reimplemented with CutFEM-Library + Gmsh
  find_path(GMSH_INCLUDE_DIR gmsh.h HINTS /usr/include)
  find_library(GMSH_LIBRARY gmsh HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
  if(GMSH_INCLUDE_DIR AND GMSH_LIBRARY)
    add_executable(SOC_cutfem SOC_cutfem.cpp)
    set_target_properties(SOC_cutfem PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(SOC_cutfem PRIVATE
      ${CUTFEM_DIR}
      ${CUTFEM_DIR}/cpp
      ${CUTFEM_DIR}/cpp/problem
      ${CUTFEM_DIR}/cpp/FESpace
      ${CUTFEM_DIR}/cpp/common
      ${CUTFEM_DIR}/cpp/solver
      ${CUTFEM_DIR}/cpp/parallel
      ${CUTFEM_DIR}/cpp/num
      ${CUTFEM_DIR}/cpp/algoim
      ${CUTFEM_DIR}/cpp/concept
      ${CUTFEM_BUILD_DIR}
      ${GMSH_INCLUDE_DIR}
    )
    target_link_directories(SOC_cutfem PRIVATE ${CUTFEM_BUILD_DIR}/lib)
    target_link_libraries(SOC_cutfem PRIVATE parallel common FESpace solver problem ${GMSH_LIBRARY})

    # growth_cutfem: Growth/ossification with CutFEM-Library + Gmsh
    add_executable(growth_cutfem growth_cutfem.cpp)
    set_target_properties(growth_cutfem PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(growth_cutfem PRIVATE
      ${CUTFEM_DIR}
      ${CUTFEM_DIR}/cpp
      ${CUTFEM_DIR}/cpp/problem
      ${CUTFEM_DIR}/cpp/FESpace
      ${CUTFEM_DIR}/cpp/common
      ${CUTFEM_DIR}/cpp/solver
      ${CUTFEM_DIR}/cpp/parallel
      ${CUTFEM_DIR}/cpp/num
      ${CUTFEM_DIR}/cpp/algoim
      ${CUTFEM_DIR}/cpp/concept
      ${CUTFEM_BUILD_DIR}
      ${GMSH_INCLUDE_DIR}
    )
    target_link_directories(growth_cutfem PRIVATE ${CUTFEM_BUILD_DIR}/lib)
    target_link_libraries(growth_cutfem PRIVATE parallel common FESpace solver problem ${GMSH_LIBRARY})

    # growth_iterative_cutfem: Iterative ossification with CutFEM-Library + Gmsh
    add_executable(growth_iterative_cutfem growth_iterative_cutfem.cpp)
    set_target_properties(growth_iterative_cutfem PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(growth_iterative_cutfem PRIVATE
      ${CUTFEM_DIR}
      ${CUTFEM_DIR}/cpp
      ${CUTFEM_DIR}/cpp/problem
      ${CUTFEM_DIR}/cpp/FESpace
      ${CUTFEM_DIR}/cpp/common
      ${CUTFEM_DIR}/cpp/solver
      ${CUTFEM_DIR}/cpp/parallel
      ${CUTFEM_DIR}/cpp/num
      ${CUTFEM_DIR}/cpp/algoim
      ${CUTFEM_DIR}/cpp/concept
      ${CUTFEM_BUILD_DIR}
      ${GMSH_INCLUDE_DIR}
    )
    target_link_directories(growth_iterative_cutfem PRIVATE ${CUTFEM_BUILD_DIR}/lib)
    target_link_libraries(growth_iterative_cutfem PRIVATE parallel common FESpace solver problem ${GMSH_LIBRARY})

    # growth_iterative_lowner_cutfem: Iterative ossification with Lowner ellipse front
    add_executable(growth_iterative_lowner_cutfem growth_iterative_lowner_cutfem.cpp)
    set_target_properties(growth_iterative_lowner_cutfem PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(growth_iterative_lowner_cutfem PRIVATE
      ${CUTFEM_DIR}
      ${CUTFEM_DIR}/cpp
      ${CUTFEM_DIR}/cpp/problem
      ${CUTFEM_DIR}/cpp/FESpace
      ${CUTFEM_DIR}/cpp/common
      ${CUTFEM_DIR}/cpp/solver
      ${CUTFEM_DIR}/cpp/parallel
      ${CUTFEM_DIR}/cpp/num
      ${CUTFEM_DIR}/cpp/algoim
      ${CUTFEM_DIR}/cpp/concept
      ${CUTFEM_BUILD_DIR}
      ${GMSH_INCLUDE_DIR}
    )
    target_link_directories(growth_iterative_lowner_cutfem PRIVATE ${CUTFEM_BUILD_DIR}/lib)
    target_link_libraries(growth_iterative_lowner_cutfem PRIVATE parallel common FESpace solver problem ${GMSH_LIBRARY})

  else()
    message(STATUS "Gmsh not found — skipping SOC_cutfem, growth_cutfem, growth_iterative_cutfem, growth_iterative_lowner_cutfem")
  endif()
else()
  message(STATUS "CutFEM-Library not found at ${CUTFEM_DIR} — skipping bulk, step85_cutfem, SOC_cutfem")
endif()
