set(TARGET "step-90-revolution")
set(TARGET_SRC ${TARGET}.cc)
set(CLEAN_UP_FILES *.vtu *.csv)

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate a (sufficiently recent) version of deal.II. "
                      "Pass -DDEAL_II_DIR=/path/to/deal.II or set DEAL_II_DIR.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})

find_path(GMSH_INCLUDE_DIR gmsh.h
  HINTS /usr/include /usr/local/include
)
find_library(GMSH_LIBRARY gmsh
  HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)
if(NOT GMSH_INCLUDE_DIR OR NOT GMSH_LIBRARY)
  message(FATAL_ERROR "Could not find gmsh (headers at ${GMSH_INCLUDE_DIR}, "
                      "library at ${GMSH_LIBRARY}).")
endif()

add_executable(${TARGET} ${TARGET_SRC})
deal_ii_setup_target(${TARGET})
target_include_directories(${TARGET} PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(${TARGET} ${GMSH_LIBRARY})
