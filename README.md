# dealcut

Numerical simulation executables built with deal.II and CutFEM-Library.

## Prerequisites

- CMake >= 3.16
- deal.II (9.7.0 expected by `CMakeLists.txt`)
- Gmsh (headers + library)
- VTK
- CutFEM-Library (source + built libs)

## Configure

From repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

If dependency discovery fails:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/dealii/lib/cmake/deal.II \
  -DCUTFEM_DIR=$HOME/git/CutFEM-Library
```

## Build

Build everything:

```bash
cmake --build build -j
```

Build one target:

```bash
cmake --build build --target step-85 -j
cmake --build build --target SOC -j
cmake --build build --target growth -j
cmake --build build --target bulk -j
cmake --build build --target step85_cutfem -j
cmake --build build --target SOC_cutfem -j
cmake --build build --target growth_cutfem -j
cmake --build build --target growth_iterative_cutfem -j
```

## Run

```bash
./build/step-85
./build/SOC
./build/growth
./build/bulk
./build/step85_cutfem
./build/SOC_cutfem
./build/growth_cutfem export
./build/growth_cutfem import
```

Iterative CutFEM growth now uses a single executable with optional JSON config:

```bash
./build/growth_iterative_cutfem
./build/growth_iterative_cutfem configs/growth_iterative_default.json
./build/growth_iterative_cutfem configs/growth_iterative_soc12.json
```

Outputs go to the folder set by `output_dir` in the JSON config.

## Config files for iterative growth

- `configs/growth_iterative_default.json`: baseline setup.
- `configs/growth_iterative_soc12.json`: former `growth_iterative_12_cutfem` setup.

## Quick validation

There is no formal `ctest` suite yet. Validate changes by:

```bash
cmake --build build --target growth_iterative_cutfem -j
./build/growth_iterative_cutfem configs/growth_iterative_default.json
./build/growth_iterative_cutfem configs/growth_iterative_soc12.json
```
