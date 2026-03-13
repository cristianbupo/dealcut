# AGENTS.md
Repository guidance for autonomous coding agents.

## 1) Project Overview
- Language: C++ (`.cc` and `.cpp`).
- Build system: CMake (out-of-source build in `build/`).
- Focus: numerical simulation executables, not a packaged library.
- Key dependencies: deal.II, VTK, Gmsh, CutFEM-Library, MPI wrappers.

## 2) Important Paths
- `CMakeLists.txt`: all target definitions.
- `step-85.cc`: deal.II tutorial-based Laplace/CutFEM flow.
- `SOC.cc`, `growth.cc`: deal.II + Gmsh + VTK pipelines.
- `bulk.cpp`, `*_cutfem.cpp`: CutFEM-Library variants.
- `build/`: generated artifacts and simulation outputs.
- Never manually edit generated files under `build/`.

## 3) Configure and Build
Run from repo root.

### Configure (Release)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

If dependency discovery fails, pass explicit paths:
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/dealii/lib/cmake/deal.II \
  -DCUTFEM_DIR=$HOME/git/CutFEM-Library
```

### Build all
```bash
cmake --build build -j
```

### Build a single target (preferred during iteration)
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

### Clean
```bash
cmake --build build --target clean
```

## 4) Run Commands
```bash
./build/step-85
./build/SOC
./build/growth
./build/bulk
./build/step85_cutfem
./build/SOC_cutfem
./build/growth_cutfem export
./build/growth_cutfem import
./build/growth_iterative_cutfem
./build/growth_iterative_cutfem configs/growth_iterative_default.json
./build/growth_iterative_cutfem configs/growth_iterative_soc12.json
```
- Most outputs go under `output/*`.

## 5) Test Commands
- Current state: no `enable_testing()` / `add_test()` in `CMakeLists.txt`.
- Full test run (when tests are added):
```bash
ctest --test-dir build --output-on-failure
```
- Single test (important):
```bash
ctest --test-dir build -R "<test-name-or-substring>" --output-on-failure
```
- List tests only:
```bash
ctest --test-dir build -N
```
- Until formal tests exist, validate by building and running only the affected executable.

## 6) Lint / Format / Static Analysis
- No repo-enforced lint/format config found (`.clang-format`, `.clang-tidy`, CI lint).
- Recommended local checks for touched files:
```bash
clang-format -i <changed-files>
clang-tidy <file.cpp> -p build
```
- Do not run whole-repo reformatting.

## 7) Code Style Rules
Follow existing style in each touched file; this codebase is mixed-origin.

### Includes
- Keep external/library includes grouped before standard library includes.
- In deal.II files, preserve module-style include grouping.
- Add only required includes.

### Formatting
- Preserve indentation and brace style already used in the file.
- Keep long FEM expressions readable; break at logical boundaries.
- Avoid unrelated whitespace churn.

### Namespaces and scope
- Keep code in existing namespaces (`Step85`, `SOC`, etc.).
- `using namespace dealii;` exists inside namespaces; keep it scoped.
- Do not add new global `using namespace ...` directives.

### Types and const-correctness
- Prefer explicit numeric types (`double`, `std::size_t`, `unsigned int`).
- Use `const` for immutable values and parameters where practical.
- Match existing deal.II templates (`template <int dim>`).

### Naming
- Types/structs/classes: `PascalCase`.
- Functions/variables: keep local convention (mostly `snake_case`).
- Constants: descriptive names; avoid cryptic abbreviations unless domain-standard.

### Error handling
- Use clear `std::runtime_error` messages for invalid runtime states.
- Preserve existing `Assert*` patterns where deal.II assertions are used.
- Avoid swallowing errors unless cleanup requires guarded fallback.

### Logging and diagnostics
- Keep progress output concise (`std::cout`).
- Use `std::cerr` for user-visible failure paths.

### Numerical safety
- Guard normalization/division with tolerances.
- Validate geometry/projection assumptions before use.
- Do not silently change physical constants or stabilization parameters.

## 8) CMake Change Rules
- Add new executables in `CMakeLists.txt` with explicit link/include setup.
- Keep optional targets behind dependency checks (`deal.II`, `VTK`, `Gmsh`, `CutFEM`).
- If adding tests, also add `enable_testing()` and `add_test(...)`.

## 9) Agent Working Agreement
- Make minimal, task-focused diffs.
- Prefer single-target rebuilds before full rebuilds.
- If build/test binaries are unavailable in the current environment, still provide exact commands for local execution.

## 10) Cursor and Copilot Rules Status
Checked locations:
- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`
Result: no Cursor or Copilot instruction files were found at those paths.
