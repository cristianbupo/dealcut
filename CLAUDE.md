# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

C++ numerical simulation executables for iterative bone ossification growth using the Cut Finite Element Method (CutFEM) on unfitted meshes. Couples mechanical stress analysis (elasticity) with morphogen transport (reaction-diffusion). Not a library — each `.cc`/`.cpp` file produces a standalone executable.

## Build Commands

```bash
# Configure (from repo root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# If dependency discovery fails
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/dealii/lib/cmake/deal.II \
  -DCUTFEM_DIR=$HOME/git/CutFEM-Library

# Build all
cmake --build build -j

# Build single target (preferred during iteration)
cmake --build build --target growth_iterative_cutfem -j

# Clean
cmake --build build --target clean
```

## Test / Validate

No formal ctest suite. Validate by building and running the affected executable:

```bash
cmake --build build --target growth_iterative_cutfem -j
./build/growth_iterative_cutfem configs/growth_iterative_convex.json
```

## Lint / Format

No repo-enforced config. For touched files only:

```bash
clang-format -i <changed-files>
clang-tidy <file.cpp> -p build
```

Do not run whole-repo reformatting.

## Architecture

### Executables

| Target | Source | Framework | Purpose |
|--------|--------|-----------|---------|
| `growth_iterative_cutfem` | `growth_iterative_cutfem.cpp` | CutFEM-Library + Gmsh | **Main solver** — iterative ossification with MI/Poisson phase scheduling |
| `growth_iterative_lowner_cutfem` | `growth_iterative_lowner_cutfem.cpp` | CutFEM-Library + Gmsh | Löwner ellipse-based front approximation |
| `growth_iterative_nurbs_cutfem` | `growth_iterative_nurbs_cutfem.cpp` | CutFEM-Library + Gmsh | NURBS spline-based front representation |
| `growth` | `growth.cc` | deal.II + VTK + Gmsh | Non-cut FEM growth |
| `SOC` / `SOC_cutfem` | `SOC.cc` / `SOC_cutfem.cpp` | deal.II / CutFEM | SOC elasticity |
| `step-85` / `step85_cutfem` | `step-85.cc` / `step85_cutfem.cpp` | deal.II / CutFEM | Tutorial Laplace/Poisson |
| `growth_cutfem` | `growth_cutfem.cpp` | CutFEM-Library + Gmsh | Single-pass CutFEM growth |
| `bulk` | `bulk.cpp` | CutFEM-Library | Time-dependent convection-diffusion |

### Two-Phase Iterative Growth (core algorithm)

The main solver (`growth_iterative_cutfem`) alternates between two phases per `build_phase_schedule`:
- **MI phase**: Solve elasticity on current ossification domain → compute Carter stress (MI = τ_oct + k_mi·min(σ_h, 0)) → threshold to identify new ossified cells
- **Poisson phase**: Solve reaction-diffusion (−∇²u + αu = f) on cartilage → use isocontour as alternative front

Schedule: iterations 0–1 are MI (mechanical seed), then alternating POISSON/MI.

### Geometry & Level Sets

- SOC boundary: B-spline curve (9 control points, transfinite discretization via Gmsh)
- Material interface: horizontal line at `y = interface_y`
- Level-set functions (`phi_outer`, `phi_interface`, `phi_soc`) define cut boundaries
- Ossification grows monotonically: `phi_soc^{n+1} = max(phi_soc^n, phi_new)`

### CutFEM Type Aliases

```cpp
using mesh_t     = MeshQuad2;          // Quad background mesh
using cutmesh_t  = ActiveMesh<mesh_t>; // Trimmed mesh
using fct_t      = FunFEM<mesh_t>;     // FE function
using space_t    = GFESpace<mesh_t>;   // Standard FE space
using cutspace_t = CutFESpace<mesh_t>; // Cut FE space
```

### Configuration

JSON configs in `configs/` parsed by a custom regex-based parser (no external JSON lib). Global config struct `SimConfig g_cfg` loaded via `load_config_from_json()`. Key parameters: geometry (B-spline weights, interface position), loading (Gaussian pressure profile), mesh (nx, y_offset), MI/Poisson thresholds, boundary condition flags.

### Dependencies

- **deal.II 9.7.0** — FEM framework (non-CutFEM variants)
- **CutFEM-Library** — header-only CutFEM (expected at `$HOME/git/CutFEM-Library`)
- **Gmsh** — geometry/meshing
- **VTK** — visualization output
- **C++20** required for CutFEM targets

### Output

VTK files written to `output/<config_name>/` (git-ignored). Cell-based fields: material, MI, Poisson concentration. Trimmed-domain export includes only active cells.

## Code Style

- Follow existing conventions in each file (mixed-origin codebase)
- Types: `PascalCase`. Functions/variables: `snake_case`
- Keep `using namespace dealii;` scoped inside namespaces
- Do not add global `using namespace` directives
- Group external includes before standard library includes
- Guard normalization/division with tolerances
- Do not silently change physical constants or stabilization parameters (e.g., Nitsche penalty γ = 20·(2μ+λ)·4)

## CMake Rules

- New executables go in `CMakeLists.txt` with explicit link/include setup
- Keep optional targets behind dependency checks (`deal.II`, `VTK`, `Gmsh`, `CutFEM`)
- Prefer single-target rebuilds during development
