/**
 * @brief Step-85 equivalent using CutFEM-Library.
 *
 * Solves the Poisson equation on a unit disk Ω using CutFEM with Nitsche's
 * method for boundary conditions and ghost penalty stabilization.
 *
 * Problem:
 *   -Δu = 4        in Ω = { x ∈ R² : |x| < 1 }
 *    u  = g(x)     on Γ = ∂Ω
 *
 * where g(x) = 1 - (2/dim)(|x|² - 1), so the exact solution is
 *   u(x) = 1 - (2/dim)(|x|² - 1).
 *
 * This matches the deal.II step-85 tutorial problem.
 */

#include "../cutfem.hpp"
#include <filesystem>
#include <iomanip>
#include <cmath>

using mesh_t     = Mesh2;
using funtest_t  = TestFunction<mesh_t>;
using fct_t      = FunFEM<mesh_t>;
using cutmesh_t  = ActiveMesh<mesh_t>;
using space_t    = GFESpace<mesh_t>;
using cutspace_t = CutFESpace<mesh_t>;

// Level-set function: phi(x) = |x|² - 1  (negative inside the unit disk)
double fun_levelSet(R2 P, const int i) {
    return P.x * P.x + P.y * P.y - 1.0;
}

// Right-hand side: f = 4 (for dim = 2)
double fun_rhs(R2 P, int ci, int dom) {
    return 4.0;
}

// Exact solution: u(x) = 1 - (|x|² - 1) = 2 - |x|²
double fun_exact(R2 P, int ci, int dom) {
    return 1.0 - (P.x * P.x + P.y * P.y - 1.0);
}

// Dirichlet boundary condition (= exact solution on Γ, where |x|=1 → g=1)
double fun_dirichlet(R2 P, int ci, int dom) {
    return 1.0;
}

int main(int argc, char **argv) {

    MPIcf cfMPI(argc, argv);
    globalVariable::verbose = 0;

    const int n_refinements = 6;
    int nx = 9;  // initial mesh: starts at 2^3+1 like deal.II (refine_global(2) on [-1.21,1.21])

    std::vector<double> L2_errors, H1_errors, hs;

    std::filesystem::create_directories("output_step85_cutfem");

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Step-85 CutFEM-Library: Poisson on unit disk\n";
    std::cout << std::string(60, '=') << "\n\n";

    for (int cycle = 0; cycle < n_refinements; ++cycle) {

        std::cout << "Refinement cycle " << cycle << "\n";

        // Background mesh covering [-1.21, 1.21]²
        const double lo = -1.21;
        const double side = 2.42;
        mesh_t Kh(nx, nx, lo, lo, side, side);
        const double h = side / (nx - 1);

        // Level-set space and function
        space_t Lh(Kh, DataFE<mesh_t>::P1);
        fct_t levelSet(Lh, fun_levelSet);

        // Interface (boundary of the disk)
        InterfaceLevelSet<mesh_t> interface(Kh, levelSet);

        // Active mesh: keep elements where level set < 0 (inside disk)
        cutmesh_t Khi(Kh);
        Khi.truncate(interface, 1);

        // Solution FE space (P1)
        space_t Vh(Kh, DataFE<mesh_t>::P1);
        cutspace_t Wh(Khi, Vh);

        std::cout << "  h = " << h << ", DOFs = " << Wh.get_nb_dof() << "\n";

        // Set up the CutFEM problem
        CutFEM<mesh_t> poisson(Wh);

        Normal n;
        funtest_t u(Wh, 1), v(Wh, 1);

        // Nitsche parameters
        const double nitsche_parameter = 5.0 * 2.0 * 1.0;  // 5 * (p+1) * p for p=1
        const double ghost_parameter   = 0.5;

        // ---- Bilinear form ----

        // Bulk: ∫_Ω ∇u · ∇v dx
        poisson.addBilinear(
            innerProduct(grad(u), grad(v)),
            Khi);

        // Nitsche (boundary): -∫_Γ (n·∇u)v - ∫_Γ u(n·∇v) + (σ/h)∫_Γ uv
        poisson.addBilinear(
            -innerProduct(grad(u) * n, v)
            -innerProduct(u, grad(v) * n)
            +innerProduct((nitsche_parameter / h) * u, v),
            interface);

        // Ghost penalty: ½ γ h ∫_F [n·∇u][n·∇v]
        poisson.addFaceStabilization(
            +innerProduct(ghost_parameter * h * jump(grad(u) * n), jump(grad(v) * n)),
            Khi);

        // ---- Linear form ----

        // Interpolate RHS
        fct_t fRhs(Wh, fun_rhs);

        // Bulk: ∫_Ω f v dx
        poisson.addLinear(
            innerProduct(fRhs.expr(), v),
            Khi);

        // Nitsche RHS: -∫_Γ g(n·∇v) + (σ/h)∫_Γ g v
        fct_t gD(Wh, fun_dirichlet);
        poisson.addLinear(
            -innerProduct(gD.expr(), grad(v) * n)
            +innerProduct((nitsche_parameter / h) * gD.expr(), v),
            interface);

        // ---- Solve ----
        poisson.solve("umfpack");

        // ---- Extract solution ----
        std::span<double> data_uh(poisson.rhs_.data(), Wh.get_nb_dof());
        fct_t uh(Wh, data_uh);

        // ---- Compute errors ----
        double L2_error = L2normCut(uh, fun_exact, 0, 1);

        // H1 semi-norm error: ||∇(u - u_ex)||
        // Exact gradient: ∇u = (-2x, -2y)
        auto uh_dx = dx(uh.expr(0));
        auto uh_dy = dy(uh.expr(0));

        fct_t u_ex(Wh, fun_exact);
        auto uex_dx = dx(u_ex.expr(0));
        auto uex_dy = dy(u_ex.expr(0));

        double H1_seminorm_sq = integral(Khi,
            (uh_dx - uex_dx) * (uh_dx - uex_dx) +
            (uh_dy - uex_dy) * (uh_dy - uex_dy), 0);
        double H1_error = std::sqrt(H1_seminorm_sq);

        L2_errors.push_back(L2_error);
        H1_errors.push_back(H1_error);
        hs.push_back(h);

        std::cout << "  L2 error = " << std::scientific << std::setprecision(6) << L2_error << "\n";
        std::cout << "  H1 error = " << std::scientific << std::setprecision(6) << H1_error << "\n";

        if (cycle > 0) {
            double L2_rate = std::log2(L2_errors[cycle - 1] / L2_errors[cycle]);
            double H1_rate = std::log2(H1_errors[cycle - 1] / H1_errors[cycle]);
            std::cout << "  L2 rate  = " << std::fixed << std::setprecision(2) << L2_rate << "\n";
            std::cout << "  H1 rate  = " << std::fixed << std::setprecision(2) << H1_rate << "\n";
        }

        // ---- Paraview output ----
        {
            Paraview<mesh_t> writer(Khi,
                "output_step85_cutfem/step85_cutfem_cycle" + std::to_string(cycle) + ".vtk");
            writer.add(uh, "solution", 0, 1);
            writer.add(u_ex, "exact_solution", 0, 1);
        }

        std::cout << "\n";

        // Refine: double the number of cells in each direction
        nx = 2 * (nx - 1) + 1;
    }

    // ---- Summary ----
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Convergence summary\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(6) << "Cycle"
              << std::setw(14) << "h"
              << std::setw(14) << "L2-Error"
              << std::setw(10) << "L2-Rate"
              << std::setw(14) << "H1-Error"
              << std::setw(10) << "H1-Rate" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 0; i < n_refinements; ++i) {
        std::cout << std::setw(6) << i
                  << std::setw(14) << std::scientific << std::setprecision(4) << hs[i]
                  << std::setw(14) << L2_errors[i];
        if (i > 0)
            std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                      << std::log2(L2_errors[i - 1] / L2_errors[i]);
        else
            std::cout << std::setw(10) << "---";
        std::cout << std::setw(14) << std::scientific << std::setprecision(4) << H1_errors[i];
        if (i > 0)
            std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                      << std::log2(H1_errors[i - 1] / H1_errors[i]);
        else
            std::cout << std::setw(10) << "---";
        std::cout << "\n";
    }
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
