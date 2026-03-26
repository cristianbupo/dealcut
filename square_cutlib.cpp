/**
 * @brief Bimaterial linear elasticity on unit square [0,1]x[0,1].
 *
 * Simplified reference problem using CutFEM-Library (no Gmsh).
 * The domain boundary is defined by a level set on a slightly
 * larger background mesh, so the outer edges are cut.
 *
 * - Plane strain
 * - Bottom (y=0): clamped via Nitsche penalty
 * - Top (y=1): parabolic Neumann load p(x) = p_peak * 4x(1-x)
 * - Material interface at y=0.5 (aligned with mesh, no cut)
 *   Below: E=500, nu=0.2 (bone)
 *   Above: E=6, nu=0.47 (cartilage)
 * - Single solve (UMFPACK)
 * - Output: displacement + stress invariants (von Mises, hydrostatic,
 *   octahedral shear, Miner index)
 */

#include "../cutfem.hpp"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>

using mesh_t     = MeshQuad2;
using funtest_t  = TestFunction<mesh_t>;
using fct_t      = FunFEM<mesh_t>;
using cutmesh_t  = ActiveMesh<mesh_t>;
using space_t    = GFESpace<mesh_t>;
using cutspace_t = CutFESpace<mesh_t>;

// ============================================================
// Material (plane strain)
// ============================================================
static constexpr double E_bone  = 500.0, nu_bone = 0.2;
static constexpr double mu_bone     = E_bone / (2.0 * (1.0 + nu_bone));
static constexpr double lambda_bone = E_bone * nu_bone / ((1.0 + nu_bone) * (1.0 - 2.0 * nu_bone));

static constexpr double E_cart  = 6.0, nu_cart = 0.47;
static constexpr double mu_cart     = E_cart / (2.0 * (1.0 + nu_cart));
static constexpr double lambda_cart = E_cart * nu_cart / ((1.0 + nu_cart) * (1.0 - 2.0 * nu_cart));

static constexpr double interface_y = 0.5;
static constexpr double p_peak = 1.0;
static constexpr double k_mi   = 0.5;

// ============================================================
// Level set: signed distance to unit square [0,1]x[0,1]
// Negative inside, positive outside.
// ============================================================
double fun_levelSet(R2 P, const int /*i*/)
{
    const double dx = std::max(-P.x, P.x - 1.0);
    const double dy = std::max(-P.y, P.y - 1.0);

    if (dx <= 0.0 && dy <= 0.0)
        return std::max(dx, dy); // inside: negative
    if (dx > 0.0 && dy > 0.0)
        return std::sqrt(dx * dx + dy * dy); // outside corner
    return std::max(dx, dy); // outside edge
}

// ============================================================
// Traction on boundary: parabolic on top (y≈1), zero elsewhere
// t = (0, -p_peak * 4x(1-x)) applied as surface integral
// ============================================================
double fun_traction(R2 P, int i, int /*dom*/)
{
    // Only apply on the top edge
    if (std::abs(P.y - 1.0) > 0.05) return 0.0;
    if (P.x < -0.01 || P.x > 1.01)  return 0.0;

    const double p = p_peak * 4.0 * P.x * (1.0 - P.x);
    // Outward normal on top is (0, 1) → traction = (0, -p)
    return (i == 0) ? 0.0 : -p;
}

// ============================================================
// Stress invariants (plane strain)
// ============================================================
struct Invariants
{
    double von_mises, hydrostatic, oct_shear, miner;
};

static Invariants compute_invariants(double exx, double eyy, double exy,
                                     double lam, double mu)
{
    const double tr  = exx + eyy;
    const double sxx = 2.0 * mu * exx + lam * tr;
    const double syy = 2.0 * mu * eyy + lam * tr;
    const double sxy = 2.0 * mu * exy;
    const double szz = lam * tr;

    const double hd  = (sxx + syy + szz) / 3.0;
    const double dxx = sxx - hd, dyy = syy - hd, dzz = szz - hd;
    const double J2  = dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * sxy * sxy;
    const double vm  = std::sqrt(1.5 * J2);
    const double oct = std::sqrt(2.0 / 3.0) * vm;

    return {vm, hd, oct, oct + k_mi * std::min(hd, 0.0)};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    MPIcf cfMPI(argc, argv);
    globalVariable::verbose = 0;

    std::filesystem::create_directories("output/square_cutlib");

    // ---- Background mesh ----
    // Slightly larger than [0,1]x[0,1] so boundaries don't align with nodes.
    // With nx=41 (40 intervals) on [-0.02, 1.02]:
    //   h = 1.04/40 = 0.026
    //   y=0.5 at index 20: -0.02 + 20*0.026 = 0.50 (aligned for material)
    //   y=0.0 at 0.02/0.026 ≈ 0.77 (not aligned, good for cut)
    //   y=1.0 at 1.02/0.026 ≈ 39.23 (not aligned, good for cut)
    const int    nx      = 41;
    const double bg_xmin = -0.02, bg_xmax = 1.02;
    const double bg_ymin = -0.02, bg_ymax = 1.02;
    const double dx_bg   = bg_xmax - bg_xmin;
    const double dy_bg   = bg_ymax - bg_ymin;

    mesh_t Kh(nx, nx, bg_xmin, bg_ymin, dx_bg, dy_bg);
    const double h = dx_bg / (nx - 1);

    std::cout << "Background mesh: " << nx << "x" << nx
              << ", h=" << h << "\n";

    // ---- Level set & active mesh ----
    space_t Lh(Kh, DataFE<mesh_t>::P1);
    fct_t   levelSet(Lh, fun_levelSet);

    InterfaceLevelSet<mesh_t> interface(Kh, levelSet);

    cutmesh_t Khi(Kh);
    Khi.truncate(interface, 1);

    // ---- FE spaces ----
    LagrangeQuad2 FE_vec(1);
    space_t       Uh(Kh, FE_vec);
    cutspace_t    Wh(Khi, Uh);

    const int nb_dof = Wh.get_nb_dof();

    // Scalar P1 space (for post-processing)
    space_t Sh(Kh, DataFE<mesh_t>::P1);
    const int nb_sca = Sh.NbDoF();

    std::cout << "Displacement DOFs: " << nb_dof
              << ", scalar DOFs: " << nb_sca << "\n";

    Normal n;
    const double nitsche_penalty = 20.0 * (2.0 * mu_bone + lambda_bone) * 4.0;
    const double ghost_param     = 0.5;

    // ---- Bottom indicator (for Nitsche clamp) ----
    fct_t bottomInd(Sh, [](R2 P, int /*i*/, int /*dom*/) -> double {
        return (std::abs(P.y) < 0.05 && P.x >= -0.01 && P.x <= 1.01) ? 1.0 : 0.0;
    });

    // ---- Material FunFEMs ----
    fct_t two_mu_fh(Sh, [](R2 P, int, int) -> double {
        return 2.0 * ((P.y < interface_y) ? mu_bone : mu_cart);
    });

    fct_t lambda_fh(Sh, [](R2 P, int, int) -> double {
        return (P.y < interface_y) ? lambda_bone : lambda_cart;
    });

    // ---- Assemble & solve ----
    CutFEM<mesh_t> problem(Wh);
    funtest_t u(Wh, 2, 0), v(Wh, 2, 0);

    // Bulk: ∫ 2μ ε(u):ε(v) + λ div(u) div(v)
    problem.addBilinear(
        contractProduct(two_mu_fh.expr() * Eps(u), Eps(v))
        + innerProduct(lambda_fh.expr() * div(u), div(v)),
        Khi);

    // Nitsche clamp on bottom (penalty only, via bottomInd)
    problem.addBilinear(
        innerProduct(bottomInd.expr() * (nitsche_penalty / h) * u, v),
        interface);

    // Ghost penalty (face stabilization)
    problem.addFaceStabilization(
        innerProduct(ghost_param * pow(h, -1) * jump(u), jump(v))
        + innerProduct(ghost_param * pow(h, 1) * jump(grad(u) * n), jump(grad(v) * n)),
        Khi);

    // Traction RHS (parabolic load on top edge)
    fct_t tractionFh(Uh, fun_traction);
    problem.addLinear(innerProduct(tractionFh.exprList(), v), interface);

    problem.solve("umfpack");

    // ---- Extract solution ----
    std::span<double>  data_uh(problem.rhs_.data(), nb_dof);
    std::vector<double> sol(data_uh.begin(), data_uh.end());
    std::span<double>  sol_span(sol);
    fct_t uh(Wh, sol_span);

    std::cout << "Solved.\n";

    // ---- Post-process: nodal stress invariants ----
    // Area-weighted averaging from element centroids to nodes
    std::vector<double> vm_data(nb_sca, 0.0), hd_data(nb_sca, 0.0),
                        oct_data(nb_sca, 0.0), mi_data(nb_sca, 0.0),
                        wt_data(nb_sca, 0.0);

    const int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        if (Khi.isInactive(ka, 0)) continue;

        const int kb  = Khi.idxElementInBackMesh(ka);
        const auto &FK = Sh[kb];
        const int  ndf = FK.NbDoF();

        // Centroid
        R2 centroid(0.0, 0.0);
        for (int j = 0; j < ndf; ++j) {
            centroid.x += FK.Pt(j).x;
            centroid.y += FK.Pt(j).y;
        }
        centroid.x /= ndf;
        centroid.y /= ndf;

        // Displacement gradients at centroid
        const double du0_dx = uh.eval(ka, (const double *)&centroid, 0, 1);
        const double du0_dy = uh.eval(ka, (const double *)&centroid, 0, 2);
        const double du1_dx = uh.eval(ka, (const double *)&centroid, 1, 1);
        const double du1_dy = uh.eval(ka, (const double *)&centroid, 1, 2);

        const double exx = du0_dx;
        const double eyy = du1_dy;
        const double exy = 0.5 * (du0_dy + du1_dx);

        // Material at centroid
        const double lam = (centroid.y < interface_y) ? lambda_bone : lambda_cart;
        const double mu  = (centroid.y < interface_y) ? mu_bone     : mu_cart;

        const auto inv = compute_invariants(exx, eyy, exy, lam, mu);

        // Element area
        const R2 P0 = FK.Pt(0), P2 = FK.Pt(2);
        const double elem_area = std::abs((P2.x - P0.x) * (P2.y - P0.y));

        for (int j = 0; j < ndf; ++j) {
            const int iglo = Sh(kb, j);
            if (iglo < 0 || iglo >= nb_sca) continue;
            vm_data[iglo]  += inv.von_mises * elem_area;
            hd_data[iglo]  += inv.hydrostatic * elem_area;
            oct_data[iglo] += inv.oct_shear * elem_area;
            mi_data[iglo]  += inv.miner * elem_area;
            wt_data[iglo]  += elem_area;
        }
    }

    for (int i = 0; i < nb_sca; ++i) {
        if (wt_data[i] > 0.0) {
            const double inv_w = 1.0 / wt_data[i];
            vm_data[i]  *= inv_w;
            hd_data[i]  *= inv_w;
            oct_data[i] *= inv_w;
            mi_data[i]  *= inv_w;
        }
    }

    // Material indicator
    std::vector<double> mat_data(nb_sca);
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        for (int j = 0; j < FK.NbDoF(); ++j) {
            const int iglo = Sh(k, j);
            if (iglo >= 0 && iglo < nb_sca)
                mat_data[iglo] = (FK.Pt(j).y < interface_y) ? 1.0 : 0.0;
        }
    }

    // Wrap as FunFEMs
    std::span<double> vm_span(vm_data), hd_span(hd_data),
                      oct_span(oct_data), mi_span(mi_data),
                      mat_span(mat_data);

    fct_t vm_fh(Sh, vm_span);
    fct_t hd_fh(Sh, hd_span);
    fct_t oct_fh(Sh, oct_span);
    fct_t mi_fh(Sh, mi_span);
    fct_t mat_fh(Sh, mat_span);

    // ---- VTK output ----
    {
        Paraview<mesh_t> writer(Khi, "output/square_cutlib/square_cutlib.vtk");
        writer.add(uh, "displacement", 0, 2);
        writer.add(vm_fh, "von_mises", 0, 1);
        writer.add(hd_fh, "hydrostatic", 0, 1);
        writer.add(oct_fh, "oct_shear", 0, 1);
        writer.add(mi_fh, "miner_index", 0, 1);
        writer.add(mat_fh, "material", 0, 1);
    }

    std::cout << "Output: output/square_cutlib/square_cutlib.vtk\n";
    return 0;
}
