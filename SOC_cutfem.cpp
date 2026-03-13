/**
 * @brief SOC problem implemented with CutFEM-Library.
 *
 * 2D plane-strain linear elasticity on an SOC-shaped domain defined by
 * a B-spline polygon (via gmsh). Bi-material with interface at y=1.0.
 *
 * - Bottom segment clamped via Nitsche (u=0)
 * - 5-step moving parabolic traction on the SOC boundary
 * - Post-processing: von Mises, octahedral shear, hydrostatic, Miner index
 * - Per-step + averaged VTK output with native cut-cell visualization
 */

#include "../cutfem.hpp"
#include <gmsh.h>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>

using mesh_t     = MeshQuad2;
using funtest_t  = TestFunction<mesh_t>;
using fct_t      = FunFEM<mesh_t>;
using cutmesh_t  = ActiveMesh<mesh_t>;
using space_t    = GFESpace<mesh_t>;
using cutspace_t = CutFESpace<mesh_t>;

// ============================================================
// Geometry parameters
// ============================================================
static constexpr double s = 1.0;   // scale factor
static constexpr double p1_geom = s * 0.9;
static constexpr double p2_geom = s * 0.2;

// Background mesh bounds
static constexpr double bg_xmin = s * -1.21;
static constexpr double bg_ymin = 0.0;
static constexpr double bg_xmax = s *  1.21;
static constexpr double bg_ymax = s * (2.41 + 0.5);

// Bottom clamp region
static constexpr double x_bottom_min = s * -0.5;
static constexpr double x_bottom_max = s *  0.5;
static constexpr double y_bottom     = 0.0;

// ============================================================
// Material (bi-material — plane strain, interface at y = 1.0)
// ============================================================
static constexpr double interfaceY = s * 1.0;

// Below interface (stiff substrate)
static constexpr double E_below  = 500.0;
static constexpr double nu_below = 0.2;
static constexpr double mu_below     = E_below / (2.0 * (1.0 + nu_below));
static constexpr double lambda_below = E_below * nu_below / ((1.0 + nu_below) * (1.0 - 2.0 * nu_below));

// Above interface (soft layer)
static constexpr double E_above  = 6.0;
static constexpr double nu_above = 0.47;
static constexpr double mu_above     = E_above / (2.0 * (1.0 + nu_above));
static constexpr double lambda_above = E_above * nu_above / ((1.0 + nu_above) * (1.0 - 2.0 * nu_above));

// ============================================================
// Load parameters
// ============================================================
static constexpr double load_center_u_frac = 0.50;
static constexpr double load_du_frac       = 0.08;
static constexpr double load_p_peak        = 1.0;

struct StepDef {
    double u_center;
    double u_radius;
    double p_peak;
};

// ============================================================
// Geometry helpers
// ============================================================

// Squared distance from point p to segment [a,b], returns alpha ∈ [0,1]
static double dist_point_segment_sq(const R2 &p, const R2 &a, const R2 &b, double &alpha_out) {
    const double abx = b.x - a.x, aby = b.y - a.y;
    const double apx = p.x - a.x, apy = p.y - a.y;
    const double dotAB = abx * abx + aby * aby;

    if (dotAB < 1e-30) {
        alpha_out = 0.0;
        return apx * apx + apy * apy;
    }

    double t = (apx * abx + apy * aby) / dotAB;
    t = std::max(0.0, std::min(1.0, t));
    alpha_out = t;

    const double qx = a.x + t * abx - p.x;
    const double qy = a.y + t * aby - p.y;
    return qx * qx + qy * qy;
}

// Ray-casting point-in-polygon test
static bool point_in_polygon(const R2 &p, const std::vector<R2> &poly) {
    bool inside = false;
    const size_t n = poly.size();
    if (n < 3) return false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const double xi = poly[i].x, yi = poly[i].y;
        const double xj = poly[j].x, yj = poly[j].y;
        if (((yi > p.y) != (yj > p.y)) &&
            (p.x < (xj - xi) * (p.y - yi) / (yj - yi + 1e-30) + xi))
            inside = !inside;
    }
    return inside;
}

// Signed distance to closed polygon (negative inside)
static double signed_distance_polygon(const R2 &p, const std::vector<R2> &poly) {
    const size_t n = poly.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    double best_d2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i < n; ++i) {
        double a = 0.0;
        const double d2 = dist_point_segment_sq(p, poly[i], poly[(i + 1) % n], a);
        best_d2 = std::min(best_d2, d2);
    }

    const double d = std::sqrt(std::max(0.0, best_d2));
    return point_in_polygon(p, poly) ? -d : +d;
}

// Storage for polygon + spline data (global for level-set functor)
static std::vector<R2> g_polygon;
static std::vector<R2> g_top_spline;
static std::vector<double> g_top_u;

// Level-set functor for CutFEM
double fun_levelSet(R2 P, const int i) {
    return signed_distance_polygon(P, g_polygon);
}

// Unit outward normal from polyline segment
static R2 unit_normal_from_spline(const std::vector<R2> &top, unsigned int seg) {
    if (top.size() < 2 || seg + 1 >= top.size()) return R2(0.0, 0.0);
    double tx = top[seg + 1].x - top[seg].x;
    double ty = top[seg + 1].y - top[seg].y;
    double tn = std::sqrt(tx * tx + ty * ty);
    if (tn < 1e-14) return R2(0.0, 0.0);
    tx /= tn; ty /= tn;
    // Normal = rotated tangent (outward = pointing up for the top curve)
    R2 n(-ty, tx);
    if (n.y < 0.0) { n.x = -n.x; n.y = -n.y; }
    return n;
}

// Project point onto polyline, return segment index, alpha, and u-coordinate
struct SegmentProjection {
    double dist2  = std::numeric_limits<double>::max();
    double alpha  = 0.0;
    unsigned int seg = 0;
    double u_along = 0.0;
};

static SegmentProjection project_to_polyline_with_u(
    const R2 &p, const std::vector<R2> &line, const std::vector<double> &u_nodes) {
    SegmentProjection best;
    if (line.size() < 2 || u_nodes.size() != line.size()) return best;
    for (unsigned int k = 0; k + 1 < line.size(); ++k) {
        double alpha = 0.0;
        double d2 = dist_point_segment_sq(p, line[k], line[k + 1], alpha);
        if (d2 < best.dist2) {
            best.dist2 = d2;
            best.seg = k;
            best.alpha = alpha;
            best.u_along = u_nodes[k] + alpha * (u_nodes[k + 1] - u_nodes[k]);
        }
    }
    return best;
}

// ============================================================
// Build SOC geometry via gmsh B-spline
// ============================================================
static void build_soc_geometry() {
    gmsh::initialize();
    gmsh::model::add("SOC_cutfem");

    // 9 control points for the SOC B-spline
    gmsh::model::occ::addPoint( s*0.5,  s*0.0,          0.0, 0.1, 1);
    gmsh::model::occ::addPoint( s*0.5,  s*1.0,          0.0, 0.1, 2);
    gmsh::model::occ::addPoint( s*1.0,  p1_geom,        0.0, 0.1, 3);
    gmsh::model::occ::addPoint( s*1.0,  s*2.4,          0.0, 0.1, 4);
    gmsh::model::occ::addPoint( s*0.0,  s*2.4 + p2_geom,0.0, 0.1, 5);
    gmsh::model::occ::addPoint(-s*1.0,  s*2.4,          0.0, 0.1, 6);
    gmsh::model::occ::addPoint(-s*1.0,  p1_geom,        0.0, 0.1, 7);
    gmsh::model::occ::addPoint(-s*0.5,  s*1.0,          0.0, 0.1, 8);
    gmsh::model::occ::addPoint(-s*0.5,  s*0.0,          0.0, 0.1, 9);

    gmsh::model::occ::addLine(9, 1, 1);  // bottom segment
    gmsh::model::occ::addBSpline({1,2,3,4,5,6,7,8,9}, 2, 3,
        {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0});
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::setTransfiniteCurve(2, 101);
    gmsh::model::mesh::generate(1);

    std::vector<std::size_t> node_tags;
    std::vector<double> coords;
    std::vector<double> param_coords;

    gmsh::model::mesh::getNodes(node_tags, coords, param_coords,
                                1, 2, true, true);

    if (node_tags.empty())
        throw std::runtime_error("Gmsh: getNodes failed on curve 2.");

    struct Node1D { std::size_t tag; double u; R2 p; };
    std::vector<Node1D> nodes;
    nodes.reserve(node_tags.size());
    for (size_t i = 0; i < node_tags.size(); ++i) {
        double x = coords[3*i], y = coords[3*i + 1];
        double u = param_coords.empty() ? (double)i : param_coords[i];
        nodes.push_back({node_tags[i], u, R2(x, y)});
    }
    std::sort(nodes.begin(), nodes.end(),
              [](const Node1D &a, const Node1D &b) { return a.u < b.u; });

    g_top_spline.clear();
    g_top_u.clear();
    std::unordered_set<std::size_t> seen;
    for (const auto &n : nodes) {
        if (!seen.insert(n.tag).second) continue;
        g_top_spline.push_back(n.p);
        g_top_u.push_back(n.u);
    }

    // Ensure start near right endpoint (s*0.5, 0)
    const R2 right_ref(s*0.5, 0.0);
    auto d_front = std::sqrt((g_top_spline.front().x - right_ref.x) * (g_top_spline.front().x - right_ref.x)
                           + (g_top_spline.front().y - right_ref.y) * (g_top_spline.front().y - right_ref.y));
    auto d_back  = std::sqrt((g_top_spline.back().x - right_ref.x) * (g_top_spline.back().x - right_ref.x)
                           + (g_top_spline.back().y - right_ref.y) * (g_top_spline.back().y - right_ref.y));
    if (d_front > d_back) {
        std::reverse(g_top_spline.begin(), g_top_spline.end());
        std::reverse(g_top_u.begin(), g_top_u.end());
    }

    // Build closed polygon: bottom left -> right (line 1), then spline interior nodes
    const R2 left(-s*0.5, 0.0);
    const R2 right(s*0.5, 0.0);
    g_polygon.clear();
    g_polygon.push_back(left);
    g_polygon.push_back(right);
    for (unsigned int i = 1; i + 1 < g_top_spline.size(); ++i)
        g_polygon.push_back(g_top_spline[i]);

    std::cout << "Top curve nodes: " << g_top_spline.size() << "\n";
    std::cout << "Polygon vertices: " << g_polygon.size() << "\n";
    std::cout << "top_u range: [" << g_top_u.front() << ", " << g_top_u.back() << "]\n";

    gmsh::finalize();
}

// ============================================================
// Build load steps
// ============================================================
static std::vector<StepDef> build_default_steps() {
    const double u0 = g_top_u.front();
    const double u1 = g_top_u.back();
    const double ur = u1 - u0;
    const double uc = u0 + load_center_u_frac * ur;
    const double du = load_du_frac * ur;
    const double overlap_radius = du;

    std::vector<StepDef> steps;
    steps.push_back({uc + 2.0*du, overlap_radius, 0.5  * load_p_peak});
    steps.push_back({uc + du,     overlap_radius, 0.75 * load_p_peak});
    steps.push_back({uc,          overlap_radius, 1.0  * load_p_peak});
    steps.push_back({uc - du,     overlap_radius, 0.75 * load_p_peak});
    steps.push_back({uc - 2.0*du, overlap_radius, 0.5  * load_p_peak});

    for (auto &st : steps)
        st.u_center = std::min(std::max(st.u_center, u0), u1);
    return steps;
}

// ============================================================
// Stress invariants (plane strain)
// ============================================================
struct Invariants {
    double von_mises = 0.0;
    double hydrostatic = 0.0;
    double oct_shear = 0.0;
    double miner = 0.0;
};

static Invariants compute_invariants_plane_strain(
    double eps_xx, double eps_yy, double eps_xy,
    double lam, double mu, double kMi = 0.5) {

    // Stress in 2D
    double trace_eps = eps_xx + eps_yy;
    double s_xx = 2.0 * mu * eps_xx + lam * trace_eps;
    double s_yy = 2.0 * mu * eps_yy + lam * trace_eps;
    double s_xy = 2.0 * mu * eps_xy;
    // Out-of-plane (plane strain): sigma_zz = lambda * tr(eps)
    double s_zz = lam * trace_eps;

    // Hydrostatic
    double hydro = (s_xx + s_yy + s_zz) / 3.0;

    // Deviatoric
    double d_xx = s_xx - hydro;
    double d_yy = s_yy - hydro;
    double d_zz = s_zz - hydro;

    double s_contract = d_xx*d_xx + d_yy*d_yy + d_zz*d_zz + 2.0*s_xy*s_xy;
    double vm = std::sqrt(1.5 * s_contract);
    double oct = std::sqrt(2.0 / 3.0) * vm;

    Invariants inv;
    inv.von_mises = vm;
    inv.hydrostatic = hydro;
    inv.oct_shear = oct;
    inv.miner = oct + kMi * hydro;
    return inv;
}

// ============================================================
// Assemble traction RHS for one load step
// ============================================================
// Since CutFEM-Library doesn't have a direct "integrate custom traction on interface"
// we'll manually assemble the traction by iterating over interface quadrature points.
// We use the interface object's integration and project each quad point onto the spline.
//
// Actually, CutFEM addLinear on "interface" integrates on the level-set zero contour.
// The problem is we need to apply traction only on PART of the boundary (the top curve,
// not the bottom segment), and the traction depends on the parametric u-coordinate.
//
// Strategy: Use a FunFEM for traction components, but we need to define traction as a
// function of space. We'll create traction component functions that:
// - Project x,y to the nearest spline segment
// - Compute the parabolic pressure and normal at that point
// - Return the traction component (tx or ty)
// Then use addLinear(innerProduct(traction, v), interface)

static double g_step_u_center, g_step_u_radius, g_step_p_peak;

// Traction function: returns the i-th component of the traction vector
// t(x) = -p(u(x)) * n(x) where p(u) = p_peak * max(0, 1 - ((u - u_c)/u_r)^2)
double fun_traction(R2 P, int i, int dom) {
    if (g_top_spline.size() < 2) return 0.0;

    const auto proj = project_to_polyline_with_u(P, g_top_spline, g_top_u);
    if (proj.seg + 1 >= g_top_spline.size()) return 0.0;

    // Check if within active u-window
    const double R = std::max(1e-14, g_step_u_radius);
    const double ua = g_step_u_center - R;
    const double ub = g_step_u_center + R;

    const double us0 = g_top_u[proj.seg];
    const double us1 = g_top_u[proj.seg + 1];
    const double useg_min = std::min(us0, us1);
    const double useg_max = std::max(us0, us1);
    if (useg_max < ua || useg_min > ub) return 0.0;

    const double tq0 = (us0 - g_step_u_center) / R;
    const double tq1 = (us1 - g_step_u_center) / R;
    const double tq  = tq0 + proj.alpha * (tq1 - tq0);
    if (std::abs(tq) >= 1.0) return 0.0;

    const double p = g_step_p_peak * (1.0 - tq * tq);
    R2 n = unit_normal_from_spline(g_top_spline, proj.seg);
    double tn = std::sqrt(n.x*n.x + n.y*n.y);
    if (tn < 1e-14) return 0.0;

    // Traction = -p * n
    if (i == 0) return -p * n.x;
    else return -p * n.y;
}

// Zero Dirichlet for bottom clamp
double fun_zero(R2 P, int i, int dom) { return 0.0; }

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {

    MPIcf cfMPI(argc, argv);
    globalVariable::verbose = 0;

    std::filesystem::create_directories("output/SOC_cutfem");

    // Build SOC geometry
    build_soc_geometry();

    // ---- Background mesh ----
    const int nx = 60;
    const double dx_bg = (bg_xmax - bg_xmin);
    const double dy_bg = (bg_ymax - bg_ymin);
    const int ny = static_cast<int>(nx * dy_bg / dx_bg);
    mesh_t Kh(nx, ny, bg_xmin + 0.00137, bg_ymin - 0.00113, dx_bg, dy_bg);
    const double h = dx_bg / (nx - 1);

    std::cout << "Background mesh: " << nx << " x " << ny
              << ", h = " << h << "\n";

    // ---- Level set ----
    space_t Lh(Kh, DataFE<mesh_t>::P1);
    fct_t levelSet(Lh, fun_levelSet);

    InterfaceLevelSet<mesh_t> interface(Kh, levelSet);

    // Active mesh: remove exterior (level set > 0 = outside polygon)
    cutmesh_t Khi(Kh);
    Khi.truncate(interface, 1);

    // ---- FE spaces ----
    // Vector space for displacement (Q1, 2 components)
    LagrangeQuad2 FE_vec(1);
    space_t Uh(Kh, FE_vec);
    cutspace_t Wh(Khi, Uh);

    int nb_dof = Wh.get_nb_dof();
    std::cout << "Displacement DOFs: " << nb_dof << "\n";

    Normal n;

    // Nitsche and ghost parameters (use stiff material for penalty)
    const double nitsche_penalty = 20.0 * (2.0 * mu_below + lambda_below) * 4.0;
    const double ghost_param = 0.5;

    // Bottom indicator: 1 on the bottom segment (y≈0, x∈[-0.5,0.5]), 0 on top spline
    auto bottom_ind_fun = [](R2 P, int i, int dom) -> double {
        const double tol = 0.05;
        if (std::abs(P.y - y_bottom) < tol &&
            P.x >= x_bottom_min - tol &&
            P.x <= x_bottom_max + tol)
            return 1.0;
        return 0.0;
    };
    space_t Sh(Kh, DataFE<mesh_t>::P1);
    fct_t bottomInd(Sh, bottom_ind_fun);

    // Material coefficient FunFEMs (bi-material)
    auto two_mu_fun = [](R2 P, int i, int dom) -> double {
        return 2.0 * ((P.y < interfaceY) ? mu_below : mu_above);
    };
    fct_t two_mu_fh(Sh, two_mu_fun);

    auto lambda_fun = [](R2 P, int i, int dom) -> double {
        return (P.y < interfaceY) ? lambda_below : lambda_above;
    };
    fct_t lambda_fh(Sh, lambda_fun);

    // ---- Load steps ----
    auto steps = build_default_steps();
    const size_t nSteps = steps.size();

    std::vector<double> U_sum(nb_dof, 0.0);

    for (unsigned int sidx = 0; sidx < nSteps; ++sidx) {
        std::cout << "\n=== Step " << (sidx + 1) << "/" << nSteps
                  << " (u_center=" << steps[sidx].u_center
                  << ", u_radius=" << steps[sidx].u_radius
                  << ", p_peak=" << steps[sidx].p_peak << ") ===\n";

        // Set traction parameters for this step
        g_step_u_center = steps[sidx].u_center;
        g_step_u_radius = steps[sidx].u_radius;
        g_step_p_peak   = steps[sidx].p_peak;

        // Build problem
        CutFEM<mesh_t> problem(Wh);
        funtest_t u(Wh, 2, 0), v(Wh, 2, 0);

        // Bulk: ∫ 2μ(x) ε(u):ε(v) + λ(x) div(u) div(v)  [bi-material]
        problem.addBilinear(
            contractProduct(two_mu_fh.expr() * Eps(u), Eps(v))
            + innerProduct(lambda_fh.expr() * div(u), div(v)),
            Khi);

        // Nitsche u=0 on BOTTOM segment only (penalty method, weighted by bottomInd)
        problem.addBilinear(
            innerProduct(bottomInd.expr() * (nitsche_penalty / h) * u, v),
            interface);

        // Ghost penalty
        problem.addFaceStabilization(
            +innerProduct(ghost_param * pow(h, -1) * jump(u), jump(v))
            +innerProduct(ghost_param * pow(h, 1) * jump(grad(u) * n), jump(grad(v) * n)),
            Khi);

        // Traction RHS on interface
        fct_t tractionFh(Uh, fun_traction);
        problem.addLinear(
            innerProduct(tractionFh.exprList(), v),
            interface);

        // Solve
        problem.solve("umfpack");

        // Extract solution
        std::span<double> data_uh(problem.rhs_.data(), nb_dof);
        std::vector<double> sol_data(data_uh.begin(), data_uh.end());

        // Accumulate for averaging
        for (int i = 0; i < nb_dof; ++i)
            U_sum[i] += sol_data[i];

        // Write per-step VTK
        {
            std::span<double> sol_span(sol_data);
            fct_t uh(Wh, sol_span);

            std::string fname = "output/SOC_cutfem/SOC_cutfem_step_" +
                                std::to_string(sidx + 1) + ".vtk";
            Paraview<mesh_t> writer(Khi, fname);
            writer.add(uh, "displacement", 0, 2);
        }

        std::cout << "Step " << (sidx + 1) << " output written.\n";
    }

    // ---- Averaged output ----
    {
        double invN = 1.0 / nSteps;
        std::vector<double> U_avg(nb_dof);
        for (int i = 0; i < nb_dof; ++i)
            U_avg[i] = U_sum[i] * invN;

        std::span<double> avg_span(U_avg);
        fct_t uh_avg(Wh, avg_span);

        Paraview<mesh_t> writer(Khi, "output/SOC_cutfem/SOC_cutfem_averaged.vtk");
        writer.add(uh_avg, "displacement", 0, 2);
    }

    std::cout << "\nDone. Output in output/SOC_cutfem/\n";
    return 0;
}
