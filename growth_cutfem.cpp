/**
 * @brief Growth/ossification with CutFEM-Library.
 *
 * Two-step workflow on SOC domain:
 *   Step 1 (export): bi-material elasticity → Miner index → threshold → save
 *   Step 2 (import): load threshold → ossification level set → 3-material solve
 *                    → export cartilage region with cut-cell visualization
 *
 * Materials:
 *   bone      (y < 1.0):  E=500,  ν=0.2
 *   cartilage (y ≥ 1.0):  E=6,    ν=0.47
 *   ossified  (mi ≥ thr): E=253,  ν=0.335  (average of bone+cartilage)
 */

#include "../cutfem.hpp"
#include <gmsh.h>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <fstream>
#include <iomanip>
#include <functional>

using mesh_t     = MeshQuad2;
using funtest_t  = TestFunction<mesh_t>;
using fct_t      = FunFEM<mesh_t>;
using cutmesh_t  = ActiveMesh<mesh_t>;
using space_t    = GFESpace<mesh_t>;
using cutspace_t = CutFESpace<mesh_t>;

// ============================================================
// Run mode
// ============================================================
enum class RunMode { export_mi, import_mi };

// Change this to switch between step 1 and step 2
// run_mode is set from command line: ./growth_cutfem export|import
static constexpr unsigned int iteration_id = 0;
static constexpr unsigned int import_iteration_id = 0;

// ============================================================
// Geometry
// ============================================================
static constexpr double s = 1.0;
static constexpr double p1_geom = s * 0.9;
static constexpr double p2_geom = s * 0.2;

static constexpr double bg_xmin = s * -1.21;
static constexpr double bg_ymin = 0.0;
static constexpr double bg_xmax = s *  1.21;
static constexpr double bg_ymax = s * (2.41 + 0.5);

static constexpr double x_bottom_min = s * -0.5;
static constexpr double x_bottom_max = s *  0.5;
static constexpr double y_bottom     = 0.0;

// ============================================================
// Material (plane strain)
// ============================================================
static constexpr double interfaceY = s * 1.0;

// Bone (below interface)
static constexpr double E_bone  = 500.0, nu_bone = 0.2;
static constexpr double mu_bone     = E_bone / (2.0 * (1.0 + nu_bone));
static constexpr double lambda_bone = E_bone * nu_bone / ((1.0 + nu_bone) * (1.0 - 2.0 * nu_bone));

// Cartilage (above interface)
static constexpr double E_cart  = 6.0, nu_cart = 0.47;
static constexpr double mu_cart     = E_cart / (2.0 * (1.0 + nu_cart));
static constexpr double lambda_cart = E_cart * nu_cart / ((1.0 + nu_cart) * (1.0 - 2.0 * nu_cart));

// Ossified (average bone+cartilage)
static constexpr double E_oss  = 0.5 * (E_bone + E_cart);
static constexpr double nu_oss = 0.5 * (nu_bone + nu_cart);
static constexpr double mu_oss     = E_oss / (2.0 * (1.0 + nu_oss));
static constexpr double lambda_oss = E_oss * nu_oss / ((1.0 + nu_oss) * (1.0 - 2.0 * nu_oss));

// Ossification parameters
static constexpr double oss_height_target = 0.1; // approximate target height of ossified region
static constexpr double kMi = 0.5;
static constexpr double oss_spline_band = 0.3;    // distance over which ossification is inhibited near spline

// ============================================================
// Load
// ============================================================
static constexpr double load_center_u_frac = 0.50;
static constexpr double load_du_frac       = 0.08;
static constexpr double load_p_peak        = 1.0;

struct StepDef { double u_center, u_radius, p_peak; };

// ============================================================
// Globals
// ============================================================
static std::vector<R2> g_polygon;
static std::vector<R2> g_top_spline;
static std::vector<double> g_top_u;
static double g_step_u_center, g_step_u_radius, g_step_p_peak;

// Miner index fields (filled after step 1 or loaded from restart)
static std::vector<double> g_mi_avg;
static double g_mi_threshold = std::numeric_limits<double>::quiet_NaN();

// ============================================================
// Geometry helpers
// ============================================================
static double dist_point_segment_sq(const R2 &p, const R2 &a, const R2 &b, double &alpha_out) {
    const double abx = b.x - a.x, aby = b.y - a.y;
    const double apx = p.x - a.x, apy = p.y - a.y;
    const double dotAB = abx * abx + aby * aby;
    if (dotAB < 1e-30) { alpha_out = 0.0; return apx*apx + apy*apy; }
    double t = std::clamp((apx*abx + apy*aby) / dotAB, 0.0, 1.0);
    alpha_out = t;
    double qx = a.x + t*abx - p.x, qy = a.y + t*aby - p.y;
    return qx*qx + qy*qy;
}

static bool point_in_polygon(const R2 &p, const std::vector<R2> &poly) {
    bool inside = false;
    for (size_t i = 0, j = poly.size()-1; i < poly.size(); j = i++)
        if (((poly[i].y > p.y) != (poly[j].y > p.y)) &&
            (p.x < (poly[j].x - poly[i].x) * (p.y - poly[i].y) / (poly[j].y - poly[i].y + 1e-30) + poly[i].x))
            inside = !inside;
    return inside;
}

static double signed_distance_polygon(const R2 &p, const std::vector<R2> &poly) {
    double best_d2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i < poly.size(); ++i) {
        double a; double d2 = dist_point_segment_sq(p, poly[i], poly[(i+1)%poly.size()], a);
        best_d2 = std::min(best_d2, d2);
    }
    double d = std::sqrt(std::max(0.0, best_d2));
    return point_in_polygon(p, poly) ? -d : +d;
}

double fun_levelSet(R2 P, const int i) { return signed_distance_polygon(P, g_polygon); }

static R2 unit_normal_from_spline(const std::vector<R2> &top, unsigned int seg) {
    if (top.size() < 2 || seg+1 >= top.size()) return R2(0,0);
    double tx = top[seg+1].x - top[seg].x, ty = top[seg+1].y - top[seg].y;
    double tn = std::sqrt(tx*tx + ty*ty);
    if (tn < 1e-14) return R2(0,0);
    tx /= tn; ty /= tn;
    R2 n(-ty, tx);
    if (n.y < 0.0) { n.x = -n.x; n.y = -n.y; }
    return n;
}

struct SegmentProjection { double dist2 = 1e30, alpha = 0; unsigned int seg = 0; double u_along = 0; };

static SegmentProjection project_to_polyline_with_u(
    const R2 &p, const std::vector<R2> &line, const std::vector<double> &u_nodes) {
    SegmentProjection best;
    for (unsigned int k = 0; k+1 < line.size(); ++k) {
        double alpha; double d2 = dist_point_segment_sq(p, line[k], line[k+1], alpha);
        if (d2 < best.dist2) { best = {d2, alpha, k, u_nodes[k] + alpha*(u_nodes[k+1]-u_nodes[k])}; }
    }
    return best;
}

// ============================================================
// Gmsh geometry
// ============================================================
static void build_soc_geometry() {
    gmsh::initialize();
    gmsh::model::add("growth_cutfem");

    gmsh::model::occ::addPoint( s*0.5,  0.0,          0, 0.1, 1);
    gmsh::model::occ::addPoint( s*0.5,  s*1.0,        0, 0.1, 2);
    gmsh::model::occ::addPoint( s*1.0,  p1_geom,      0, 0.1, 3);
    gmsh::model::occ::addPoint( s*1.0,  s*2.4,        0, 0.1, 4);
    gmsh::model::occ::addPoint( 0.0,    s*2.4+p2_geom,0, 0.1, 5);
    gmsh::model::occ::addPoint(-s*1.0,  s*2.4,        0, 0.1, 6);
    gmsh::model::occ::addPoint(-s*1.0,  p1_geom,      0, 0.1, 7);
    gmsh::model::occ::addPoint(-s*0.5,  s*1.0,        0, 0.1, 8);
    gmsh::model::occ::addPoint(-s*0.5,  0.0,          0, 0.1, 9);

    gmsh::model::occ::addLine(9, 1, 1);
    gmsh::model::occ::addBSpline({1,2,3,4,5,6,7,8,9}, 2, 3,
        {1,2,1,1,1,1,1,2,1});
    gmsh::model::occ::synchronize();
    gmsh::model::mesh::setTransfiniteCurve(2, 101);
    gmsh::model::mesh::generate(1);

    std::vector<std::size_t> tags; std::vector<double> coords, params;
    gmsh::model::mesh::getNodes(tags, coords, params, 1, 2, true, true);

    struct N1D { std::size_t t; double u; R2 p; };
    std::vector<N1D> nodes;
    for (size_t i = 0; i < tags.size(); ++i)
        nodes.push_back({tags[i], params[i], R2(coords[3*i], coords[3*i+1])});
    std::sort(nodes.begin(), nodes.end(), [](auto &a, auto &b){return a.u < b.u;});

    g_top_spline.clear(); g_top_u.clear();
    std::unordered_set<std::size_t> seen;
    for (auto &n : nodes) { if (seen.insert(n.t).second) { g_top_spline.push_back(n.p); g_top_u.push_back(n.u); } }

    R2 rref(s*0.5, 0);
    auto df = std::hypot(g_top_spline.front().x-rref.x, g_top_spline.front().y-rref.y);
    auto db = std::hypot(g_top_spline.back().x-rref.x, g_top_spline.back().y-rref.y);
    if (df > db) { std::reverse(g_top_spline.begin(), g_top_spline.end()); std::reverse(g_top_u.begin(), g_top_u.end()); }

    g_polygon.clear();
    g_polygon.push_back(R2(-s*0.5, 0)); g_polygon.push_back(R2(s*0.5, 0));
    for (unsigned int i = 1; i+1 < g_top_spline.size(); ++i) g_polygon.push_back(g_top_spline[i]);

    gmsh::finalize();
}

// ============================================================
// Load steps
// ============================================================
static std::vector<StepDef> build_default_steps() {
    double u0 = g_top_u.front(), u1 = g_top_u.back(), ur = u1-u0;
    double uc = u0 + load_center_u_frac*ur, du = load_du_frac*ur;
    std::vector<StepDef> steps = {
        {uc+2*du, du, 0.50*load_p_peak}, {uc+du, du, 0.75*load_p_peak},
        {uc,      du, 1.00*load_p_peak}, {uc-du, du, 0.75*load_p_peak},
        {uc-2*du, du, 0.50*load_p_peak}
    };
    for (auto &st : steps) st.u_center = std::clamp(st.u_center, u0, u1);
    return steps;
}

// ============================================================
// Traction
// ============================================================
double fun_traction(R2 P, int i, int dom) {
    if (g_top_spline.size() < 2) return 0.0;
    auto proj = project_to_polyline_with_u(P, g_top_spline, g_top_u);
    if (proj.seg+1 >= g_top_spline.size()) return 0.0;
    double R = std::max(1e-14, g_step_u_radius);
    double ua = g_step_u_center-R, ub = g_step_u_center+R;
    double us0 = g_top_u[proj.seg], us1 = g_top_u[proj.seg+1];
    if (std::max(us0,us1) < ua || std::min(us0,us1) > ub) return 0.0;
    double tq = ((us0-g_step_u_center)/R) + proj.alpha*((us1-us0)/R);
    if (std::abs(tq) >= 1.0) return 0.0;
    double p = g_step_p_peak * (1.0 - tq*tq);
    R2 n = unit_normal_from_spline(g_top_spline, proj.seg);
    return (i == 0) ? -p*n.x : -p*n.y;
}

// ============================================================
// Stress invariants (plane strain)
// ============================================================
struct Invariants { double von_mises, hydrostatic, oct_shear, miner; };

static Invariants compute_invariants(double exx, double eyy, double exy, double lam, double mu) {
    double tr = exx + eyy;
    double sxx = 2*mu*exx + lam*tr, syy = 2*mu*eyy + lam*tr, sxy = 2*mu*exy;
    double szz = lam*tr;
    double hd = (sxx + syy + szz) / 3.0;
    double dxx = sxx-hd, dyy = syy-hd, dzz = szz-hd;
    double J2 = dxx*dxx + dyy*dyy + dzz*dzz + 2.0*sxy*sxy;
    double vm = std::sqrt(1.5 * J2);
    double oct = std::sqrt(2.0/3.0) * vm;
    return {vm, hd, oct, oct + kMi*std::min(hd, 0.0)};
}

// ============================================================
// Algoim level-set for per-element material interface
// ============================================================
// Bilinear interpolation of (interfaceY - P.y) on an axis-aligned quad.
// phi < 0 ↔ cartilage (above interface), phi > 0 ↔ bone (below interface).
struct ElementInterfaceLS {
    double xmin, xmax, ymin, ymax;
    double v00, v10, v01, v11;
    double t = 0.0; // unused, required by Algoim interface

    template <typename V>
    typename V::value_type operator()(const V &P) const {
        auto s = (P[0] - xmin) / (xmax - xmin);
        auto t_ = (P[1] - ymin) / (ymax - ymin);
        return (1 - s) * (1 - t_) * v00 + s * (1 - t_) * v10
             + (1 - s) * t_ * v01       + s * t_ * v11;
    }

    template <typename T>
    algoim::uvector<T, 2> grad(const algoim::uvector<T, 2> &x) const {
        T s  = (x(0) - xmin) / (xmax - xmin);
        T t_ = (x(1) - ymin) / (ymax - ymin);
        T dfdx = ((1 - t_) * (v10 - v00) + t_ * (v11 - v01)) / (xmax - xmin);
        T dfdy = ((1 - s)  * (v01 - v00) + s  * (v11 - v10)) / (ymax - ymin);
        return algoim::uvector<T, 2>(dfdx, dfdy);
    }
};

// ============================================================
// Stress fields: hydrostatic, octahedral shear, Miner index
// ============================================================
struct StressFields {
    std::vector<double> hydrostatic, oct_shear, miner;
};

// Compute stress fields at nodes using Algoim quadrature on cut elements.
// Uncut cartilage: centroid evaluation. Cut elements (straddling bone-cartilage
// interface): Algoim quadrature on the cartilage sub-domain with pure cartilage
// material. Fully bone elements: skipped.
static StressFields compute_stress_fields(
    const fct_t &uh, const space_t &Sh,
    const cutmesh_t &Khi, const mesh_t &Kh, int nb_sca_dof)
{
    static constexpr int algoim_order = 3;

    std::vector<double> hd_sum(nb_sca_dof, 0.0), oct_sum(nb_sca_dof, 0.0), mi_sum(nb_sca_dof, 0.0);
    std::vector<double> wt_sum(nb_sca_dof, 0.0);

    int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        int kb = Khi.idxElementInBackMesh(ka);
        const auto &FK = Sh[kb];
        int ndf = FK.NbDoF();

        // Gather node positions and classify material side
        R2 pts[4];
        bool has_bone = false, has_cart = false;
        for (int j = 0; j < ndf; ++j) {
            pts[j] = FK.Pt(j);
            if (pts[j].y < interfaceY) has_bone = true;
            else                        has_cart = true;
        }

        if (!has_cart) continue; // fully bone → skip

        if (!has_bone) {
            // ---- Fully cartilage: centroid evaluation ----
            R2 centroid(0.0, 0.0);
            for (int j = 0; j < ndf; ++j) {
                centroid.x += pts[j].x;
                centroid.y += pts[j].y;
            }
            centroid.x /= ndf;
            centroid.y /= ndf;

            double du0_dx = uh.eval(ka, (const double*)&centroid, 0, 1);
            double du0_dy = uh.eval(ka, (const double*)&centroid, 0, 2);
            double du1_dx = uh.eval(ka, (const double*)&centroid, 1, 1);
            double du1_dy = uh.eval(ka, (const double*)&centroid, 1, 2);

            double exx = du0_dx, eyy = du1_dy;
            double exy = 0.5 * (du0_dy + du1_dx);

            auto inv = compute_invariants(exx, eyy, exy, lambda_cart, mu_cart);

            double elem_area = (pts[1].x - pts[0].x) * (pts[3].y - pts[0].y);
            for (int j = 0; j < ndf; ++j) {
                int iglo = Sh(kb, j);
                if (iglo < 0 || iglo >= nb_sca_dof) continue;
                hd_sum[iglo]  += inv.hydrostatic * elem_area;
                oct_sum[iglo] += inv.oct_shear   * elem_area;
                mi_sum[iglo]  += inv.miner       * elem_area;
                wt_sum[iglo]  += elem_area;
            }
        } else {
            // ---- Cut element: Algoim quadrature on cartilage sub-domain ----
            double xmin = pts[0].x, ymin = pts[0].y;
            double xmax = pts[2].x, ymax = pts[2].y;

            // phi = interfaceY - P.y : negative above interface (cartilage)
            ElementInterfaceLS phi_iface;
            phi_iface.xmin = xmin; phi_iface.xmax = xmax;
            phi_iface.ymin = ymin; phi_iface.ymax = ymax;
            phi_iface.v00 = interfaceY - pts[0].y;
            phi_iface.v10 = interfaceY - pts[1].y;
            phi_iface.v11 = interfaceY - pts[2].y;
            phi_iface.v01 = interfaceY - pts[3].y;

            algoim::QuadratureRule<2> q = algoim::quadGen<2>(
                phi_iface,
                algoim::HyperRectangle<double, 2>(
                    algoim::uvector<double, 2>{xmin, ymin},
                    algoim::uvector<double, 2>{xmax, ymax}),
                -1, -1, algoim_order);

            if (q.nodes.empty()) continue;

            // Accumulate area-weighted stress over cartilage quadrature points
            double hd_acc = 0, oct_acc = 0, mi_acc = 0, w_acc = 0;
            for (size_t iq = 0; iq < q.nodes.size(); ++iq) {
                R2 mip(q.nodes[iq].x(0), q.nodes[iq].x(1));
                double w = q.nodes[iq].w;

                double du0_dx = uh.eval(ka, (const double*)&mip, 0, 1);
                double du0_dy = uh.eval(ka, (const double*)&mip, 0, 2);
                double du1_dx = uh.eval(ka, (const double*)&mip, 1, 1);
                double du1_dy = uh.eval(ka, (const double*)&mip, 1, 2);

                double exx = du0_dx, eyy = du1_dy;
                double exy = 0.5 * (du0_dy + du1_dx);

                auto inv = compute_invariants(exx, eyy, exy, lambda_cart, mu_cart);
                hd_acc  += inv.hydrostatic * w;
                oct_acc += inv.oct_shear   * w;
                mi_acc  += inv.miner       * w;
                w_acc   += w;
            }

            if (w_acc <= 0) continue;

            // Distribute to cartilage-side nodes only
            for (int j = 0; j < ndf; ++j) {
                if (pts[j].y < interfaceY) continue; // bone node
                int iglo = Sh(kb, j);
                if (iglo < 0 || iglo >= nb_sca_dof) continue;
                hd_sum[iglo]  += hd_acc;
                oct_sum[iglo] += oct_acc;
                mi_sum[iglo]  += mi_acc;
                wt_sum[iglo]  += w_acc;
            }
        }
    }

    StressFields sf;
    sf.hydrostatic.resize(nb_sca_dof, 0.0);
    sf.oct_shear.resize(nb_sca_dof, 0.0);
    sf.miner.resize(nb_sca_dof, 0.0);
    for (int i = 0; i < nb_sca_dof; ++i) {
        if (wt_sum[i] > 0) {
            double inv = 1.0 / wt_sum[i];
            sf.hydrostatic[i] = hd_sum[i] * inv;
            sf.oct_shear[i]   = oct_sum[i] * inv;
            sf.miner[i]       = mi_sum[i] * inv;
        }
    }
    return sf;
}

// ============================================================
// Area-based threshold bisection
// ============================================================
// Compute ossified area for a given threshold value:
// count area of elements above interface where average mi ≥ threshold.
// This is approximate (element-level), but fine for the bisection.
static double compute_oss_area(const std::vector<double> &mi_nodal,
                               const space_t &Sh, const mesh_t &Kh,
                               double threshold) {
    double area = 0.0;
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        // Check if element is in the cartilage region (above interface)
        double y_avg = 0.0;
        int ndf = FK.NbDoF();
        double mi_avg = 0.0;
        for (int j = 0; j < ndf; ++j) {
            R2 P = FK.Pt(j);
            y_avg += P.y;
            int iglo = Sh(k, j);
            if (iglo >= 0 && iglo < (int)mi_nodal.size())
                mi_avg += mi_nodal[iglo];
        }
        y_avg /= ndf;
        mi_avg /= ndf;

        if (y_avg < interfaceY) continue; // skip bone region
        if (mi_avg < threshold) continue; // below threshold

        // Approximate element area
        R2 P0 = FK.Pt(0), P1 = FK.Pt(1), P2 = FK.Pt(2);
        // For quads, area ≈ |diagonal cross product|
        if (ndf >= 4) {
            R2 P3 = FK.Pt(3);
            area += 0.5 * std::abs((P2.x-P0.x)*(P3.y-P1.y) - (P3.x-P1.x)*(P2.y-P0.y));
        } else {
            area += 0.5 * std::abs((P1.x-P0.x)*(P2.y-P0.y) - (P2.x-P0.x)*(P1.y-P0.y));
        }
    }
    return area;
}

static double compute_cartilage_area(const space_t &Sh, const mesh_t &Kh) {
    double area = 0.0;
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        double y_avg = 0.0;
        int ndf = FK.NbDoF();
        for (int j = 0; j < ndf; ++j) y_avg += FK.Pt(j).y;
        y_avg /= ndf;
        if (y_avg < interfaceY) continue;
        R2 P0 = FK.Pt(0), P1 = FK.Pt(1), P2 = FK.Pt(2);
        if (ndf >= 4) {
            R2 P3 = FK.Pt(3);
            area += 0.5 * std::abs((P2.x-P0.x)*(P3.y-P1.y) - (P3.x-P1.x)*(P2.y-P0.y));
        } else {
            area += 0.5 * std::abs((P1.x-P0.x)*(P2.y-P0.y) - (P2.x-P0.x)*(P1.y-P0.y));
        }
    }
    return area;
}

// Build sorted (y, mi) profile along symmetry axis (x=0).
// For each element crossing x=0 above interfaceY, interpolate MI at x=0
// using bilinear shape functions, and record (y_center, mi_interp).
static std::vector<std::pair<double,double>> build_axis_profile(
    const std::vector<double> &mi_nodal,
    const space_t &Sh, const mesh_t &Kh)
{
    std::vector<std::pair<double,double>> profile;
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        int ndf = FK.NbDoF();
        double xmin_e = 1e30, xmax_e = -1e30;
        double ymin_e = 1e30, ymax_e = -1e30;
        for (int j = 0; j < ndf; ++j) {
            R2 P = FK.Pt(j);
            xmin_e = std::min(xmin_e, P.x);
            xmax_e = std::max(xmax_e, P.x);
            ymin_e = std::min(ymin_e, P.y);
            ymax_e = std::max(ymax_e, P.y);
        }
        if (xmin_e > 0.0 || xmax_e < 0.0) continue;
        if (ymax_e < interfaceY) continue;

        // Bilinear interpolation at x=0: get (xi, eta) in ref element
        double hx = xmax_e - xmin_e, hy = ymax_e - ymin_e;
        if (hx < 1e-14 || hy < 1e-14) continue;
        double xi = (0.0 - xmin_e) / hx; // in [0,1]

        // Sample at bottom and top of element (eta=0 and eta=1)
        // Q1 quad nodes: 0=(xmin,ymin), 1=(xmax,ymin), 2=(xmax,ymax), 3=(xmin,ymax)
        double mi0=-1e30, mi1=-1e30, mi2=-1e30, mi3=-1e30;
        for (int j = 0; j < ndf; ++j) {
            int iglo = Sh(k, j);
            if (iglo < 0 || iglo >= (int)mi_nodal.size()) continue;
            R2 P = FK.Pt(j);
            bool left  = (P.x - xmin_e) < 0.5*hx;
            bool bottom = (P.y - ymin_e) < 0.5*hy;
            if (left && bottom)       mi0 = mi_nodal[iglo];
            else if (!left && bottom) mi1 = mi_nodal[iglo];
            else if (!left && !bottom)mi2 = mi_nodal[iglo];
            else                      mi3 = mi_nodal[iglo];
        }
        // Interpolated MI at (xi, eta) for several eta values
        int nsamp = 5;
        for (int s = 0; s <= nsamp; ++s) {
            double eta = (double)s / nsamp;
            double y_s = ymin_e + eta * hy;
            if (y_s < interfaceY) continue;
            double mi_s = (1-xi)*(1-eta)*mi0 + xi*(1-eta)*mi1
                        + xi*eta*mi2 + (1-xi)*eta*mi3;
            profile.push_back({y_s, mi_s});
        }
    }
    // Sort by y ascending
    std::sort(profile.begin(), profile.end());
    return profile;
}

// Compute total ossified depth along symmetry axis for a given threshold.
// Sums the y-intervals where consecutive profile points both have MI >= threshold.
// Monotonically decreasing with threshold → well-conditioned for root-finding.
static double compute_oss_height_axis(const std::vector<std::pair<double,double>> &profile,
                                      double threshold) {
    double total = 0.0;
    for (size_t i = 0; i + 1 < profile.size(); ++i) {
        if (profile[i].second >= threshold && profile[i+1].second >= threshold)
            total += profile[i+1].first - profile[i].first;
    }
    return total;
}

static double find_threshold(const std::vector<double> &mi_nodal,
                             const space_t &Sh, const mesh_t &Kh) {
    double cart_area = compute_cartilage_area(Sh, Kh);

    // Build MI profile along x=0
    auto profile = build_axis_profile(mi_nodal, Sh, Kh);

    // Find MI range from profile
    double mi_min_ax = 1e30, mi_max_ax = -1e30;
    for (auto &[y, mi] : profile) {
        mi_min_ax = std::min(mi_min_ax, mi);
        mi_max_ax = std::max(mi_max_ax, mi);
    }

    // Regula falsi (Illinois variant):
    // f(threshold) = compute_oss_height_axis(threshold) - target
    // We want f(threshold) = 0.
    // At a (low threshold) → height is large → f(a) > 0
    // At b (high threshold) → height is small → f(b) < 0
    double a = mi_min_ax, b = mi_max_ax;
    double fa = compute_oss_height_axis(profile, a) - oss_height_target;
    double fb = compute_oss_height_axis(profile, b) - oss_height_target;

    double threshold = 0.5*(a+b);
    for (int iter = 0; iter < 100; ++iter) {
        if (std::abs(fa - fb) < 1e-15) break;
        double c = (a*fb - b*fa) / (fb - fa); // regula falsi
        c = std::clamp(c, mi_min_ax, mi_max_ax);
        double fc = compute_oss_height_axis(profile, c) - oss_height_target;
        if (std::abs(fc) < 1e-12) { threshold = c; break; }

        if (fc * fa > 0) {
            a = c; fa = fc;
            fb *= 0.5; // Illinois modification to avoid stagnation
        } else {
            b = c; fb = fc;
            fa *= 0.5;
        }
        threshold = c;
    }

    double achieved_height = compute_oss_height_axis(profile, threshold);
    double achieved_area   = compute_oss_area(mi_nodal, Sh, Kh, threshold);
    std::cout << "Ossification: target_h(axis)=" << oss_height_target
              << ", achieved_h(axis)=" << achieved_height
              << ", area=" << achieved_area << "/" << cart_area
              << ", threshold=" << threshold
              << ", mi range(axis)=[" << mi_min_ax << ", " << mi_max_ax << "]\n";

    return threshold;
}

// ============================================================
// Restart file I/O
// ============================================================
static std::string restart_filename(unsigned int id) {
    return "output/growth_cutfem/growth_cutfem_restart_" + std::to_string(id) + ".dat";
}

static void save_restart(const std::vector<double> &mi_avg, double threshold, unsigned int id) {
    std::ofstream out(restart_filename(id));
    out << std::setprecision(17);
    out << "GROWTH_CUTFEM_V1\n";
    out << mi_avg.size() << "\n";
    out << threshold << "\n";
    for (double v : mi_avg) out << v << "\n";
    std::cout << "Saved restart: " << restart_filename(id) << "\n";
}

static void load_restart(std::vector<double> &mi_avg, double &threshold, unsigned int id) {
    std::ifstream in(restart_filename(id));
    if (!in.is_open()) throw std::runtime_error("Cannot open restart: " + restart_filename(id));
    std::string header; std::getline(in, header);
    if (header != "GROWTH_CUTFEM_V1") throw std::runtime_error("Bad restart format");
    size_t n; in >> n;
    in >> threshold;
    mi_avg.resize(n);
    for (size_t i = 0; i < n; ++i) in >> mi_avg[i];
    std::cout << "Loaded restart: " << restart_filename(id)
              << " (n=" << n << ", threshold=" << threshold << ")\n";
}

// ============================================================
// Both-sides-trimmed VTK export
// ============================================================
// Writes active elements with cells split by multiple level sets.
// Each level set clips every polygon into its positive and negative
// sub-polygon, so both sides of every interface are preserved.
struct ScalarField { fct_t *fh; std::string name; };
struct CellFn { std::function<double(const R2&, int kb)> fn; std::string name; };
using PhiFn = std::function<double(const R2&, int kb)>;

static void clip_polygon_both_sides(
    const std::vector<R2> &poly,
    const std::vector<double> &phi,
    std::vector<std::vector<R2>> &out)
{
    int n = (int)poly.size();
    if (n < 3) return;
    bool all_neg = true, all_pos = true;
    for (double v : phi) { if (v >= 0) all_neg = false; if (v < 0) all_pos = false; }
    if (all_neg || all_pos) { out.push_back(poly); return; }

    std::vector<R2> neg, pos;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        if (phi[i] < 0) neg.push_back(poly[i]);
        else            pos.push_back(poly[i]);
        if ((phi[i] < 0) != (phi[j] < 0)) {
            double t = phi[i] / (phi[i] - phi[j]);
            R2 x(poly[i].x + t*(poly[j].x - poly[i].x),
                 poly[i].y + t*(poly[j].y - poly[i].y));
            neg.push_back(x);
            pos.push_back(x);
        }
    }
    if (neg.size() >= 3) out.push_back(std::move(neg));
    if (pos.size() >= 3) out.push_back(std::move(pos));
}

static void write_trimmed_both_vtk(
    const std::string &filename,
    const cutmesh_t &Khi,
    const std::vector<PhiFn> &levelsets,
    const std::vector<ScalarField> &sca_fields,
    const std::vector<ScalarField> &cell_fields = {},
    const std::vector<CellFn> &cell_fns = {},
    fct_t *vec_fh = nullptr,
    const std::string &vec_name = "")
{
    struct Cell { std::vector<R2> nodes; int kb; };
    std::vector<Cell> cells;

    int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        if (Khi.isInactive(ka, 0)) continue;
        int kb = Khi.idxElementInBackMesh(ka);

        // Start with the full quad
        std::vector<std::vector<R2>> polys;
        { std::vector<R2> q; for (int i = 0; i < 4; ++i) q.push_back(Khi[ka][i]); polys.push_back(std::move(q)); }

        // Clip by each level set sequentially
        for (auto &phi : levelsets) {
            std::vector<std::vector<R2>> next;
            for (auto &poly : polys) {
                std::vector<double> vals;
                vals.reserve(poly.size());
                for (auto &p : poly) vals.push_back(phi(p, kb));
                clip_polygon_both_sides(poly, vals, next);
            }
            polys = std::move(next);
        }

        for (auto &poly : polys) cells.push_back({std::move(poly), kb});
    }

    int total_nodes = 0;
    for (auto &c : cells) total_nodes += (int)c.nodes.size();

    std::ofstream out(filename);
    out << "# vtk DataFile Version 1.0\n"
        << "unstructured Grid\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n"
        << "POINTS " << total_nodes << " float\n";
    for (auto &c : cells)
        for (auto &p : c.nodes)
            out << p.x << " " << p.y << " 0.0\n";

    int cells_data = 0;
    for (auto &c : cells) cells_data += 1 + (int)c.nodes.size();
    out << "CELLS " << cells.size() << " " << cells_data << "\n";
    int off = 0;
    for (auto &c : cells) {
        out << c.nodes.size();
        for (size_t i = 0; i < c.nodes.size(); ++i) out << " " << off + (int)i;
        out << "\n";
        off += (int)c.nodes.size();
    }

    out << "CELL_TYPES " << cells.size() << "\n";
    for (auto &c : cells) out << ((c.nodes.size() == 4) ? 9 : 7) << "\n";

    out << "POINT_DATA " << total_nodes << "\n";

    for (auto &[fh, name] : sca_fields) {
        out << "SCALARS " << name << " float\n"
            << "LOOKUP_TABLE default\n";
        for (auto &c : cells)
            for (auto &p : c.nodes)
                out << paraviewFormat(fh->evalOnBackMesh(c.kb, 0, p, 0, 0)) << "\n";
    }

    if (vec_fh) {
        out << "VECTORS " << vec_name << " float\n";
        for (auto &c : cells)
            for (auto &p : c.nodes) {
                int kf = vec_fh->idxElementFromBackMesh(c.kb, 0);
                for (int dd = 0; dd < 2; ++dd)
                    out << paraviewFormat(vec_fh->eval(kf, p, dd)) << "\t";
                out << "0.0\n";
            }
    }

    if (!cell_fields.empty() || !cell_fns.empty()) {
        out << "CELL_DATA " << cells.size() << "\n";
        for (auto &[fh, name] : cell_fields) {
            out << "SCALARS " << name << " float\n"
                << "LOOKUP_TABLE default\n";
            for (auto &c : cells) {
                R2 centroid(0.0, 0.0);
                for (auto &p : c.nodes) { centroid.x += p.x; centroid.y += p.y; }
                centroid.x /= c.nodes.size();
                centroid.y /= c.nodes.size();
                out << paraviewFormat(fh->evalOnBackMesh(c.kb, 0, centroid, 0, 0)) << "\n";
            }
        }
        for (auto &[fn, name] : cell_fns) {
            out << "SCALARS " << name << " float\n"
                << "LOOKUP_TABLE default\n";
            for (auto &c : cells) {
                R2 centroid(0.0, 0.0);
                for (auto &p : c.nodes) { centroid.x += p.x; centroid.y += p.y; }
                centroid.x /= c.nodes.size();
                centroid.y /= c.nodes.size();
                out << paraviewFormat(fn(centroid, c.kb)) << "\n";
            }
        }
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {

    MPIcf cfMPI(argc, argv);
    globalVariable::verbose = 0;

    // Parse run mode from command line
    RunMode run_mode = RunMode::export_mi;
    if (argc > 1) {
        std::string arg(argv[1]);
        if (arg == "import") run_mode = RunMode::import_mi;
        else if (arg == "export") run_mode = RunMode::export_mi;
        else { std::cerr << "Usage: " << argv[0] << " [export|import]\n"; return 1; }
    }

    std::filesystem::create_directories("output/growth_cutfem");
    build_soc_geometry();

    // ---- Mesh ----
    const int nx = 200;
    double dx_bg = bg_xmax - bg_xmin, dy_bg = bg_ymax - bg_ymin;
    int ny = static_cast<int>(nx * dy_bg / dx_bg);
    mesh_t Kh(nx, ny, bg_xmin + 0.00137, bg_ymin - 0.00113, dx_bg, dy_bg);
    double h = dx_bg / (nx - 1);

    // ---- SOC boundary level set ----
    space_t Lh(Kh, DataFE<mesh_t>::P1);
    fct_t levelSet(Lh, fun_levelSet);
    InterfaceLevelSet<mesh_t> interface(Kh, levelSet);

    cutmesh_t Khi(Kh);
    Khi.truncate(interface, 1);

    // ---- FE spaces ----
    LagrangeQuad2 FE_vec(1);
    space_t Uh(Kh, FE_vec);
    cutspace_t Wh(Khi, Uh);
    int nb_dof = Wh.get_nb_dof();

    space_t Sh(Kh, DataFE<mesh_t>::P1);
    int nb_sca = Sh.NbDoF();

    std::cout << "Mesh: " << nx << "x" << ny << ", h=" << h
              << ", vec DOFs=" << nb_dof << ", sca DOFs=" << nb_sca
              << ", Kh.nt=" << Kh.nt << ", Kh.nv=" << Kh.nv << "\n";

    Normal n;
    double nitsche_penalty = 20.0 * (2.0*mu_bone + lambda_bone) * 4.0;
    double ghost_param = 0.5;

    // Bottom indicator
    fct_t bottomInd(Sh, [](R2 P, int i, int dom) -> double {
        return (std::abs(P.y - y_bottom) < 0.05 &&
                P.x >= x_bottom_min - 0.05 &&
                P.x <= x_bottom_max + 0.05) ? 1.0 : 0.0;
    });

    // ---- Material coefficients (base: bone/cartilage, no ossification) ----
    auto make_base_two_mu = [&]() {
        return fct_t(Sh, [](R2 P, int, int) -> double {
            return 2.0 * ((P.y < interfaceY) ? mu_bone : mu_cart);
        });
    };
    auto make_base_lambda = [&]() {
        return fct_t(Sh, [](R2 P, int, int) -> double {
            return (P.y < interfaceY) ? lambda_bone : lambda_cart;
        });
    };

    // ============================================================
    // Solve all load steps, compute per-step stress fields, average
    // ============================================================
    struct RunResult {
        std::vector<double> U_avg;
        std::vector<std::vector<double>> all_sols;
        std::vector<StressFields> all_sf;
        StressFields sf_avg; // averaged stress fields (per-step MI → average)
    };

    auto run_elasticity = [&](const fct_t &two_mu_fh, const fct_t &lambda_fh) -> RunResult
    {
        auto steps = build_default_steps();
        std::vector<double> U_sum(nb_dof, 0.0);
        std::vector<double> hd_sum(nb_sca, 0.0), oct_sum(nb_sca, 0.0), mi_sum(nb_sca, 0.0);
        std::vector<std::vector<double>> all_sols;
        std::vector<StressFields> all_sf;

        for (unsigned int sidx = 0; sidx < steps.size(); ++sidx) {
            g_step_u_center = steps[sidx].u_center;
            g_step_u_radius = steps[sidx].u_radius;
            g_step_p_peak   = steps[sidx].p_peak;

            CutFEM<mesh_t> problem(Wh);
            funtest_t u(Wh, 2, 0), v(Wh, 2, 0);

            problem.addBilinear(
                contractProduct(two_mu_fh.expr() * Eps(u), Eps(v))
                + innerProduct(lambda_fh.expr() * div(u), div(v)),
                Khi);

            problem.addBilinear(
                innerProduct(bottomInd.expr() * (nitsche_penalty / h) * u, v),
                interface);

            problem.addFaceStabilization(
                innerProduct(ghost_param * pow(h,-1) * jump(u), jump(v))
                + innerProduct(ghost_param * pow(h,1) * jump(grad(u)*n), jump(grad(v)*n)),
                Khi);

            fct_t tractionFh(Uh, fun_traction);
            problem.addLinear(innerProduct(tractionFh.exprList(), v), interface);

            problem.solve("umfpack");

            std::span<double> data_uh(problem.rhs_.data(), nb_dof);
            std::vector<double> sol(data_uh.begin(), data_uh.end());
            for (int i = 0; i < nb_dof; ++i) U_sum[i] += sol[i];

            // Compute stress fields for THIS step's displacement
            {
                std::span<double> sp(sol);
                fct_t uh_step(Wh, sp);
                auto sf_step = compute_stress_fields(uh_step, Sh, Khi, Kh, nb_sca);
                for (int i = 0; i < nb_sca; ++i) {
                    hd_sum[i]  += sf_step.hydrostatic[i];
                    oct_sum[i] += sf_step.oct_shear[i];
                    mi_sum[i]  += sf_step.miner[i];
                }
                all_sf.push_back(std::move(sf_step));
            }

            all_sols.push_back(std::move(sol));
            std::cout << "  Step " << sidx+1 << "/" << steps.size() << " done.\n";
        }

        // Average displacement and stress fields
        double invN = 1.0 / steps.size();
        for (auto &v : U_sum) v *= invN;
        for (int i = 0; i < nb_sca; ++i) {
            hd_sum[i]  *= invN;
            oct_sum[i] *= invN;
            mi_sum[i]  *= invN;
        }

        RunResult res;
        res.U_avg = std::move(U_sum);
        res.all_sols = std::move(all_sols);
        res.all_sf = std::move(all_sf);
        res.sf_avg.hydrostatic = std::move(hd_sum);
        res.sf_avg.oct_shear   = std::move(oct_sum);
        res.sf_avg.miner       = std::move(mi_sum);
        return res;
    };

    // ============================================================
    // EXPORT MODE
    // ============================================================
    // if (run_mode == RunMode::export_mi) {
    {
        std::cout << "\n=== EXPORT MODE (Step 1) ===\n";

        fct_t two_mu_fh = make_base_two_mu();
        fct_t lambda_fh = make_base_lambda();
        auto result = run_elasticity(two_mu_fh, lambda_fh);

        // Level set FE functions: SOC boundary + bone/cartilage interface
        std::vector<double> phi_outer_data(nb_sca), phi_iface_data(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                phi_outer_data[iglo]   = signed_distance_polygon(P, g_polygon);
                phi_iface_data[iglo] = P.y - interfaceY;
            }
        }
        std::span<double> phi_outer_span(phi_outer_data);
        fct_t phi_outer_fh(Sh, phi_outer_span);
        std::span<double> phi_iface_span(phi_iface_data);
        fct_t phi_iface_fh(Sh, phi_iface_span);

        // Zero phi_soc (no ossification in step 0)
        std::vector<double> phi_soc_data(nb_sca, 0.0);
        std::span<double> phi_soc_span(phi_soc_data);
        fct_t phi_soc_fh(Sh, phi_soc_span);

        g_mi_avg = result.sf_avg.miner;
        g_mi_threshold = find_threshold(g_mi_avg, Sh, Kh);

        // Prepare averaged fields
        std::span<double> sp_avg(result.U_avg);
        fct_t uh_avg(Wh, sp_avg);
        std::span<double> hd_span(result.sf_avg.hydrostatic);
        fct_t hd_fh(Sh, hd_span);
        std::span<double> oct_span(result.sf_avg.oct_shear);
        fct_t oct_fh(Sh, oct_span);
        std::span<double> mi_span(result.sf_avg.miner);
        fct_t mi_fh(Sh, mi_span);

        // Material indicator (step 0): 0=bone, 1=cartilage
        std::vector<double> mat_data(nb_sca, 0.0);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                mat_data[iglo] = (FK.Pt(j).y < interfaceY) ? 0.0 : 1.0;
            }
        }
        std::span<double> mat_span(mat_data);
        fct_t mat_fh(Sh, mat_span);

        // Write per-step outputs (trimmed)
        for (unsigned int sidx = 0; sidx < result.all_sols.size(); ++sidx) {
            std::span<double> sp(result.all_sols[sidx]);
            fct_t uh(Wh, sp);

            std::span<double> hd_step_span(result.all_sf[sidx].hydrostatic);
            fct_t hd_step(Sh, hd_step_span);
            std::span<double> oct_step_span(result.all_sf[sidx].oct_shear);
            fct_t oct_step(Sh, oct_step_span);
            std::span<double> mi_step_span(result.all_sf[sidx].miner);
            fct_t mi_step(Sh, mi_step_span);

            Paraview<mesh_t> w(Khi, "output/growth_cutfem/growth_cutfem_iter_0_step_" + std::to_string(sidx+1) + ".vtk");
            w.add(uh, "displacement", 0, 2);
            w.add(hd_step, "hydrostatic", 0, 1);
            w.add(oct_step, "oct_shear", 0, 1);
            w.add(mi_step, "miner_index", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Trimmed (cut polygons at SOC boundary)
        {
            Paraview<mesh_t> w(Khi, "output/growth_cutfem/growth_cutfem_fields_0.vtk");
            w.add(uh_avg, "displacement", 0, 2);
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Active elements (full quads)
        {
            Paraview<mesh_t> w;
            w.writeActiveMesh(Khi, "output/growth_cutfem/growth_cutfem_active_0.vtk");
            w.add(uh_avg, "displacement", 0, 2);
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Trimmed both sides (SOC boundary + bone/cartilage interface)
        write_trimmed_both_vtk(
            "output/growth_cutfem/growth_cutfem_trimmed_0.vtk", Khi,
            {[](const R2& P, int) { return signed_distance_polygon(P, g_polygon); },
             [](const R2& P, int) { return P.y - interfaceY; }},
            {{&hd_fh, "hydrostatic"}, {&oct_fh, "oct_shear"},
             {&mi_fh, "miner_index"},
             {&phi_outer_fh, "phi_outer"}, {&phi_iface_fh, "phi_interface"},
             {&phi_soc_fh, "phi_soc"}},
            {{&mat_fh, "material"}},
            {{[](const R2& P, int) -> double { return P.y < interfaceY ? 0.0 : 1.0; },
              "material_sharp"}},
            &uh_avg, "displacement");

        // Cartilage only (above interface, inside SOC)
        {
            std::vector<double> phi_cart_data(nb_sca);
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    double phi_soc = signed_distance_polygon(P, g_polygon);
                    double phi_y   = interfaceY - P.y;
                    phi_cart_data[iglo] = std::max(phi_soc, phi_y);
                }
            }
            std::span<double> phi_cart_span(phi_cart_data);
            fct_t phi_cart(Sh, phi_cart_span);
            InterfaceLevelSet<mesh_t> cart_interface(Kh, phi_cart);

            cutmesh_t Khi_cart(Kh);
            Khi_cart.truncate(cart_interface, 1);

            Paraview<mesh_t> w(Khi_cart, "output/growth_cutfem/growth_cutfem_cartilage_0.vtk");
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        save_restart(g_mi_avg, g_mi_threshold, iteration_id);
        std::cout << "Export done.\n";
    }
    // ============================================================
    // IMPORT MODE
    // ============================================================
    // else {
    {
        std::cout << "\n=== IMPORT MODE (Step 2) ===\n";

        // Load Miner index + threshold from step 1
        load_restart(g_mi_avg, g_mi_threshold, import_iteration_id);

        // Build ossification level set with spline-proximity inhibition.
        // Use signed distance to SOC polygon: near the spline boundary the
        // effective MI is scaled down so ossification cannot nucleate there.
        //   inhibition = smoothstep(dist / oss_spline_band)
        //   phi_oss    = MI * inhibition - threshold
        std::vector<double> phi_soc_data(nb_sca);
        std::vector<double> inhibition_data(nb_sca, 1.0);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                double dist = std::abs(signed_distance_polygon(P, g_polygon));
                double t = std::clamp(dist / oss_spline_band, 0.0, 1.0);
                double inhibition = t * t * (3.0 - 2.0 * t); // smoothstep
                inhibition_data[iglo] = inhibition;
                phi_soc_data[iglo] = g_mi_avg[iglo] * inhibition - g_mi_threshold;
            }
        }

        std::span<double> phi_soc_span(phi_soc_data);
        fct_t phi_oss(Sh, phi_soc_span);

        // Build material coefficients using φ_oss sign:
        // bone (y < interface): always bone
        // cartilage (φ_oss < 0): E=6, ν=0.47
        // ossified  (φ_oss ≥ 0): E=253, ν=0.335
        fct_t two_mu_fh(Sh, [](R2 P, int, int) -> double {
            if (P.y < interfaceY) return 2.0 * mu_bone;
            return 2.0 * mu_cart; // default cartilage
        });
        fct_t lambda_fh(Sh, [](R2 P, int, int) -> double {
            if (P.y < interfaceY) return lambda_bone;
            return lambda_cart;
        });

        // Override material in ossified region using the level set nodal values
        // For nodes where φ_oss >= 0 and y ≥ interfaceY, set ossified material
        {
            auto &tmu_v = two_mu_fh.v;
            auto &lam_v = lambda_fh.v;
            int n_ossified = 0;
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    if (phi_soc_data[iglo] >= 0.0 && FK.Pt(j).y >= interfaceY) {
                        tmu_v[iglo] = 2.0 * mu_oss;
                        lam_v[iglo] = lambda_oss;
                        ++n_ossified;
                    }
                }
            }
            std::cout << "Ossified nodes: " << n_ossified << " / " << nb_sca << "\n";
        }

        // Solve with ossified materials
        auto result = run_elasticity(two_mu_fh, lambda_fh);

        // Material indicator: 0=bone, 1=cartilage, 2=ossified
        std::vector<double> mat_data(nb_sca, 0.0);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                if (P.y < interfaceY) mat_data[iglo] = 0.0;
                else if (phi_soc_data[iglo] >= 0.0) mat_data[iglo] = 2.0;
                else mat_data[iglo] = 1.0;
            }
        }

        // Level set FE functions: SOC boundary + bone/cartilage interface
        std::vector<double> phi_outer_data(nb_sca), phi_iface_data(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                phi_outer_data[iglo]   = signed_distance_polygon(P, g_polygon);
                phi_iface_data[iglo] = P.y - interfaceY;
            }
        }
        std::span<double> phi_outer_span(phi_outer_data);
        fct_t phi_outer_fh(Sh, phi_outer_span);
        std::span<double> phi_iface_span(phi_iface_data);
        fct_t phi_iface_fh(Sh, phi_iface_span);

        // Write per-step outputs (trimmed)
        for (unsigned int sidx = 0; sidx < result.all_sols.size(); ++sidx) {
            std::span<double> sp(result.all_sols[sidx]);
            fct_t uh(Wh, sp);

            std::span<double> hd_step_span(result.all_sf[sidx].hydrostatic);
            fct_t hd_step(Sh, hd_step_span);
            std::span<double> oct_step_span(result.all_sf[sidx].oct_shear);
            fct_t oct_step(Sh, oct_step_span);
            std::span<double> mi_step_span(result.all_sf[sidx].miner);
            fct_t mi_step(Sh, mi_step_span);

            Paraview<mesh_t> w(Khi, "output/growth_cutfem/growth_cutfem_iter_1_step_" + std::to_string(sidx+1) + ".vtk");
            w.add(uh, "displacement", 0, 2);
            w.add(hd_step, "hydrostatic", 0, 1);
            w.add(oct_step, "oct_shear", 0, 1);
            w.add(mi_step, "miner_index", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_oss, "phi_soc", 0, 1);
        }

        // Prepare averaged fields
        std::span<double> sp_avg(result.U_avg);
        fct_t uh_avg(Wh, sp_avg);
        std::span<double> hd_span(result.sf_avg.hydrostatic);
        fct_t hd_fh(Sh, hd_span);
        std::span<double> oct_span(result.sf_avg.oct_shear);
        fct_t oct_fh(Sh, oct_span);
        std::span<double> mi_span(result.sf_avg.miner);
        fct_t mi_fh(Sh, mi_span);
        std::span<double> mat_span(mat_data);
        fct_t mat_fh(Sh, mat_span);
        std::span<double> inhib_span(inhibition_data);
        fct_t inhib_fh(Sh, inhib_span);

        // Trimmed (cut polygons at SOC boundary)
        {
            Paraview<mesh_t> w(Khi, "output/growth_cutfem/growth_cutfem_fields_1.vtk");
            w.add(uh_avg, "displacement", 0, 2);
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(phi_oss, "phi_soc", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(inhib_fh, "spline_inhibition", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
        }

        // Active elements (full quads)
        {
            Paraview<mesh_t> w;
            w.writeActiveMesh(Khi, "output/growth_cutfem/growth_cutfem_active_1.vtk");
            w.add(uh_avg, "displacement", 0, 2);
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(phi_oss, "phi_soc", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(inhib_fh, "spline_inhibition", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
        }

        // Trimmed both sides (SOC boundary + bone/cartilage interface + ossification front)
        write_trimmed_both_vtk(
            "output/growth_cutfem/growth_cutfem_trimmed_1.vtk", Khi,
            {[](const R2& P, int) { return signed_distance_polygon(P, g_polygon); },
             [](const R2& P, int) { return P.y - interfaceY; },
             [&phi_oss](const R2& P, int kb) { return phi_oss.evalOnBackMesh(kb, 0, &P.x, 0, 0); }},
            {{&hd_fh, "hydrostatic"}, {&oct_fh, "oct_shear"},
             {&mi_fh, "miner_index"},
             {&phi_oss, "phi_soc"}, {&inhib_fh, "spline_inhibition"},
             {&phi_outer_fh, "phi_outer"}, {&phi_iface_fh, "phi_interface"}},
            {{&mat_fh, "material"}},
            {{[&phi_oss](const R2& P, int kb) -> double {
                  if (P.y < interfaceY) return 0.0;
                  if (phi_oss.evalOnBackMesh(kb, 0, &P.x, 0, 0) >= 0.0) return 2.0;
                  return 1.0;
              }, "material_sharp"}},
            &uh_avg, "displacement");

        // Cartilage only (above interface, inside SOC, not ossified)
        {
            std::vector<double> phi_cart_data(nb_sca);
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    double phi_soc = signed_distance_polygon(P, g_polygon);
                    double phi_y   = interfaceY - P.y;
                    double phi_o   = phi_soc_data[iglo];
                    phi_cart_data[iglo] = std::max({phi_soc, phi_y, phi_o});
                }
            }
            std::span<double> phi_cart_span(phi_cart_data);
            fct_t phi_cart(Sh, phi_cart_span);
            InterfaceLevelSet<mesh_t> cart_interface(Kh, phi_cart);

            cutmesh_t Khi_cart(Kh);
            Khi_cart.truncate(cart_interface, 1);

            Paraview<mesh_t> w(Khi_cart, "output/growth_cutfem/growth_cutfem_cartilage_1.vtk");
            w.add(hd_fh, "hydrostatic", 0, 1);
            w.add(oct_fh, "oct_shear", 0, 1);
            w.add(mi_fh, "miner_index", 0, 1);
            w.add(phi_oss, "phi_soc", 0, 1);
            w.add(mat_fh, "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
        }

        save_restart(result.sf_avg.miner, g_mi_threshold, iteration_id);
        std::cout << "Import done.\n";
    }

    std::cout << "\nDone. Output in output/growth_cutfem/\n";
    return 0;
}
