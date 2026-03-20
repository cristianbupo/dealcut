/**
 * @brief Iterative growth/ossification with CutFEM-Library.
 *
 * Single-run iterative workflow on SOC domain:
 *   Iteration 0: bi-material (bone + cartilage) → MI → threshold → ossified region
 *   Iteration 1: ossified region becomes bone → new MI → new ossified region
 *   Iteration 2: repeat ossification update
 *   ... and so on for N_ITERATIONS
 *
 * Materials:
 *   bone      (y < 1.0):            E=500,  ν=0.2
 *   cartilage (y ≥ 1.0):            E=6,    ν=0.47
 */

#include "../cutfem.hpp"
#include <gmsh.h>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <iomanip>
#include <functional>
#include <regex>
#include <sstream>

using mesh_t     = MeshQuad2;
using funtest_t  = TestFunction<mesh_t>;
using fct_t      = FunFEM<mesh_t>;
using cutmesh_t  = ActiveMesh<mesh_t>;
using space_t    = GFESpace<mesh_t>;
using cutspace_t = CutFESpace<mesh_t>;

// ============================================================
// Parameters
// ============================================================
struct SimConfig {
    int n_iterations = 10;
    bool export_cartilage = false;
    bool export_active = false;
    bool export_trimmed = true;

    double p1_geom = 0.9;
    double p2_geom = 0.2;
    std::vector<double> bspline_weights = {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0};
    int transfinite_nodes = 101;

    double interface_y = 1.0;
    double oss_height_target = 0.1;
    double k_mi = 0.5;
    double oss_spline_band = 0.3;

    double load_center_u_frac = 0.50;
    double load_du_frac = 0.08;
    double load_radius_scale = 1.0;
    double load_p_peak = 1.0;

    int mesh_nx = 200;
    double mesh_y_offset = -0.00113;

    double mi_threshold_frac = 0.9; // fraction of selected max MI used as threshold

    double poisson_source = -1.0; // f in -Δu + αu = f (negative = consumption)
    double poisson_flux   =  1.0; // flux from ossification front into cartilage
    double poisson_decay  =  0.1; // α: natural decay rate (regularizes pure Neumann)
    bool poisson_dirichlet_outer = true;  // Dirichlet u=0 on SOC outer boundary
    bool poisson_dirichlet_iface = true;  // Dirichlet u=0 on bone-cartilage interface

    std::string output_dir = "output/growth_iterative";
    std::string gmsh_model_name = "growth_iterative_cutfem";
};

static SimConfig g_cfg;

static std::string trim_copy(const std::string &s) {
    const std::string ws = " \t\n\r";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

static bool parse_json_number(const std::string &text, const std::string &key, double &value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    value = std::stod(m[1].str());
    return true;
}

static bool parse_json_int(const std::string &text, const std::string &key, int &value) {
    double tmp = 0.0;
    if (!parse_json_number(text, key, tmp)) return false;
    value = static_cast<int>(std::llround(tmp));
    return true;
}

static bool parse_json_bool(const std::string &text, const std::string &key, bool &value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    value = (m[1].str() == "true");
    return true;
}

static bool parse_json_string(const std::string &text, const std::string &key, std::string &value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    value = m[1].str();
    return true;
}

static bool parse_json_number_array(
    const std::string &text,
    const std::string &key,
    std::vector<double> &values)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;

    std::vector<double> parsed;
    std::stringstream ss(m[1].str());
    std::string item;
    while (std::getline(ss, item, ',')) {
        const std::string tok = trim_copy(item);
        if (tok.empty()) continue;
        parsed.push_back(std::stod(tok));
    }
    if (parsed.empty()) return false;
    values = std::move(parsed);
    return true;
}

static bool load_config_from_json(const std::string &path, SimConfig &cfg) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    parse_json_int(text, "n_iterations", cfg.n_iterations);
    parse_json_bool(text, "export_cartilage", cfg.export_cartilage);
    parse_json_bool(text, "export_active", cfg.export_active);
    parse_json_bool(text, "export_trimmed", cfg.export_trimmed);

    parse_json_number(text, "p1_geom", cfg.p1_geom);
    parse_json_number(text, "p2_geom", cfg.p2_geom);
    parse_json_number_array(text, "bspline_weights", cfg.bspline_weights);
    parse_json_int(text, "transfinite_nodes", cfg.transfinite_nodes);

    parse_json_number(text, "interface_y", cfg.interface_y);
    parse_json_number(text, "oss_height_target", cfg.oss_height_target);
    parse_json_number(text, "k_mi", cfg.k_mi);
    parse_json_number(text, "oss_spline_band", cfg.oss_spline_band);

    parse_json_number(text, "load_center_u_frac", cfg.load_center_u_frac);
    parse_json_number(text, "load_du_frac", cfg.load_du_frac);
    parse_json_number(text, "load_radius_scale", cfg.load_radius_scale);
    parse_json_number(text, "load_p_peak", cfg.load_p_peak);

    parse_json_int(text, "mesh_nx", cfg.mesh_nx);
    parse_json_number(text, "mesh_y_offset", cfg.mesh_y_offset);
    parse_json_number(text, "mi_threshold_frac", cfg.mi_threshold_frac);
    parse_json_number(text, "poisson_source", cfg.poisson_source);
    parse_json_number(text, "poisson_flux", cfg.poisson_flux);
    parse_json_number(text, "poisson_decay", cfg.poisson_decay);
    parse_json_bool(text, "poisson_dirichlet_outer", cfg.poisson_dirichlet_outer);
    parse_json_bool(text, "poisson_dirichlet_iface", cfg.poisson_dirichlet_iface);

    parse_json_string(text, "output_dir", cfg.output_dir);
    parse_json_string(text, "gmsh_model_name", cfg.gmsh_model_name);
    return true;
}

// Geometry
static constexpr double s = 1.0;

static constexpr double bg_xmin = s * -1.21;
static constexpr double bg_ymin = 0.0;
static constexpr double bg_xmax = s *  1.21;
static constexpr double bg_ymax = s * (2.41 + 0.5);

static constexpr double x_bottom_min = s * -0.5;
static constexpr double x_bottom_max = s *  0.5;
static constexpr double y_bottom     = 0.0;

// Material (plane strain)

static constexpr double E_bone  = 500.0, nu_bone = 0.2;
static constexpr double mu_bone     = E_bone / (2.0 * (1.0 + nu_bone));
static constexpr double lambda_bone = E_bone * nu_bone / ((1.0 + nu_bone) * (1.0 - 2.0 * nu_bone));

static constexpr double E_cart  = 6.0, nu_cart = 0.47;
static constexpr double mu_cart     = E_cart / (2.0 * (1.0 + nu_cart));
static constexpr double lambda_cart = E_cart * nu_cart / ((1.0 + nu_cart) * (1.0 - 2.0 * nu_cart));

// Ossification parameters

struct StepDef { double u_center, u_radius, p_peak; };

enum class Phase { MI, POISSON };

static const char* phase_name(Phase p) {
    return (p == Phase::MI) ? "MI" : "Poisson";
}

// Schedule: MI, MI, then alternating Poisson/MI
static std::vector<Phase> build_phase_schedule(int n_iterations) {
    std::vector<Phase> schedule;
    for (int i = 0; i < n_iterations; ++i) {
        if (i < 2)
            schedule.push_back(Phase::MI);
        else
            schedule.push_back((i % 2 == 0) ? Phase::POISSON : Phase::MI);
    }
    return schedule;
}

// ============================================================
// Globals
// ============================================================
static std::vector<R2> g_polygon;
static std::vector<R2> g_top_spline;
static std::vector<double> g_top_u;
static double g_step_u_center, g_step_u_radius, g_step_p_peak;

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
    gmsh::model::add(g_cfg.gmsh_model_name);

    gmsh::model::occ::addPoint( s*0.5,  0.0,          0, 0.1, 1);
    gmsh::model::occ::addPoint( s*0.5,  s*1.0,        0, 0.1, 2);
    gmsh::model::occ::addPoint( s*1.0,  s * g_cfg.p1_geom,      0, 0.1, 3);
    gmsh::model::occ::addPoint( s*1.0,  s*2.4,        0, 0.1, 4);
    gmsh::model::occ::addPoint( 0.0,    s*2.4 + s * g_cfg.p2_geom,0, 0.1, 5);
    gmsh::model::occ::addPoint(-s*1.0,  s*2.4,        0, 0.1, 6);
    gmsh::model::occ::addPoint(-s*1.0,  s * g_cfg.p1_geom,      0, 0.1, 7);
    gmsh::model::occ::addPoint(-s*0.5,  s*1.0,        0, 0.1, 8);
    gmsh::model::occ::addPoint(-s*0.5,  0.0,          0, 0.1, 9);

    gmsh::model::occ::addLine(9, 1, 1);
    gmsh::model::occ::addBSpline({1,2,3,4,5,6,7,8,9}, 2, 3, g_cfg.bspline_weights);
    gmsh::model::occ::synchronize();
    gmsh::model::mesh::setTransfiniteCurve(2, g_cfg.transfinite_nodes);
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

    // Enforce perfect symmetry about x=0.
    // Spline goes from right endpoint (x>0) to left endpoint (x<0).
    // Node i mirrors node N-1-i: average y, negate x.
    {
        int N = (int)g_top_spline.size();
        for (int i = 0; i < N/2; ++i) {
            int j = N - 1 - i;
            double x_avg = 0.5 * (g_top_spline[i].x - g_top_spline[j].x);
            double y_avg = 0.5 * (g_top_spline[i].y + g_top_spline[j].y);
            g_top_spline[i] = R2( x_avg, y_avg);
            g_top_spline[j] = R2(-x_avg, y_avg);
        }
        if (N % 2 == 1) {
            g_top_spline[N/2].x = 0.0;
        }
        // Symmetrize u parameters about u_mid
        double u0 = g_top_u.front(), u1 = g_top_u.back();
        double u_mid = 0.5 * (u0 + u1);
        for (int i = 0; i < N/2; ++i) {
            int j = N - 1 - i;
            double du = 0.5 * ((g_top_u[i] - u_mid) - (g_top_u[j] - u_mid));
            g_top_u[i] = u_mid + du;
            g_top_u[j] = u_mid - du;
        }
        if (N % 2 == 1) g_top_u[N/2] = u_mid;
    }

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
    double uc = u0 + g_cfg.load_center_u_frac * ur;
    double du = g_cfg.load_du_frac * ur;
    double radius = std::max(1e-14, g_cfg.load_radius_scale * du);
    std::vector<StepDef> steps = {
        {uc + 2*du, radius, 0.50 * g_cfg.load_p_peak},
        {uc + du,   radius, 0.75 * g_cfg.load_p_peak},
        {uc,        radius, 1.00 * g_cfg.load_p_peak},
        {uc - du,   radius, 0.75 * g_cfg.load_p_peak},
        {uc - 2*du, radius, 0.50 * g_cfg.load_p_peak}
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
    return {vm, hd, oct, oct + g_cfg.k_mi * std::min(hd, 0.0)};
}

// ============================================================
// Algoim level-set for per-element material interface
// ============================================================
// Bilinear interpolation of (lambda - lam_mid) on an axis-aligned quad.
// phi < 0 ↔ cartilage, phi > 0 ↔ bone.
// Node layout: v00=(xmin,ymin), v10=(xmax,ymin), v01=(xmin,ymax), v11=(xmax,ymax).
struct ElementMaterialLS {
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
// Stress fields
// ============================================================
struct StressFields {
    std::vector<double> hydrostatic, oct_shear, miner;
};

static StressFields compute_stress_fields(
    const fct_t &uh, const space_t &Sh,
    const cutmesh_t &Khi, const mesh_t &Kh, int nb_sca_dof,
    const fct_t &two_mu_fh, const fct_t &lambda_fh)
{
    static constexpr double lam_mid = 0.5 * (lambda_bone + lambda_cart);
    static constexpr int algoim_order = 3;

    std::vector<double> hd_sum(nb_sca_dof, 0.0), oct_sum(nb_sca_dof, 0.0), mi_sum(nb_sca_dof, 0.0);
    std::vector<double> wt_sum(nb_sca_dof, 0.0);

    int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        int kb = Khi.idxElementInBackMesh(ka);
        const auto &FK = Sh[kb];
        int ndf = FK.NbDoF();

        // Gather node positions and material
        R2 pts[4];
        double node_lam[4];
        bool has_bone = false, has_cart = false;
        for (int j = 0; j < ndf; ++j) {
            pts[j] = FK.Pt(j);
            node_lam[j] = lambda_fh.evalOnBackMesh(kb, 0, pts[j], 0, 0);
            if (node_lam[j] > lam_mid) has_bone = true;
            else                        has_cart = true;
        }

        if (!has_cart) continue; // fully bone → skip

        if (!has_bone) {
            // ---- Fully cartilage: centroid evaluation as before ----
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

            double lam = lambda_fh.evalOnBackMesh(kb, 0, centroid, 0, 0);
            double mu  = two_mu_fh.evalOnBackMesh(kb, 0, centroid, 0, 0) * 0.5;

            auto inv = compute_invariants(exx, eyy, exy, lam, mu);

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
            // Node ordering: 0=(xmin,ymin), 1=(xmax,ymin), 2=(xmax,ymax), 3=(xmin,ymax)
            double xmin = pts[0].x, ymin = pts[0].y;
            double xmax = pts[2].x, ymax = pts[2].y;

            ElementMaterialLS phi_mat;
            phi_mat.xmin = xmin; phi_mat.xmax = xmax;
            phi_mat.ymin = ymin; phi_mat.ymax = ymax;
            phi_mat.v00 = node_lam[0] - lam_mid;
            phi_mat.v10 = node_lam[1] - lam_mid;
            phi_mat.v11 = node_lam[2] - lam_mid;
            phi_mat.v01 = node_lam[3] - lam_mid;

            algoim::QuadratureRule<2> q = algoim::quadGen<2>(
                phi_mat,
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

                // Pure cartilage material on this sub-domain
                auto inv = compute_invariants(exx, eyy, exy, lambda_cart, mu_cart);
                hd_acc  += inv.hydrostatic * w;
                oct_acc += inv.oct_shear   * w;
                mi_acc  += inv.miner       * w;
                w_acc   += w;
            }

            if (w_acc <= 0) continue;

            // Distribute to cartilage-side nodes only, weighted by sub-area
            for (int j = 0; j < ndf; ++j) {
                if (node_lam[j] > lam_mid) continue; // bone node
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
// Threshold finding
// ============================================================
static double compute_cartilage_area(const space_t &Sh, const mesh_t &Kh) {
    double area = 0.0;
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        double y_avg = 0.0;
        int ndf = FK.NbDoF();
        for (int j = 0; j < ndf; ++j) y_avg += FK.Pt(j).y;
        y_avg /= ndf;
        if (y_avg < g_cfg.interface_y) continue;
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
        if (ymax_e < g_cfg.interface_y) continue;

        double hx = xmax_e - xmin_e, hy = ymax_e - ymin_e;
        if (hx < 1e-14 || hy < 1e-14) continue;
        double xi = (0.0 - xmin_e) / hx;

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
        int nsamp = 5;
        for (int ss = 0; ss <= nsamp; ++ss) {
            double eta = (double)ss / nsamp;
            double y_s = ymin_e + eta * hy;
            if (y_s < g_cfg.interface_y) continue;
            double mi_s = (1-xi)*(1-eta)*mi0 + xi*(1-eta)*mi1
                        + xi*eta*mi2 + (1-xi)*eta*mi3;
            profile.push_back({y_s, mi_s});
        }
    }
    std::sort(profile.begin(), profile.end());
    return profile;
}

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
    auto profile = build_axis_profile(mi_nodal, Sh, Kh);

    double mi_min_ax = 1e30, mi_max_ax = -1e30;
    for (auto &[y, mi] : profile) {
        mi_min_ax = std::min(mi_min_ax, mi);
        mi_max_ax = std::max(mi_max_ax, mi);
    }

    double a = mi_min_ax, b = mi_max_ax;
    double fa = compute_oss_height_axis(profile, a) - g_cfg.oss_height_target;
    double fb = compute_oss_height_axis(profile, b) - g_cfg.oss_height_target;

    double threshold = 0.5*(a+b);
    for (int iter = 0; iter < 100; ++iter) {
        if (std::abs(fa - fb) < 1e-15) break;
        double c = (a*fb - b*fa) / (fb - fa);
        c = std::clamp(c, mi_min_ax, mi_max_ax);
        double fc = compute_oss_height_axis(profile, c) - g_cfg.oss_height_target;
        if (std::abs(fc) < 1e-12) { threshold = c; break; }

        if (fc * fa > 0) { a = c; fa = fc; fb *= 0.5; }
        else             { b = c; fb = fc; fa *= 0.5; }
        threshold = c;
    }

    double achieved_height = compute_oss_height_axis(profile, threshold);
    std::cout << "  Threshold: " << threshold
              << ", achieved_h(axis)=" << achieved_height
              << ", mi range(axis)=[" << mi_min_ax << ", " << mi_max_ax << "]\n";

    return threshold;
}

// ============================================================
// Poisson threshold: minimum Poisson value at ossification interface
// ============================================================
static double compute_poisson_threshold(
    const std::vector<double> &poisson_data,
    const std::vector<double> &phi_oss_data,
    const std::vector<double> &pre_mi_phi_oss_data,
    int nb_sca)
{
    double min_val = std::numeric_limits<double>::max();
    int count = 0;
    // Newly ossified nodes: was cartilage before last MI step, now ossified
    for (int i = 0; i < nb_sca; ++i) {
        if (phi_oss_data[i] >= 0.0 && pre_mi_phi_oss_data[i] < 0.0) {
            min_val = std::min(min_val, poisson_data[i]);
            ++count;
        }
    }
    std::cout << "  Poisson threshold: " << min_val
              << " (from " << count << " newly-ossified nodes)\n";
    return (count > 0) ? min_val : 0.0;
}

// ============================================================
// Both-sides-trimmed VTK export
// ============================================================
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

// One-sided clip: keeps only the phi < 0 side.
// Used for the SOC outer boundary so that the outside piece of cut elements is discarded.
static void clip_polygon_keep_negative(
    const std::vector<R2> &poly,
    const std::vector<double> &phi,
    std::vector<std::vector<R2>> &out)
{
    int n = (int)poly.size();
    if (n < 3) return;
    bool all_pos = true;
    for (double v : phi) { if (v < 0) { all_pos = false; break; } }
    if (all_pos) return;  // entirely outside – discard
    bool all_neg = true;
    for (double v : phi) { if (v >= 0) { all_neg = false; break; } }
    if (all_neg) { out.push_back(poly); return; }  // entirely inside – keep

    std::vector<R2> neg;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        if (phi[i] < 0) neg.push_back(poly[i]);
        if ((phi[i] < 0) != (phi[j] < 0)) {
            double t = phi[i] / (phi[i] - phi[j]);
            neg.push_back(R2(poly[i].x + t*(poly[j].x - poly[i].x),
                             poly[i].y + t*(poly[j].y - poly[i].y)));
        }
    }
    if ((int)neg.size() >= 3) out.push_back(std::move(neg));
}

// boundary_clips : one-sided (keep phi<0), used for the SOC outer boundary
// split_levelsets: two-sided (keep both sides), used for bone/cartilage and ossification interfaces
static void write_trimmed_both_vtk(
    const std::string &filename,
    const cutmesh_t &Khi,
    const std::vector<PhiFn> &boundary_clips,
    const std::vector<PhiFn> &split_levelsets,
    const std::vector<ScalarField> &sca_fields,
    const std::vector<ScalarField> &cell_fields = {},
    const std::vector<CellFn> &cell_fns = {},
    fct_t *vec_fh = nullptr,
    const std::string &vec_name = "")
{
    struct Cell { std::vector<R2> nodes; int kb; int domain = 0; };
    std::vector<Cell> cells;

    int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        if (Khi.isInactive(ka, 0)) continue;
        int kb = Khi.idxElementInBackMesh(ka);

        std::vector<std::vector<R2>> polys;
        { std::vector<R2> q; for (int i = 0; i < 4; ++i) q.push_back(Khi[ka][i]); polys.push_back(std::move(q)); }

        // One-sided clips first: trim to SOC interior, discard outside pieces
        for (auto &phi : boundary_clips) {
            std::vector<std::vector<R2>> next;
            for (auto &poly : polys) {
                std::vector<double> vals;
                vals.reserve(poly.size());
                for (auto &p : poly) vals.push_back(phi(p, kb));
                clip_polygon_keep_negative(poly, vals, next);
            }
            polys = std::move(next);
        }

        // Two-sided splits with domain tracking:
        // bit i of domain = 0 → cell is on phi < 0 side of split i
        // bit i of domain = 1 → cell is on phi >= 0 side of split i
        std::vector<int> domains(polys.size(), 0);
        for (int si = 0; si < (int)split_levelsets.size(); ++si) {
            auto &phi = split_levelsets[si];
            std::vector<std::vector<R2>> next;
            std::vector<int> next_domains;
            for (size_t pi = 0; pi < polys.size(); ++pi) {
                auto &poly = polys[pi];
                int n = (int)poly.size();
                if (n < 3) continue;
                std::vector<double> vals;
                vals.reserve(n);
                for (auto &p : poly) vals.push_back(phi(p, kb));

                bool all_neg = true, all_pos = true;
                for (double v : vals) { if (v >= 0) all_neg = false; if (v < 0) all_pos = false; }

                if (all_neg) {
                    next.push_back(std::move(poly));
                    next_domains.push_back(domains[pi]);
                } else if (all_pos) {
                    next.push_back(std::move(poly));
                    next_domains.push_back(domains[pi] | (1 << si));
                } else {
                    std::vector<R2> neg, pos;
                    for (int i = 0; i < n; ++i) {
                        int j = (i + 1) % n;
                        if (vals[i] < 0) neg.push_back(poly[i]);
                        else             pos.push_back(poly[i]);
                        if ((vals[i] < 0) != (vals[j] < 0)) {
                            double t = vals[i] / (vals[i] - vals[j]);
                            R2 x(poly[i].x + t*(poly[j].x - poly[i].x),
                                 poly[i].y + t*(poly[j].y - poly[i].y));
                            neg.push_back(x);
                            pos.push_back(x);
                        }
                    }
                    if ((int)neg.size() >= 3) {
                        next.push_back(std::move(neg));
                        next_domains.push_back(domains[pi]);
                    }
                    if ((int)pos.size() >= 3) {
                        next.push_back(std::move(pos));
                        next_domains.push_back(domains[pi] | (1 << si));
                    }
                }
            }
            polys = std::move(next);
            domains = std::move(next_domains);
        }

        for (size_t pi = 0; pi < polys.size(); ++pi)
            cells.push_back({std::move(polys[pi]), kb, domains[pi]});
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

    {
        out << "CELL_DATA " << cells.size() << "\n";

        // domain: bitmask from split_levelsets (bit i = side of split i)
        out << "SCALARS domain float\n"
            << "LOOKUP_TABLE default\n";
        for (auto &c : cells) out << c.domain << "\n";

        // material_sharp: derived from domain (exact, no interpolation)
        // split 0: P.y - interface_y  → bit 0 = 1 means above interface
        // split 1: phi_soc            → bit 1 = 1 means ossified
        // domain == 1 (above, not ossified) → cartilage (0), else → bone (1)
        out << "SCALARS material_sharp float\n"
            << "LOOKUP_TABLE default\n";
        for (auto &c : cells) out << ((c.domain == 1) ? 0.0 : 1.0) << "\n";

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

    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        const std::string default_path = "configs/growth_iterative_convex.json";
        if (std::filesystem::exists(default_path)) config_path = default_path;
    }

    if (!config_path.empty()) {
        if (!load_config_from_json(config_path, g_cfg)) {
            std::cerr << "Failed to load config file: " << config_path << "\n";
            return 1;
        }
        std::cout << "Loaded config: " << config_path << "\n";
    } else {
        std::cout << "No config provided. Using built-in defaults.\n";
    }

    if (g_cfg.bspline_weights.size() != 9) {
        std::cerr << "Invalid config: bspline_weights must have exactly 9 values.\n";
        return 1;
    }
    if (g_cfg.n_iterations < 1 || g_cfg.mesh_nx < 2 || g_cfg.transfinite_nodes < 2) {
        std::cerr << "Invalid config: n_iterations >= 1, mesh_nx >= 2, transfinite_nodes >= 2 are required.\n";
        return 1;
    }
    if (g_cfg.output_dir.empty()) {
        std::cerr << "Invalid config: output_dir cannot be empty.\n";
        return 1;
    }

    std::filesystem::create_directories(g_cfg.output_dir);
    build_soc_geometry();

    // ---- Mesh ----
    // Ensure nx is odd so x=0 falls exactly on a node line (symmetric domain)
    const int nx = (g_cfg.mesh_nx % 2 == 0) ? g_cfg.mesh_nx + 1 : g_cfg.mesh_nx;
    double dx_bg = bg_xmax - bg_xmin, dy_bg = bg_ymax - bg_ymin;
    int ny = static_cast<int>(nx * dy_bg / dx_bg);
    mesh_t Kh(nx, ny, bg_xmin, bg_ymin + g_cfg.mesh_y_offset, dx_bg, dy_bg);
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

    fct_t bottomInd(Sh, [](R2 P, int i, int dom) -> double {
        return (std::abs(P.y - y_bottom) < 0.05 &&
                P.x >= x_bottom_min - 0.05 &&
                P.x <= x_bottom_max + 0.05) ? 1.0 : 0.0;
    });

    // Level set FE functions (constant across iterations)
    std::vector<double> phi_outer_data(nb_sca), phi_iface_data(nb_sca);
    for (int k = 0; k < Kh.nt; ++k) {
        const auto &FK = Sh[k];
        for (int j = 0; j < FK.NbDoF(); ++j) {
            int iglo = Sh(k, j);
            if (iglo < 0 || iglo >= nb_sca) continue;
            R2 P = FK.Pt(j);
            phi_outer_data[iglo] = signed_distance_polygon(P, g_polygon);
            double phi_y = P.y - g_cfg.interface_y;
            // If a node sits exactly on the interface, nudge it below
            // to avoid degenerate CutFEM cuts
            if (std::abs(phi_y) < 1e-10) phi_y = -1e-10;
            phi_iface_data[iglo] = phi_y;
        }
    }
    std::span<double> phi_outer_span(phi_outer_data);
    fct_t phi_outer_fh(Sh, phi_outer_span);
    std::span<double> phi_iface_span(phi_iface_data);
    fct_t phi_iface_fh(Sh, phi_iface_span);

    // ============================================================
    // Solve all load steps for given material, return averaged fields
    // ============================================================
    struct RunResult {
        std::vector<double> U_avg;
        std::vector<std::vector<double>> all_sols;
        StressFields sf_avg;
    };

    auto run_elasticity = [&](const fct_t &two_mu_fh, const fct_t &lambda_fh) -> RunResult
    {
        auto steps = build_default_steps();
        std::vector<double> U_sum(nb_dof, 0.0);
        std::vector<double> hd_sum(nb_sca, 0.0), oct_sum(nb_sca, 0.0), mi_sum(nb_sca, 0.0);
        std::vector<std::vector<double>> all_sols;

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

            {
                std::span<double> sp(sol);
                fct_t uh_step(Wh, sp);
                auto sf_step = compute_stress_fields(uh_step, Sh, Khi, Kh, nb_sca,
                                                     two_mu_fh, lambda_fh);
                for (int i = 0; i < nb_sca; ++i) {
                    hd_sum[i]  += sf_step.hydrostatic[i];
                    oct_sum[i] += sf_step.oct_shear[i];
                    mi_sum[i]  += sf_step.miner[i];
                }
            }

            all_sols.push_back(std::move(sol));
            std::cout << "    Step " << sidx+1 << "/" << steps.size() << " done.\n";
        }

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
        res.sf_avg.hydrostatic = std::move(hd_sum);
        res.sf_avg.oct_shear   = std::move(oct_sum);
        res.sf_avg.miner       = std::move(mi_sum);
        return res;
    };

    // ============================================================
    // Export helper: write all VTK outputs for one iteration
    // ============================================================
    auto export_iteration = [&](int iter,
                                const RunResult &result,
                                const fct_t &uh_avg,
                                const fct_t &hd_fh, const fct_t &oct_fh,
                                const fct_t &mi_fh, const fct_t &mat_fh,
                                fct_t &phi_soc_fh,
                                const std::vector<double> &phi_oss_data,
                                fct_t *poisson_fh = nullptr)
    {
        std::string tag = std::to_string(iter);
        const std::string output_prefix = std::filesystem::path(g_cfg.output_dir).filename().string();
        const std::string iter_prefix = output_prefix + "_iter_" + tag;

        // Per-step outputs
        for (unsigned int sidx = 0; sidx < result.all_sols.size(); ++sidx) {
            std::span<double> sp(const_cast<double*>(result.all_sols[sidx].data()),
                                 result.all_sols[sidx].size());
            fct_t uh(Wh, sp);
            Paraview<mesh_t> w(Khi, g_cfg.output_dir + "/" + iter_prefix + "_step_" + std::to_string(sidx+1) + ".vtk");
            w.add(uh, "displacement", 0, 2);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Fields (trimmed at SOC boundary)
        {
            Paraview<mesh_t> w(Khi, g_cfg.output_dir + "/" + iter_prefix + "_fields.vtk");
            w.add(const_cast<fct_t&>(uh_avg), "displacement", 0, 2);
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
            if (poisson_fh) w.add(*poisson_fh, "poisson_u", 0, 1);
        }

        // Active elements (full quads)
        if (g_cfg.export_active) {
            Paraview<mesh_t> w;
            w.writeActiveMesh(Khi, g_cfg.output_dir + "/" + iter_prefix + "_active.vtk");
            w.add(const_cast<fct_t&>(uh_avg), "displacement", 0, 2);
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
            if (poisson_fh) w.add(*poisson_fh, "poisson_u", 0, 1);
        }

        // Trimmed both sides
        if (g_cfg.export_trimmed) {
        // One-sided: trim outside the SOC outer boundary (discard outside pieces of cut elements)
        std::vector<PhiFn> soc_boundary_clip = {
            [](const R2& P, int) { return signed_distance_polygon(P, g_polygon); }
        };
        // Two-sided: keep both sub-cells at each material interface
        std::vector<PhiFn> interface_splits = {
            [](const R2& P, int) { return P.y - g_cfg.interface_y; }
        };
        interface_splits.push_back(
            [&phi_soc_fh](const R2& P, int kb) {
                return phi_soc_fh.evalOnBackMesh(kb, 0, &P.x, 0, 0);
            });

        std::vector<ScalarField> sca = {
            {const_cast<fct_t*>(&hd_fh), "hydrostatic"},
            {const_cast<fct_t*>(&oct_fh), "oct_shear"},
            {const_cast<fct_t*>(&mi_fh), "miner_index"},
            {&phi_outer_fh, "phi_outer"},
            {&phi_iface_fh, "phi_interface"},
            {&phi_soc_fh, "phi_soc"}
        };
        if (poisson_fh) sca.push_back({poisson_fh, "poisson_u"});

        write_trimmed_both_vtk(
            g_cfg.output_dir + "/" + iter_prefix + "_trimmed.vtk", Khi,
            soc_boundary_clip, interface_splits, sca,
            {{const_cast<fct_t*>(&mat_fh), "material"}},
            {},
            const_cast<fct_t*>(&uh_avg), "displacement");
        }

        // Cartilage only
        if (g_cfg.export_cartilage) {
            std::vector<double> phi_cart_data(nb_sca);
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    double phi_out = signed_distance_polygon(P, g_polygon);
                    double phi_y   = g_cfg.interface_y - P.y;
                    double phi_c   = std::max(phi_out, phi_y);
                    if (iter > 0) phi_c = std::max(phi_c, phi_oss_data[iglo]);
                    phi_cart_data[iglo] = phi_c;
                }
            }
            std::span<double> phi_cart_span(phi_cart_data);
            fct_t phi_cart(Sh, phi_cart_span);
            InterfaceLevelSet<mesh_t> cart_interface(Kh, phi_cart);

            cutmesh_t Khi_cart(Kh);
            Khi_cart.truncate(cart_interface, 1);

            Paraview<mesh_t> w(Khi_cart, g_cfg.output_dir + "/" + iter_prefix + "_cartilage.vtk");
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
            if (poisson_fh) w.add(*poisson_fh, "poisson_u", 0, 1);
        }
    };

    // ============================================================
    // Critical point detection (gradient + Laplacian classification)
    // ============================================================
    struct GradientField { std::vector<double> gx, gy, mag; };

    auto find_critical_points = [&](const std::vector<double> &field) -> std::vector<double>
    {
        std::vector<std::vector<int>> adj(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            int ndf = FK.NbDoF();
            for (int i = 0; i < ndf; ++i) {
                int gi = Sh(k, i);
                if (gi < 0 || gi >= nb_sca) continue;
                for (int j = 0; j < ndf; ++j) {
                    if (i == j) continue;
                    int gj = Sh(k, j);
                    if (gj < 0 || gj >= nb_sca) continue;
                    adj[gi].push_back(gj);
                }
            }
        }
        for (auto &v : adj) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }

        std::vector<R2> dof_pos(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo >= 0 && iglo < nb_sca)
                    dof_pos[iglo] = FK.Pt(j);
            }
        }

        std::vector<double> crit(nb_sca, 0.0);
        for (int i = 0; i < nb_sca; ++i) {
            R2 Pi = dof_pos[i];
            if (Pi.y < g_cfg.interface_y) continue;
            if (signed_distance_polygon(Pi, g_polygon) > 0.0) continue;
            if (adj[i].size() < 3) continue;

            std::vector<std::pair<double, int>> angle_nb;
            for (int nb : adj[i]) {
                R2 Pn = dof_pos[nb];
                double ang = std::atan2(Pn.y - Pi.y, Pn.x - Pi.x);
                angle_nb.push_back({ang, nb});
            }
            std::sort(angle_nb.begin(), angle_nb.end());

            bool all_below = true, all_above = true;
            for (auto &[ang, nb] : angle_nb) {
                if (field[nb] >= field[i]) all_below = false;
                if (field[nb] <= field[i]) all_above = false;
            }

            if (all_below) { crit[i] = 1.0; continue; }   // local maximum
            if (all_above) { crit[i] = -1.0; continue; }   // local minimum

            std::vector<int> signs;
            for (auto &[ang, nb] : angle_nb) {
                double diff = field[nb] - field[i];
                if (diff > 0) signs.push_back(1);
                else if (diff < 0) signs.push_back(-1);
            }
            if ((int)signs.size() < 4) continue;

            int n_changes = 0;
            for (int j = 1; j < (int)signs.size(); ++j)
                if (signs[j] != signs[j-1]) ++n_changes;
            if (signs.back() != signs.front()) ++n_changes;

            if (n_changes >= 4) crit[i] = 0.5; // saddle point
        }
        return crit;
    };

    auto compute_laplacian = [&](const std::vector<double> &field) -> std::vector<double>
    {
        std::vector<std::vector<int>> adj(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            int ndf = FK.NbDoF();
            for (int i = 0; i < ndf; ++i) {
                int gi = Sh(k, i);
                if (gi < 0 || gi >= nb_sca) continue;
                for (int j = 0; j < ndf; ++j) {
                    if (i == j) continue;
                    int gj = Sh(k, j);
                    if (gj < 0 || gj >= nb_sca) continue;
                    adj[gi].push_back(gj);
                }
            }
        }
        for (auto &v : adj) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }

        std::vector<double> lap(nb_sca, 0.0);
        for (int i = 0; i < nb_sca; ++i) {
            if (adj[i].empty()) continue;
            double sum = 0.0;
            for (int nb : adj[i])
                sum += field[nb] - field[i];
            lap[i] = sum / (double)adj[i].size();
        }
        return lap;
    };

    // ============================================================
    // Iterative loop
    // ============================================================
    auto phase_schedule = build_phase_schedule(g_cfg.n_iterations);
    std::vector<double> prev_phi_oss_data(nb_sca, -1.0);
    double mi_threshold = 0.0;

    // Persistent storage for last MI-phase results (reused in Poisson-phase export)
    std::vector<double> last_U_avg(nb_dof, 0.0);
    std::vector<double> last_hd(nb_sca, 0.0), last_oct(nb_sca, 0.0), last_mi(nb_sca, 0.0);
    std::vector<double> last_mat(nb_sca, 0.0);
    std::vector<double> pre_mi_phi_oss_data(nb_sca, -1.0); // state before last MI update
    RunResult last_mi_result;

    for (int iter = 0; iter < g_cfg.n_iterations; ++iter) {
        Phase phase = phase_schedule[iter];
        std::cout << "\n=== Iteration " << iter << " | Phase = " << phase_name(phase) << " ===\n";

        std::vector<double> phi_oss_data(nb_sca, -1.0);

        if (phase == Phase::MI) {
        // ---- MI PHASE: mechanical solve -> MI field -> threshold -> level set ----

        std::vector<double> tmu_data(nb_sca), lam_data(nb_sca);
        std::vector<double> mat_data(nb_sca, 0.0);

        // Base material, with carry-over ossification from previous iteration
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                const bool ossified_prev = (iter > 0 && prev_phi_oss_data[iglo] >= 0.0);
                if (P.y < g_cfg.interface_y || ossified_prev) {
                    tmu_data[iglo] = 2.0 * mu_bone;
                    lam_data[iglo] = lambda_bone;
                    mat_data[iglo] = 1.0; // bone
                } else {
                    tmu_data[iglo] = 2.0 * mu_cart;
                    lam_data[iglo] = lambda_cart;
                    mat_data[iglo] = 0.0; // cartilage
                }
            }
        }

        // Build FE functions for material
        std::span<double> tmu_span(tmu_data);
        fct_t two_mu_fh(Sh, tmu_span);
        std::span<double> lam_span(lam_data);
        fct_t lambda_fh(Sh, lam_span);

        // Solve
        auto result = run_elasticity(two_mu_fh, lambda_fh);

        // Build averaged field FE functions
        std::span<double> sp_avg(result.U_avg);
        fct_t uh_avg(Wh, sp_avg);
        std::span<double> hd_span(result.sf_avg.hydrostatic);
        fct_t hd_fh(Sh, hd_span);
        std::span<double> oct_span(result.sf_avg.oct_shear);
        fct_t oct_fh(Sh, oct_span);
        std::span<double> mi_span(result.sf_avg.miner);
        fct_t mi_fh(Sh, mi_span);

        // mat_fh reflects the material used in this iteration's solve (current state)
        std::span<double> mat_span(mat_data);
        fct_t mat_fh(Sh, mat_span);

        std::cout << "  DOFs (elasticity) = " << nb_dof << "\n";

        // ---- Critical point analysis on MI field ----
        auto crit_data = find_critical_points(result.sf_avg.miner);
        auto mi_laplacian = compute_laplacian(result.sf_avg.miner);

        // Build DOF positions for critical point export and SOC distance
        std::vector<R2> dof_pos(nb_sca);
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo >= 0 && iglo < nb_sca)
                    dof_pos[iglo] = FK.Pt(j);
            }
        }

        // Collect local maxima (crit == 1.0), select the highest MI near the SOC surface
        // Status: 0=unused, 1=candidate max, 2=selected max
        std::vector<double> crit_status(nb_sca, 0.0);

        // Gather all maxima with their distance to ossification front
        struct MaxCandidate { int iglo; double mi_val; double dist_to_oss; };
        std::vector<MaxCandidate> max_candidates;

        // Collect ossified node positions for distance computation
        std::vector<R2> oss_front_pts;
        for (int i = 0; i < nb_sca; ++i) {
            if (prev_phi_oss_data[i] >= 0.0)
                oss_front_pts.push_back(dof_pos[i]);
        }

        int n_max = 0, n_min = 0, n_saddle = 0;
        for (int i = 0; i < nb_sca; ++i) {
            if (crit_data[i] > 0.9) {
                ++n_max;
                crit_status[i] = 1.0;
                double dist_oss = std::numeric_limits<double>::max();
                for (auto &op : oss_front_pts) {
                    double dx = dof_pos[i].x - op.x;
                    double dy = dof_pos[i].y - op.y;
                    dist_oss = std::min(dist_oss, std::sqrt(dx*dx + dy*dy));
                }
                max_candidates.push_back({i, result.sf_avg.miner[i], dist_oss});
            } else if (crit_data[i] < -0.9) {
                ++n_min;
            } else if (std::abs(crit_data[i] - 0.5) < 0.1) {
                ++n_saddle;
            }
        }

        // Select best candidate: weighted score of MI value and proximity to ossification front
        double best_mi_val = -std::numeric_limits<double>::max();
        int best_iglo = -1;
        if (!oss_front_pts.empty()) {
            double best_score = -std::numeric_limits<double>::max();
            for (auto &c : max_candidates) {
                double score = c.mi_val / (1.0 + c.dist_to_oss / g_cfg.oss_spline_band);
                if (score > best_score) {
                    best_score = score;
                    best_iglo = c.iglo;
                    best_mi_val = c.mi_val;
                }
            }
        } else {
            // No ossification front yet, select highest MI
            for (auto &c : max_candidates) {
                if (c.mi_val > best_mi_val) {
                    best_mi_val = c.mi_val;
                    best_iglo = c.iglo;
                }
            }
        }
        if (best_iglo >= 0)
            crit_status[best_iglo] = 2.0; // selected

        std::cout << "  Critical points: " << n_max << " max, " << n_min << " min, "
                  << n_saddle << " saddle\n";

        if (iter == 0) {
            // Iteration 0: use target-height bisection approach
            mi_threshold = find_threshold(result.sf_avg.miner, Sh, Kh);
            std::cout << "  Threshold (MI, target-height) = " << mi_threshold << "\n";
        } else if (best_iglo >= 0) {
            mi_threshold = g_cfg.mi_threshold_frac * best_mi_val;
            double sel_dist_oss = 0.0;
            for (auto &c : max_candidates) {
                if (c.iglo == best_iglo) { sel_dist_oss = c.dist_to_oss; break; }
            }
            std::cout << "  Selected max: node " << best_iglo
                      << " at (" << dof_pos[best_iglo].x << ", " << dof_pos[best_iglo].y << ")"
                      << ", MI=" << best_mi_val
                      << ", dist_to_oss_front=" << sel_dist_oss << "\n";
            std::cout << "  Threshold (MI) = " << g_cfg.mi_threshold_frac << " * " << best_mi_val
                      << " = " << mi_threshold << "\n";
        } else {
            std::cout << "  WARNING: no local maxima found, keeping previous threshold = "
                      << mi_threshold << "\n";
        }

        // Export critical points as VTK point cloud
        {
            std::string tag = std::to_string(iter);
            const std::string output_prefix = std::filesystem::path(g_cfg.output_dir).filename().string();
            std::string fname = g_cfg.output_dir + "/" + output_prefix + "_iter_" + tag + "_crit_points.vtk";

            std::vector<R2> pts;
            std::vector<double> types, statuses, laps, mi_vals;
            for (int i = 0; i < nb_sca; ++i) {
                if (std::abs(crit_data[i]) < 0.01) continue;
                pts.push_back(dof_pos[i]);
                types.push_back(crit_data[i]);
                statuses.push_back(crit_status[i]);
                laps.push_back(mi_laplacian[i]);
                mi_vals.push_back(result.sf_avg.miner[i]);
            }

            std::ofstream ofs(fname);
            ofs << "# vtk DataFile Version 3.0\n"
                << "Critical points of miner index\n"
                << "ASCII\n"
                << "DATASET POLYDATA\n"
                << "POINTS " << pts.size() << " double\n";
            for (auto &P : pts)
                ofs << P.x << " " << P.y << " 0.0\n";
            ofs << "VERTICES " << pts.size() << " " << 2 * pts.size() << "\n";
            for (int i = 0; i < (int)pts.size(); ++i)
                ofs << "1 " << i << "\n";
            ofs << "POINT_DATA " << pts.size() << "\n";
            ofs << "SCALARS crit_type double 1\nLOOKUP_TABLE default\n";
            for (double v : types) ofs << v << "\n";
            ofs << "SCALARS status double 1\nLOOKUP_TABLE default\n";
            for (double v : statuses) ofs << v << "\n";
            ofs << "SCALARS laplacian double 1\nLOOKUP_TABLE default\n";
            for (double v : laps) ofs << v << "\n";
            ofs << "SCALARS miner_index double 1\nLOOKUP_TABLE default\n";
            for (double v : mi_vals) ofs << v << "\n";
            ofs.close();
            std::cout << "  Exported " << pts.size() << " critical points to " << fname << "\n";
        }

        // Save state before MI level set update (for Poisson newly-ossified detection)
        pre_mi_phi_oss_data = prev_phi_oss_data;

        // Update level set from MI field
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                if (P.y < g_cfg.interface_y) { phi_oss_data[iglo] = -1.0; continue; }
                if (signed_distance_polygon(P, g_polygon) > 0.0) { phi_oss_data[iglo] = -1.0; continue; }

                double dist_iface = P.y - g_cfg.interface_y;
                double t_iface = std::clamp(dist_iface / g_cfg.oss_spline_band, 0.0, 1.0);
                double inh_iface = t_iface * t_iface * (3.0 - 2.0 * t_iface);

                double phi_new = result.sf_avg.miner[iglo] * inh_iface - mi_threshold;
                phi_oss_data[iglo] = std::max(prev_phi_oss_data[iglo], phi_new);
            }
        }

        // Export MI phase
        std::span<double> prev_span(prev_phi_oss_data);
        fct_t phi_soc_fh(Sh, prev_span);
        export_iteration(iter, result, uh_avg, hd_fh, oct_fh, mi_fh, mat_fh,
                         phi_soc_fh, prev_phi_oss_data);

        // Store for Poisson-phase reuse
        last_U_avg = result.U_avg;
        last_hd = result.sf_avg.hydrostatic;
        last_oct = result.sf_avg.oct_shear;
        last_mi = result.sf_avg.miner;
        last_mat = mat_data;
        last_mi_result.all_sols = std::move(result.all_sols);

        } else {
        // ---- POISSON PHASE: solve Poisson -> threshold -> level set ----

        std::vector<double> poisson_data(nb_sca, 0.0);
        int nb_dof_p = 0;

        int n_prev_ossified = 0;
        for (int i = 0; i < nb_sca; ++i)
            if (prev_phi_oss_data[i] >= 0.0) ++n_prev_ossified;

        if (n_prev_ossified > 0) {
            std::cout << "  Solving Poisson on cartilage domain...\n";

            // Upper SOC domain: max(phi_outer, interface_y - y) < 0
            std::vector<double> phi_upper_data(nb_sca);
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    phi_upper_data[iglo] = std::max(
                        signed_distance_polygon(P, g_polygon),
                        g_cfg.interface_y - P.y);
                }
            }
            std::span<double> phi_upper_span(phi_upper_data);
            fct_t phi_upper_fh(Sh, phi_upper_span);
            InterfaceLevelSet<mesh_t> upper_interface(Kh, phi_upper_fh);

            cutmesh_t Khi_upper(Kh);
            Khi_upper.truncate(upper_interface, 1);

            // Ossification level set from previous iteration
            // Restrict to Khi_upper: force nodes outside the upper domain
            // to -1 so zero crossings only exist within Khi_upper
            std::vector<double> oss_perturbed(prev_phi_oss_data);
            for (int i = 0; i < nb_sca; ++i) {
                if (phi_upper_data[i] > 0.0)
                    oss_perturbed[i] = -1.0;
                else if (std::abs(oss_perturbed[i]) < 1e-10)
                    oss_perturbed[i] = -1e-10;
            }
            std::span<double> oss_p_span(oss_perturbed);
            fct_t phi_oss_fh(Sh, oss_p_span);
            InterfaceLevelSet<mesh_t> oss_interface(Kh, phi_oss_fh);

            // Scalar P1 CutFE space
            cutspace_t Wh_p(Khi_upper, Sh);
            nb_dof_p = Wh_p.get_nb_dof();

            CutFEM<mesh_t> poisson(Wh_p);
            funtest_t u_p(Wh_p, 1, 0), v_p(Wh_p, 1, 0);

            // ∫ ∇u·∇v + α∫ u·v
            poisson.addBilinear(
                innerProduct(grad(u_p), grad(v_p))
                + innerProduct(g_cfg.poisson_decay * u_p, v_p),
                Khi_upper);

            // Flux from ossification front: ∫ flux·v ds
            fct_t flux_fh(Sh, [](R2 P, int i, int dom) -> double {
                return g_cfg.poisson_flux;
            });
            poisson.addLinear(
                innerProduct(flux_fh.expr(), v_p),
                oss_interface);

            // Source term: ∫ f·v in cartilage (phi_oss < 0)
            fct_t source_fh(Sh, [](R2 P, int i, int dom) -> double {
                return g_cfg.poisson_source;
            });
            std::vector<double> cart_mask_vec(nb_sca, 0.0);
            for (int i = 0; i < nb_sca; ++i)
                if (prev_phi_oss_data[i] < 0.0) cart_mask_vec[i] = 1.0;
            std::span<double> cart_mask_span(cart_mask_vec);
            fct_t oss_mask_fh(Sh, cart_mask_span);
            poisson.addLinear(
                innerProduct(oss_mask_fh.expr() * source_fh.expr(), v_p),
                Khi_upper);

            // Dirichlet u=0 on domain boundaries via Nitsche
            // upper_interface is the combined boundary of Khi_upper (outer + iface),
            // so all its cut elements are guaranteed to be in Khi_upper.
            if (g_cfg.poisson_dirichlet_outer && g_cfg.poisson_dirichlet_iface) {
                // Dirichlet on entire boundary
                poisson.addBilinear(
                    -innerProduct(grad(u_p)*n, v_p)
                    -innerProduct(u_p, grad(v_p)*n)
                    +innerProduct((20.0/h) * u_p, v_p),
                    upper_interface);
            } else if (g_cfg.poisson_dirichlet_outer) {
                // Restrict outer level set: force positive below interface_y
                // to avoid zero crossings outside Khi_upper
                std::vector<double> phi_outer_restr(phi_outer_data);
                for (int i = 0; i < nb_sca; ++i)
                    if (phi_iface_data[i] < 0.0) phi_outer_restr[i] = std::max(phi_outer_restr[i], 1.0);
                std::span<double> phi_o_span(phi_outer_restr);
                fct_t phi_o_fh(Sh, phi_o_span);
                InterfaceLevelSet<mesh_t> outer_interface(Kh, phi_o_fh);
                poisson.addBilinear(
                    -innerProduct(grad(u_p)*n, v_p)
                    -innerProduct(u_p, grad(v_p)*n)
                    +innerProduct((20.0/h) * u_p, v_p),
                    outer_interface);
            } else if (g_cfg.poisson_dirichlet_iface) {
                // Use upper_interface (applies on full boundary);
                // the outer part becomes natural Neumann by not penalizing it
                // TODO: implement selective masking for iface-only Dirichlet
                poisson.addBilinear(
                    -innerProduct(grad(u_p)*n, v_p)
                    -innerProduct(u_p, grad(v_p)*n)
                    +innerProduct((20.0/h) * u_p, v_p),
                    upper_interface);
            }

            // Ghost-penalty stabilization
            poisson.addFaceStabilization(
                innerProduct(ghost_param * pow(h,-1) * jump(u_p), jump(v_p))
                + innerProduct(ghost_param * pow(h,1) * jump(grad(u_p)*n), jump(grad(v_p)*n)),
                Khi_upper);

            poisson.solve("umfpack");

            // Project CutFE solution onto Sh
            std::span<double> poisson_sol(poisson.rhs_.data(), nb_dof_p);
            fct_t poisson_cut(Wh_p, poisson_sol);

            int nact_upper = Khi_upper.get_nb_element();
            for (int ka = 0; ka < nact_upper; ++ka) {
                int kb = Khi_upper.idxElementInBackMesh(ka);
                const auto &FK = Sh[kb];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(kb, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    poisson_data[iglo] = poisson_cut.eval(ka, (const double*)&P, 0, 0);
                }
            }
            std::cout << "  Poisson solved, DOFs=" << nb_dof_p << "\n";
        }

        // Poisson threshold: minimum Poisson value on newly ossified boundary
        double poisson_thresh = compute_poisson_threshold(
            poisson_data, prev_phi_oss_data, pre_mi_phi_oss_data, nb_sca);
        std::cout << "  Threshold (Poisson) = " << poisson_thresh << "\n";
        std::cout << "  DOFs (Poisson) = " << nb_dof_p << "\n";

        // Update level set from Poisson field
        for (int k = 0; k < Kh.nt; ++k) {
            const auto &FK = Sh[k];
            for (int j = 0; j < FK.NbDoF(); ++j) {
                int iglo = Sh(k, j);
                if (iglo < 0 || iglo >= nb_sca) continue;
                R2 P = FK.Pt(j);
                if (P.y < g_cfg.interface_y) { phi_oss_data[iglo] = -1.0; continue; }
                if (signed_distance_polygon(P, g_polygon) > 0.0) { phi_oss_data[iglo] = -1.0; continue; }

                double phi_new = poisson_data[iglo] - poisson_thresh;
                phi_oss_data[iglo] = std::max(prev_phi_oss_data[iglo], phi_new);
            }
        }

        // Export using last MI fields + new Poisson
        std::span<double> poisson_span(poisson_data);
        fct_t poisson_sh(Sh, poisson_span);

        std::span<double> u_span(last_U_avg);
        fct_t uh_avg(Wh, u_span);
        std::span<double> hd_span(last_hd);
        fct_t hd_fh(Sh, hd_span);
        std::span<double> oct_span(last_oct);
        fct_t oct_fh(Sh, oct_span);
        std::span<double> mi_span(last_mi);
        fct_t mi_fh(Sh, mi_span);
        std::span<double> mat_span(last_mat);
        fct_t mat_fh(Sh, mat_span);

        std::span<double> prev_span(prev_phi_oss_data);
        fct_t phi_soc_fh(Sh, prev_span);

        RunResult empty_result;
        export_iteration(iter, empty_result, uh_avg, hd_fh, oct_fh, mi_fh, mat_fh,
                         phi_soc_fh, prev_phi_oss_data, &poisson_sh);

        } // end phase branch

        // Common logging
        int n_ossified = 0;
        for (int i = 0; i < nb_sca; ++i)
            if (phi_oss_data[i] >= 0.0) ++n_ossified;
        std::cout << "  Ossified nodes: " << n_ossified << " / " << nb_sca << "\n";

        prev_phi_oss_data = phi_oss_data;
        std::cout << "  Iteration " << iter << " done.\n";
    }

    std::cout << "\nDone. Output in " << g_cfg.output_dir << "/\n";
    return 0;
}
