/**
 * @brief Iterative growth/ossification with CutFEM-Library.
 *
 * Single-run iterative workflow on SOC domain:
 *   Iteration 0: bi-material (bone + cartilage) → MI → threshold → ossified region
 *   Iteration 1: ossified region gets intermediate props → new MI → new ossified region
 *   Iteration 2: iter-1 ossified becomes bone, new ossified from iter-2 MI
 *   ... and so on for N_ITERATIONS
 *
 * Materials:
 *   bone      (y < 1.0):            E=500,  ν=0.2
 *   cartilage (y ≥ 1.0, not oss):   E=6,    ν=0.47
 *   ossified  (new, current iter):   E=253,  ν=0.335
 *   matured   (ossified from prev):  becomes bone (E=500, ν=0.2)
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

    std::string output_dir = "output_growth_iterative";
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

static constexpr double E_oss  = 0.5 * (E_bone + E_cart);
static constexpr double nu_oss = 0.5 * (nu_bone + nu_cart);
static constexpr double mu_oss     = E_oss / (2.0 * (1.0 + nu_oss));
static constexpr double lambda_oss = E_oss * nu_oss / ((1.0 + nu_oss) * (1.0 - 2.0 * nu_oss));

// Ossification parameters

struct StepDef { double u_center, u_radius, p_peak; };

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
    return {vm, hd, oct, oct + g_cfg.k_mi * hd};
}

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
    std::vector<double> hd_sum(nb_sca_dof, 0.0), oct_sum(nb_sca_dof, 0.0), mi_sum(nb_sca_dof, 0.0);
    std::vector<int>    cnt(nb_sca_dof, 0);

    int nact = Khi.get_nb_element();
    for (int ka = 0; ka < nact; ++ka) {
        int kb = Khi.idxElementInBackMesh(ka);
        const auto &FK = Sh[kb];
        int ndf = FK.NbDoF();

        R2 centroid(0.0, 0.0);
        for (int j = 0; j < ndf; ++j) {
            R2 Pj = FK.Pt(j);
            centroid.x += Pj.x;
            centroid.y += Pj.y;
        }
        centroid.x /= ndf;
        centroid.y /= ndf;

        double du0_dx = uh.eval(ka, (const double*)&centroid, 0, 1);
        double du0_dy = uh.eval(ka, (const double*)&centroid, 0, 2);
        double du1_dx = uh.eval(ka, (const double*)&centroid, 1, 1);
        double du1_dy = uh.eval(ka, (const double*)&centroid, 1, 2);

        double exx = du0_dx, eyy = du1_dy;
        double exy = 0.5 * (du0_dy + du1_dx);

        // Use actual material properties at this element
        double lam = lambda_fh.evalOnBackMesh(kb, 0, centroid, 0, 0);
        double mu  = two_mu_fh.evalOnBackMesh(kb, 0, centroid, 0, 0) * 0.5;

        auto inv = compute_invariants(exx, eyy, exy, lam, mu);

        for (int j = 0; j < ndf; ++j) {
            int iglo = Sh(kb, j);
            if (iglo < 0 || iglo >= nb_sca_dof) continue;
            hd_sum[iglo]  += inv.hydrostatic;
            oct_sum[iglo] += inv.oct_shear;
            mi_sum[iglo]  += inv.miner;
            cnt[iglo]     += 1;
        }
    }

    StressFields sf;
    sf.hydrostatic.resize(nb_sca_dof, 0.0);
    sf.oct_shear.resize(nb_sca_dof, 0.0);
    sf.miner.resize(nb_sca_dof, 0.0);
    for (int i = 0; i < nb_sca_dof; ++i) {
        if (cnt[i] > 0) {
            double inv = 1.0 / cnt[i];
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

        std::vector<std::vector<R2>> polys;
        { std::vector<R2> q; for (int i = 0; i < 4; ++i) q.push_back(Khi[ka][i]); polys.push_back(std::move(q)); }

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

    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        const std::string default_path = "configs/growth_iterative_default.json";
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
    const int nx = g_cfg.mesh_nx;
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
            phi_iface_data[iglo] = P.y - g_cfg.interface_y;
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
                                const std::vector<double> &phi_oss_data)
    {
        std::string tag = std::to_string(iter);

        // Per-step outputs
        for (unsigned int sidx = 0; sidx < result.all_sols.size(); ++sidx) {
            std::span<double> sp(const_cast<double*>(result.all_sols[sidx].data()),
                                 result.all_sols[sidx].size());
            fct_t uh(Wh, sp);
            Paraview<mesh_t> w(Khi, g_cfg.output_dir + "/growth_" + tag + "-step-" + std::to_string(sidx+1) + ".vtk");
            w.add(uh, "displacement", 0, 2);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Fields (trimmed at SOC boundary)
        {
            Paraview<mesh_t> w(Khi, g_cfg.output_dir + "/growth-fields_" + tag + ".vtk");
            w.add(const_cast<fct_t&>(uh_avg), "displacement", 0, 2);
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Active elements (full quads)
        if (g_cfg.export_active) {
            Paraview<mesh_t> w;
            w.writeActiveMesh(Khi, g_cfg.output_dir + "/growth-active_" + tag + ".vtk");
            w.add(const_cast<fct_t&>(uh_avg), "displacement", 0, 2);
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }

        // Trimmed both sides
        if (g_cfg.export_trimmed) {
        std::vector<PhiFn> clip_levelsets = {
            [](const R2& P, int) { return signed_distance_polygon(P, g_polygon); },
            [](const R2& P, int) { return P.y - g_cfg.interface_y; }
        };
        if (iter > 0) {
            clip_levelsets.push_back(
                [&phi_soc_fh](const R2& P, int kb) {
                    return phi_soc_fh.evalOnBackMesh(kb, 0, &P.x, 0, 0);
                });
        }

        std::vector<ScalarField> sca = {
            {const_cast<fct_t*>(&hd_fh), "hydrostatic"},
            {const_cast<fct_t*>(&oct_fh), "oct_shear"},
            {const_cast<fct_t*>(&mi_fh), "miner_index"},
            {&phi_outer_fh, "phi_outer"},
            {&phi_iface_fh, "phi_interface"},
            {&phi_soc_fh, "phi_soc"}
        };

        // Build material_sharp lambda based on iteration
        CellFn mat_sharp;
        if (iter == 0) {
            mat_sharp = {[](const R2& P, int) -> double {
                return P.y < g_cfg.interface_y ? 0.0 : 1.0;
            }, "material_sharp"};
        } else {
            mat_sharp = {[&phi_soc_fh](const R2& P, int kb) -> double {
                if (P.y < g_cfg.interface_y) return 0.0;
                if (phi_soc_fh.evalOnBackMesh(kb, 0, &P.x, 0, 0) >= 0.0) return 2.0;
                return 1.0;
            }, "material_sharp"};
        }

        write_trimmed_both_vtk(
            g_cfg.output_dir + "/growth-trimmed_" + tag + ".vtk", Khi,
            clip_levelsets, sca,
            {{const_cast<fct_t*>(&mat_fh), "material"}},
            {mat_sharp},
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

            Paraview<mesh_t> w(Khi_cart, g_cfg.output_dir + "/growth-cartilage_" + tag + ".vtk");
            w.add(const_cast<fct_t&>(hd_fh), "hydrostatic", 0, 1);
            w.add(const_cast<fct_t&>(oct_fh), "oct_shear", 0, 1);
            w.add(const_cast<fct_t&>(mi_fh), "miner_index", 0, 1);
            w.add(const_cast<fct_t&>(mat_fh), "material", 0, 1);
            w.add(phi_outer_fh, "phi_outer", 0, 1);
            w.add(phi_iface_fh, "phi_interface", 0, 1);
            w.add(phi_soc_fh, "phi_soc", 0, 1);
        }
    };

    // ============================================================
    // Iterative loop
    // ============================================================
    // Per-node state tracking which nodes have been ossified in previous iterations
    // matured[i] = true means this node was ossified in a previous iteration
    //              and now has bone properties
    std::vector<bool> matured(nb_sca, false);
    std::vector<double> prev_phi_oss_data(nb_sca, -1.0); // φ_soc from previous iteration (negative = no ossification)
    std::vector<double> prev_mi_avg(nb_sca, 0.0);       // MI from previous iteration
    double prev_threshold = 0.0;                         // threshold from previous iteration

    for (int iter = 0; iter < g_cfg.n_iterations; ++iter) {
        std::cout << "\n=== Iteration " << iter << " ===\n";

        // Build material coefficients for this iteration:
        //   bone region (y < interfaceY):  always bone
        //   matured nodes (from prev):     bone properties
        //   ossified nodes (from prev):    intermediate → now bone (matured)
        //   remaining cartilage:           cartilage
        //
        // For iter 0: just bone + cartilage
        // For iter 1: prev ossified region → intermediate props
        // For iter 2+: prev-prev ossified → bone, prev ossified → intermediate

        std::vector<double> tmu_data(nb_sca), lam_data(nb_sca);
        std::vector<double> mat_data(nb_sca, 0.0);

        // Current ossification level set data (from previous iteration's MI)
        std::vector<double> phi_oss_data(nb_sca, -1.0);

        if (iter == 0) {
            // Pure bone + cartilage
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    if (P.y < g_cfg.interface_y) {
                        tmu_data[iglo] = 2.0 * mu_bone;
                        lam_data[iglo] = lambda_bone;
                        mat_data[iglo] = 0.0; // bone
                    } else {
                        tmu_data[iglo] = 2.0 * mu_cart;
                        lam_data[iglo] = lambda_cart;
                        mat_data[iglo] = 1.0; // cartilage
                    }
                }
            }
        } else {
            // Use MI from previous iteration to define ossification
            // Promote previous iteration's ossified nodes to matured (bone)
            // Then build new ossification from the MI we just computed

            int n_matured_new = 0;
            for (int i = 0; i < nb_sca; ++i) {
                if (prev_phi_oss_data[i] >= 0.0 && !matured[i]) {
                    matured[i] = true;
                    ++n_matured_new;
                }
            }
            if (iter > 1)
                std::cout << "  Matured (ossified→bone): " << n_matured_new << " nodes\n";

            // Start with base material
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    if (P.y < g_cfg.interface_y || matured[iglo]) {
                        tmu_data[iglo] = 2.0 * mu_bone;
                        lam_data[iglo] = lambda_bone;
                        mat_data[iglo] = 0.0; // bone
                    } else {
                        tmu_data[iglo] = 2.0 * mu_cart;
                        lam_data[iglo] = lambda_cart;
                        mat_data[iglo] = 1.0; // cartilage
                    }
                }
            }

            // Build new ossification level set from previous iteration's MI
            // with spline boundary inhibition
            std::vector<double> inhibition_data(nb_sca, 1.0);
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    R2 P = FK.Pt(j);
                    double dist = std::abs(signed_distance_polygon(P, g_polygon));
                    double t = std::clamp(dist / g_cfg.oss_spline_band, 0.0, 1.0);
                    double inhibition = t * t * (3.0 - 2.0 * t);
                    inhibition_data[iglo] = inhibition;
                    // Exclude already-matured nodes from new ossification
                    if (matured[iglo] || P.y < g_cfg.interface_y)
                        phi_oss_data[iglo] = -1.0; // not a candidate
                    else
                        phi_oss_data[iglo] = prev_mi_avg[iglo] * inhibition - prev_threshold;
                }
            }

            // Override material for newly ossified region (intermediate props)
            int n_ossified = 0;
            for (int k = 0; k < Kh.nt; ++k) {
                const auto &FK = Sh[k];
                for (int j = 0; j < FK.NbDoF(); ++j) {
                    int iglo = Sh(k, j);
                    if (iglo < 0 || iglo >= nb_sca) continue;
                    if (phi_oss_data[iglo] >= 0.0 && FK.Pt(j).y >= g_cfg.interface_y && !matured[iglo]) {
                        tmu_data[iglo] = 2.0 * mu_oss;
                        lam_data[iglo] = lambda_oss;
                        mat_data[iglo] = 2.0; // ossified (intermediate)
                        ++n_ossified;
                    }
                }
            }
            std::cout << "  Ossified nodes (intermediate): " << n_ossified << " / " << nb_sca << "\n";
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
        std::span<double> mat_span(mat_data);
        fct_t mat_fh(Sh, mat_span);

        // phi_soc FE function for this iteration
        std::span<double> phi_oss_span(phi_oss_data);
        fct_t phi_soc_fh(Sh, phi_oss_span);

        // Find threshold for next iteration
        double threshold = find_threshold(result.sf_avg.miner, Sh, Kh);

        // Export VTK
        export_iteration(iter, result, uh_avg, hd_fh, oct_fh, mi_fh, mat_fh,
                         phi_soc_fh, phi_oss_data);

        // Store MI and threshold for next iteration
        prev_phi_oss_data = phi_oss_data;
        prev_mi_avg = result.sf_avg.miner;
        if (iter == 0)
            prev_threshold = threshold;

        std::cout << "  Iteration " << iter << " done.\n";
    }

    std::cout << "\nDone. Output in " << g_cfg.output_dir << "/\n";
    return 0;
}
