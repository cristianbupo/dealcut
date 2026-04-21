// step-90-revolution.cc — 3-D linear elasticity on the revolved SOC domain.
//
// Background structured-hex mesh; bottom face at y = 0 (clamped bone base /
// the circular disk). Outer SOC boundary: revolution of the 9-point rational
// B-spline from growth_cutfem.cpp around the y-axis. Bone-cartilage interface:
// horizontal plane at y = interfaceY. Load: parabolic ring-traction centred at
// the dome apex, applied on the cut outer surface.
//
// CutFEM: deal.II NonMatching (hp-FECollection + MeshClassifier) for the outer
// boundary. The bone-cartilage material interface is handled by cell-centroid
// material assignment (no extra level-set cutting there — first-pass accuracy).
// Ghost-penalty stabilisation is omitted; add it if small cut cells cause
// ill-conditioning.

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gmsh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace Step90Revolution
{
  using namespace dealii;

  // ============================================================
  // Geometry / material constants (same values as growth_cutfem.cpp)
  // ============================================================
  constexpr double s          = 1.0;
  constexpr double p1_geom    = s * 0.9;
  constexpr double p2_geom    = s * 0.2;
  constexpr double interfaceY = s * 1.0;  // bone-cartilage interface height

  constexpr double E_bone  = 500.0, nu_bone = 0.20;
  constexpr double E_cart  =   6.0, nu_cart = 0.47;

  constexpr double load_p_peak      = 1.0;
  constexpr double load_apex_frac   = 0.20; // fraction of half-arc that receives load

  // ============================================================
  // Revolution profile in the (r, y) half-plane
  // ============================================================
  struct P2 { double r, y; };

  static std::vector<P2>     g_profile;         // spline right-half, open polyline
  static std::vector<P2>     g_closed_polygon;  // spline + bottom + axis (closed)
  static std::vector<double> g_arc_len;         // cumulative arc lengths

  // --- 2-D geometry helpers ---

  static double dist_seg_sq(const P2 &p, const P2 &a, const P2 &b, double &t_out)
  {
    const double dr = b.r - a.r, dy = b.y - a.y;
    const double len2 = dr * dr + dy * dy;
    if (len2 < 1e-30)
      {
        t_out = 0.0;
        return (p.r - a.r) * (p.r - a.r) + (p.y - a.y) * (p.y - a.y);
      }
    t_out              = std::clamp(((p.r - a.r) * dr + (p.y - a.y) * dy) / len2,
                       0.0,
                       1.0);
    const double qr = a.r + t_out * dr - p.r;
    const double qy = a.y + t_out * dy - p.y;
    return qr * qr + qy * qy;
  }

  static bool point_in_polygon(const P2 &p, const std::vector<P2> &poly)
  {
    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
      if (((poly[i].y > p.y) != (poly[j].y > p.y)) &&
          (p.r < (poly[j].r - poly[i].r) * (p.y - poly[i].y) /
                     (poly[j].y - poly[i].y + 1e-30) +
                   poly[i].r))
        inside = !inside;
    return inside;
  }

  // Signed distance from (r, y) to the spline polyline; sign from PIP on
  // g_closed_polygon.  Magnitude uses the open spline polyline only — the flat
  // bottom (y = 0) is the mesh boundary, not a level-set feature.
  static double signed_distance_to_spline(double r, double y)
  {
    if (g_profile.size() < 2)
      return 1.0;
    const P2 q{r, y};
    double   min_d2 = std::numeric_limits<double>::max();
    double   dummy;
    for (size_t i = 0; i + 1 < g_profile.size(); ++i)
      min_d2 = std::min(min_d2,
                        dist_seg_sq(q, g_profile[i], g_profile[i + 1], dummy));
    const double d   = std::sqrt(min_d2);
    const P2     qe{std::max(r, 1e-9), y};
    return point_in_polygon(qe, g_closed_polygon) ? -d : +d;
  }

  // Outward unit normal in the (r, y) plane at profile segment seg.
  // Convention: CW rotation of the tangent (profile oriented base → apex).
  static std::array<double, 2> outward_normal_2d(size_t seg)
  {
    double dr = g_profile[seg + 1].r - g_profile[seg].r;
    double dy = g_profile[seg + 1].y - g_profile[seg].y;
    double dn = std::sqrt(dr * dr + dy * dy);
    if (dn < 1e-14)
      return {{0.0, 1.0}};
    return {{dy / dn, -dr / dn}}; // (n_r, n_y)
  }

  struct ProjResult
  {
    size_t seg;
    double alpha, arc_s;
  };

  static ProjResult project_to_profile(double r, double y)
  {
    ProjResult best{0, 0.0, 0.0};
    double     best_d2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i + 1 < g_profile.size(); ++i)
      {
        double t;
        double d2 = dist_seg_sq({r, y}, g_profile[i], g_profile[i + 1], t);
        if (d2 < best_d2)
          {
            best_d2 = d2;
            double seg_len =
              std::sqrt((g_profile[i + 1].r - g_profile[i].r) *
                          (g_profile[i + 1].r - g_profile[i].r) +
                        (g_profile[i + 1].y - g_profile[i].y) *
                          (g_profile[i + 1].y - g_profile[i].y));
            best = {i, t, g_arc_len[i] + t * seg_len};
          }
      }
    return best;
  }

  // ============================================================
  // Build the revolution profile via gmsh (same B-spline as growth_cutfem.cpp).
  // ============================================================
  static void build_revolution_profile()
  {
    gmsh::initialize();
    gmsh::model::add("revolution_profile");

    gmsh::model::occ::addPoint(s * 0.5, 0.0,               0, 0.1, 1);
    gmsh::model::occ::addPoint(s * 0.5, s * 1.0,           0, 0.1, 2);
    gmsh::model::occ::addPoint(s * 1.0, p1_geom,           0, 0.1, 3);
    gmsh::model::occ::addPoint(s * 1.0, s * 2.4,           0, 0.1, 4);
    gmsh::model::occ::addPoint(0.0,     s * 2.4 + p2_geom, 0, 0.1, 5);
    gmsh::model::occ::addPoint(-s * 1.0, s * 2.4,          0, 0.1, 6);
    gmsh::model::occ::addPoint(-s * 1.0, p1_geom,          0, 0.1, 7);
    gmsh::model::occ::addPoint(-s * 0.5, s * 1.0,          0, 0.1, 8);
    gmsh::model::occ::addPoint(-s * 0.5, 0.0,              0, 0.1, 9);
    gmsh::model::occ::addBSpline({1, 2, 3, 4, 5, 6, 7, 8, 9}, 2, 3,
                                 {1, 2, 1, 1, 1, 1, 1, 2, 1});
    gmsh::model::occ::synchronize();
    gmsh::model::mesh::setTransfiniteCurve(2, 201);
    gmsh::model::mesh::generate(1);

    std::vector<std::size_t> tags;
    std::vector<double>      coords, params;
    gmsh::model::mesh::getNodes(tags, coords, params, 1, 2, true, true);

    struct N1D
    {
      std::size_t t;
      double      u, x, y;
    };
    std::vector<N1D> nodes;
    nodes.reserve(tags.size());
    for (std::size_t i = 0; i < tags.size(); ++i)
      nodes.push_back({tags[i], params[i], coords[3 * i], coords[3 * i + 1]});
    std::sort(nodes.begin(), nodes.end(),
              [](const N1D &a, const N1D &b) { return a.u < b.u; });

    std::vector<std::array<double, 2>> full;
    full.reserve(nodes.size());
    std::unordered_set<std::size_t> seen;
    for (const auto &n : nodes)
      if (seen.insert(n.t).second)
        full.push_back({{n.x, n.y}});

    {
      double df = std::hypot(full.front()[0] - s * 0.5, full.front()[1]);
      double db = std::hypot(full.back()[0] - s * 0.5, full.back()[1]);
      if (df > db)
        std::reverse(full.begin(), full.end());
    }

    // Slice to x >= 0 (right half), interpolating the crossing to x = 0.
    std::vector<P2> right_half;
    right_half.reserve(full.size());
    right_half.push_back({full[0][0], full[0][1]});
    for (std::size_t i = 1; i < full.size(); ++i)
      {
        if (full[i][0] >= 0.0)
          {
            right_half.push_back({full[i][0], full[i][1]});
          }
        else
          {
            double x0 = full[i - 1][0], y0 = full[i - 1][1];
            double x1 = full[i][0], y1 = full[i][1];
            double t  = x0 / (x0 - x1);
            right_half.push_back({0.0, y0 + t * (y1 - y0)});
            break;
          }
      }

    g_profile = right_half;

    // Cumulative arc lengths along the spline polyline.
    g_arc_len.assign(g_profile.size(), 0.0);
    for (size_t i = 1; i < g_profile.size(); ++i)
      {
        double dr = g_profile[i].r - g_profile[i - 1].r;
        double dy = g_profile[i].y - g_profile[i - 1].y;
        g_arc_len[i] = g_arc_len[i - 1] + std::sqrt(dr * dr + dy * dy);
      }

    // Closed polygon: (0, 0) → spline → apex, closed via axis return.
    g_closed_polygon.clear();
    g_closed_polygon.push_back({0.0, 0.0});
    for (const auto &p : right_half)
      g_closed_polygon.push_back(p);

    std::cout << "Profile: " << g_profile.size() << " nodes, "
              << "arc length = " << g_arc_len.back() << "\n";
    gmsh::finalize();
  }

  // ============================================================
  // Level set functions
  // ============================================================

  // Outer SOC boundary: revolution of the B-spline about the y-axis.
  // phi < 0 inside the SOC, phi > 0 outside.
  class RevolutionSurface : public Function<3>
  {
  public:
    RevolutionSurface()
      : Function<3>()
    {}
    double value(const Point<3> &p, unsigned int = 0) const override
    {
      return signed_distance_to_spline(std::hypot(p[0], p[2]), p[1]);
    }
  };

  // Bone-cartilage interface: same horizontal plane as in growth_cutfem.cpp.
  // phi < 0 in bone (y < interfaceY), phi > 0 in cartilage (y > interfaceY).
  class InterfaceLevelSet : public Function<3>
  {
  public:
    InterfaceLevelSet()
      : Function<3>()
    {}
    double value(const Point<3> &p, unsigned int = 0) const override
    {
      return p[1] - interfaceY;
    }
  };

  // ============================================================
  // Lamé parameters: cell-centroid material assignment.
  // ============================================================
  static std::pair<double, double> lame_at(const Point<3> &p)
  {
    const bool   cart = (p[1] >= interfaceY);
    const double E    = cart ? E_cart : E_bone;
    const double nu   = cart ? nu_cart : nu_bone;
    return {E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), // lambda
            E / (2.0 * (1.0 + nu))};                   // mu
  }

  // ============================================================
  // Traction function (3-component Neumann BC on the cut surface).
  // Revolution of the parabolic load from growth_cutfem.cpp.
  // Load is centred at the apex (end of g_profile) and falls off along the
  // arc toward the base.  T = -p * n_outward (inward compression).
  // ============================================================
  class TractionFunction : public Function<3>
  {
  public:
    TractionFunction()
      : Function<3>(3)
    {}

    double value(const Point<3> &p, unsigned int comp = 0) const override
    {
      const double r    = std::hypot(p[0], p[2]);
      const auto   proj = project_to_profile(r, p[1]);

      const double s_tot   = g_arc_len.back();
      const double d_apex  = s_tot - proj.arc_s; // arc distance from apex
      const double d_limit = load_apex_frac * s_tot;
      if (d_apex >= d_limit)
        return 0.0;
      const double tq   = d_apex / d_limit;
      const double pval = load_p_peak * (1.0 - tq * tq);

      const auto [n_r, n_y] = outward_normal_2d(proj.seg);
      const double nx = (r > 1e-9) ? n_r * p[0] / r : 0.0;
      const double nz = (r > 1e-9) ? n_r * p[2] / r : 0.0;

      if (comp == 0) return -pval * nx;
      if (comp == 1) return -pval * n_y;
      return -pval * nz;
    }

    void vector_value(const Point<3> &p, Vector<double> &v) const override
    {
      for (unsigned int c = 0; c < 3; ++c)
        v(c) = value(p, c);
    }
  };

  // ============================================================
  // hp active FE index
  // ============================================================
  enum class ActiveFEIndex : types::fe_index
  {
    active  = 0, // FESystem(FE_Q(1), 3)
    nothing = 1  // FE_Nothing(3)
  };

  // ============================================================
  // Elasticity solver
  // ============================================================
  class ElasticitySolver
  {
  public:
    void run();

  private:
    void make_grid();
    void setup_level_set();
    void distribute_dofs();
    void initialize_matrix();
    void assemble_system();
    void solve();
    void output_results();

    Triangulation<3> triangulation;

    FE_Q<3>      ls_fe{1};
    DoFHandler<3> ls_dof_handler{triangulation};
    Vector<double> ls_vector;

    hp::FECollection<3>       fe_collection;
    DoFHandler<3>             dof_handler{triangulation};
    AffineConstraints<double> constraints;
    SparsityPattern           sparsity;
    SparseMatrix<double>      system_matrix;
    Vector<double>            system_rhs;
    Vector<double>            solution;

    NonMatching::MeshClassifier<3> mesh_classifier{ls_dof_handler, ls_vector};
  };

  void ElasticitySolver::make_grid()
  {
    // Non-round extents reduce the chance that the level set is exactly 0 at
    // mesh vertices (which can confuse the MeshClassifier).
    const Point<3>                  p0(-1.231, 0.0, -1.231);
    const Point<3>                  p1(1.231, 2.833, 1.231);
    const std::vector<unsigned int> reps{32, 36, 32};
    GridGenerator::subdivided_hyper_rectangle(triangulation, reps, p0, p1);

    // Mark the bottom face (y = 0) as boundary_id = 1 for Dirichlet BC.
    // All other boundary faces keep id = 0 (natural Neumann / traction-free).
    for (auto &cell : triangulation.active_cell_iterators())
      for (auto &face : cell->face_iterators())
        if (face->at_boundary() && face->center()[1] < 1e-6)
          face->set_boundary_id(1);

    std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
  }

  void ElasticitySolver::setup_level_set()
  {
    ls_dof_handler.distribute_dofs(ls_fe);
    ls_vector.reinit(ls_dof_handler.n_dofs());
    VectorTools::interpolate(ls_dof_handler, RevolutionSurface(), ls_vector);
    mesh_classifier.reclassify();
  }

  void ElasticitySolver::distribute_dofs()
  {
    fe_collection.push_back(FESystem<3>(FE_Q<3>(1), 3));      // index 0 = active
    fe_collection.push_back(FESystem<3>(FE_Nothing<3>(), 3)); // index 1 = dead

    for (const auto &cell : dof_handler.active_cell_iterators())
      cell->set_active_fe_index(
        mesh_classifier.location_to_level_set(cell) ==
            NonMatching::LocationToLevelSet::outside
          ? static_cast<types::fe_index>(ActiveFEIndex::nothing)
          : static_cast<types::fe_index>(ActiveFEIndex::active));

    dof_handler.distribute_dofs(fe_collection);
    std::cout << "Active DoFs: " << dof_handler.n_dofs() << "\n";

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // Clamp all displacement components at y = 0 (boundary_id == 1).
    VectorTools::interpolate_boundary_values(
      dof_handler, 1, Functions::ZeroFunction<3>(3), constraints);
    constraints.close();
  }

  void ElasticitySolver::initialize_matrix()
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity.copy_from(dsp);
    system_matrix.reinit(sparsity);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
  }

  void ElasticitySolver::assemble_system()
  {
    const QGauss<1> q1d(2);

    NonMatching::RegionUpdateFlags rflag;
    rflag.inside = update_values | update_gradients | update_JxW_values |
                   update_quadrature_points;
    rflag.surface = update_values | update_JxW_values |
                    update_quadrature_points | update_normal_vectors;

    NonMatching::FEValues<3> nm_fev(fe_collection,
                                    q1d,
                                    rflag,
                                    mesh_classifier,
                                    ls_dof_handler,
                                    ls_vector);

    const FEValuesExtractors::Vector u_ext(0);
    const TractionFunction           traction;

    FullMatrix<double>                   local_mat;
    Vector<double>                       local_rhs;
    std::vector<types::global_dof_index> dof_idx;

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(
             static_cast<types::fe_index>(ActiveFEIndex::active)))
      {
        const unsigned int n = cell->get_fe().n_dofs_per_cell();
        local_mat.reinit(n, n);
        local_rhs.reinit(n);
        local_mat = 0;
        local_rhs = 0;

        nm_fev.reinit(cell);

        // --- Volume stiffness (inside region of cut cell) ---
        const auto &opt_in = nm_fev.get_inside_fe_values();
        if (opt_in)
          {
            const FEValues<3> &fv = *opt_in;
            for (unsigned int q = 0; q < fv.n_quadrature_points; ++q)
              {
                const auto [lam, mu] = lame_at(fv.quadrature_point(q));
                for (unsigned int i = 0; i < n; ++i)
                  {
                    const SymmetricTensor<2, 3> eps_i =
                      symmetrize(fv[u_ext].gradient(i, q));
                    const double div_i = trace(eps_i);
                    for (unsigned int j = 0; j < n; ++j)
                      {
                        const SymmetricTensor<2, 3> eps_j =
                          symmetrize(fv[u_ext].gradient(j, q));
                        local_mat(i, j) +=
                          (lam * div_i * trace(eps_j) +
                           2.0 * mu * (eps_i * eps_j)) *
                          fv.JxW(q);
                      }
                  }
              }
          }

        // --- Traction on cut surface (Neumann) ---
        const auto &opt_sf = nm_fev.get_surface_fe_values();
        if (opt_sf)
          {
            const NonMatching::FEImmersedSurfaceValues<3> &sf = *opt_sf;
            for (unsigned int q = 0; q < sf.n_quadrature_points; ++q)
              {
                const Point<3> &pt = sf.quadrature_point(q);
                Vector<double>  T_val(3);
                traction.vector_value(pt, T_val);
                Tensor<1, 3> T;
                T[0] = T_val(0); T[1] = T_val(1); T[2] = T_val(2);
                for (unsigned int i = 0; i < n; ++i)
                  local_rhs(i) += sf[u_ext].value(i, q) * T * sf.JxW(q);
              }
          }

        dof_idx.resize(n);
        cell->get_dof_indices(dof_idx);
        constraints.distribute_local_to_global(local_mat,
                                               local_rhs,
                                               dof_idx,
                                               system_matrix,
                                               system_rhs);
      }
  }

  void ElasticitySolver::solve()
  {
    SolverControl                     sc(5000, 1e-9 * system_rhs.l2_norm());
    SolverCG<Vector<double>>          cg(sc);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(system_matrix, 1.2);
    cg.solve(system_matrix, solution, system_rhs, prec);
    constraints.distribute(solution);
    std::cout << "CG iterations: " << sc.last_step() << "\n";
  }

  void ElasticitySolver::output_results()
  {
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    const std::vector<std::string> disp_names{"displacement_x",
                                              "displacement_y",
                                              "displacement_z"};
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(3, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(solution, disp_names, DataOut<3>::type_dof_data, interp);
    data_out.add_data_vector(ls_dof_handler, ls_vector, "phi_outer");

    // Only emit cells inside or intersected by the SOC boundary.
    data_out.set_cell_selection([this](const Triangulation<3>::cell_iterator &c) {
      return c->is_active() &&
             mesh_classifier.location_to_level_set(c) !=
               NonMatching::LocationToLevelSet::outside;
    });

    data_out.build_patches();
    std::ofstream out("solution.vtu");
    data_out.write_vtu(out);
    std::cout << "Wrote solution.vtu\n";
  }

  void ElasticitySolver::run()
  {
    make_grid();
    setup_level_set();
    distribute_dofs();
    initialize_matrix();

    Timer timer;
    assemble_system();
    std::cout << "Assembly: " << timer.wall_time() << " s\n";

    timer.restart();
    solve();
    std::cout << "Solve:    " << timer.wall_time() << " s\n";

    output_results();
  }

} // namespace Step90Revolution

int main()
{
  try
    {
      Step90Revolution::build_revolution_profile();
      Step90Revolution::ElasticitySolver solver;
      solver.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << "\n";
      return 1;
    }
  return 0;
}
