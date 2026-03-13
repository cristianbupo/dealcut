// growth.cc
//
// SOC geometry kept (gmsh curve nodes).
// Growth/ossification pattern:
//  - 5-step moving Neumann window (CONTINUOUS, parametric-spline based)
//  - per-step outputs:  output_growth/growth-step-i.vtu
//  - averaged output:   output_growth/growth.vtu
// Also improves convergence:
//  - tighter CG tolerance
//  - SSOR preconditioner for stiffness matrix
//  - scalar mass matrix assembled once
//
// Writes raw then patches TRUE VTK CellData via VTK rewrite.
//
// -----------------------------------------
// NEW (your request):
// -----------------------------------------
// Parabolic load defined by 3 continuous parameters:
//   1) center of parabola:      u_center  (gmsh curve param coordinate)
//   2) half-span in u-space:    u_radius  (support radius in the spline parameter)
//   3) vertex altitude:         p_peak    (peak pressure magnitude)
//
// Pressure law:
//   p(u) = p_peak * max(0, 1 - ((u - u_center)/u_radius)^2 )
// Traction:
//   t = -p(u) * n  (n is the unit normal from the spline segment)
//
// No dependency on integer node indices (k_min/k_max/i0/i1) anymore.
// The loading process is kept continuous in u, assembled with a
// local active window, local normalized coordinate,
// and direct -p*n traction on the projected top segment.

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <gmsh.h>

// VTK (TRUE CellData rewrite)
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------
// VTK helpers
// ---------------------------
static vtkDataArray *
vtk_find_array_with_suffixes(vtkFieldData *fd, const std::string &base)
{
  if (!fd)
    return nullptr;
  if (auto *a = fd->GetArray(base.c_str()))
    return a;
  if (auto *a = fd->GetArray((base + "_0").c_str()))
    return a;
  if (auto *a = fd->GetArray((base + "_1").c_str()))
    return a;
  return nullptr;
}

struct FoundArray
{
  vtkDataArray *arr = nullptr;
  enum class Where
  {
    Cell,
    Point
  } where = Where::Point;
};

static FoundArray
vtk_find_cell_or_point_array(vtkUnstructuredGrid *ug, const std::string &name)
{
  FoundArray out;
  if (!ug)
    return out;

  vtkCellData  *cd = ug->GetCellData();
  vtkPointData *pd = ug->GetPointData();

  if (cd)
    if (auto *a = vtk_find_array_with_suffixes(cd, name))
      return {a, FoundArray::Where::Cell};

  if (pd)
    if (auto *a = vtk_find_array_with_suffixes(pd, name))
      return {a, FoundArray::Where::Point};

  return out;
}

static void
vtk_add_true_celldata_from_cell_id(vtkUnstructuredGrid       *ug,
                                   const FoundArray          &cell_id,
                                   const std::vector<double> &values_per_dealii_cell,
                                   const std::string         &out_name)
{
  vtkCellData *cd = ug->GetCellData();
  if (!cd)
    throw std::runtime_error("VTK: missing CellData.");

  const vtkIdType n_cells = ug->GetNumberOfCells();

  if (cd->HasArray(out_name.c_str()))
    cd->RemoveArray(out_name.c_str());

  auto arr = vtkSmartPointer<vtkFloatArray>::New();
  arr->SetName(out_name.c_str());
  arr->SetNumberOfComponents(1);
  arr->SetNumberOfTuples(n_cells);

  auto lookup = [&](long long cid) -> float {
    if (cid >= 0 && static_cast<std::size_t>(cid) < values_per_dealii_cell.size())
      return static_cast<float>(values_per_dealii_cell[static_cast<std::size_t>(cid)]);
    return 0.0f;
  };

  if (cell_id.where == FoundArray::Where::Cell)
  {
    for (vtkIdType c = 0; c < n_cells; ++c)
    {
      const long long cid = static_cast<long long>(std::llround(cell_id.arr->GetTuple1(c)));
      arr->SetValue(c, lookup(cid));
    }
  }
  else
  {
    for (vtkIdType c = 0; c < n_cells; ++c)
    {
      vtkCell   *cell   = ug->GetCell(c);
      vtkIdList *pt_ids = cell->GetPointIds();

      std::unordered_map<long long, int> counts;
      counts.reserve(8);

      for (vtkIdType k = 0; k < pt_ids->GetNumberOfIds(); ++k)
      {
        const vtkIdType pid = pt_ids->GetId(k);
        const long long cid = static_cast<long long>(std::llround(cell_id.arr->GetTuple1(pid)));
        counts[cid] += 1;
      }

      long long best_cid = -1;
      int best_count = -1;
      for (const auto &kv : counts)
        if (kv.second > best_count)
        {
          best_cid = kv.first;
          best_count = kv.second;
        }

      arr->SetValue(c, lookup(best_cid));
    }
  }

  cd->AddArray(arr);
}

static void
vtk_postprocess_add_true_celldata(
  const std::string &in_vtu,
  const std::string &out_vtu,
  const std::string &cell_id_name,
  const std::vector<std::pair<std::string, std::vector<double>>> &arrays_to_add)
{
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(in_vtu.c_str());
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();
  if (!ug)
    throw std::runtime_error("VTK: reader output is null.");

  auto cell_id = vtk_find_cell_or_point_array(ug, cell_id_name);
  if (!cell_id.arr)
    throw std::runtime_error("VTK: array '" + cell_id_name + "' not found.");

  for (const auto &kv : arrays_to_add)
    vtk_add_true_celldata_from_cell_id(ug, cell_id, kv.second, kv.first);

  auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetFileName(out_vtu.c_str());
  writer->SetInputData(ug);
  writer->SetDataModeToBinary();
  if (!writer->Write())
    throw std::runtime_error("VTK: writer failed for " + out_vtu);
}

namespace SOC
{
  using namespace dealii;

  struct LameParameters
  {
    double lambda;
    double mu;
  };

  static LameParameters lame_from_E_nu(const double E, const double nu)
  {
    LameParameters lame;
    lame.mu     = E / (2.0 * (1.0 + nu));
    lame.lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    return lame;
  }

  static inline void
  compute_invariants_plane_strain(const SymmetricTensor<2, 2> &eps,
                                  const double                lambda,
                                  const double                mu,
                                  double                     &von_mises,
                                  double                     &hydro,
                                  double                     &oct_shear,
                                  double                     &mi,
                                  const double                kMi = 0.5)
  {
    const SymmetricTensor<2, 2> sigma_2d =
      2.0 * mu * eps + lambda * trace(eps) * unit_symmetric_tensor<2>();

    const double sigma_zz = lambda * trace(eps);

    const double sigma_xx = sigma_2d[0][0];
    const double sigma_yy = sigma_2d[1][1];
    const double sigma_xy = sigma_2d[0][1];

    hydro = (sigma_xx + sigma_yy + sigma_zz) / 3.0;

    const double s_xx = sigma_xx - hydro;
    const double s_yy = sigma_yy - hydro;
    const double s_zz = sigma_zz - hydro;
    const double s_xy = sigma_xy;

    const double s_contract_s =
      s_xx * s_xx + s_yy * s_yy + s_zz * s_zz + 2.0 * s_xy * s_xy;

    von_mises = std::sqrt(1.5 * s_contract_s);
    oct_shear = std::sqrt(2.0 / 3.0) * von_mises;
    mi        = oct_shear + kMi * hydro;
  }

  static double clamp01(const double a)
  {
    return std::max(0.0, std::min(1.0, a));
  }

  static double dist_point_segment_sq(const Point<2> &p,
                                      const Point<2> &a,
                                      const Point<2> &b,
                                      double          &alpha_out)
  {
    const Tensor<1, 2> ab = b - a;
    const Tensor<1, 2> ap = p - a;

    const double ab2 = ab.norm_square();
    double       t   = 0.0;
    if (ab2 > 0.0)
      t = (ap * ab) / ab2;

    t         = clamp01(t);
    alpha_out = t;

    const Point<2> q = a + t * ab;
    return (p - q).norm_square();
  }

  static bool point_in_polygon(const Point<2> &p, const std::vector<Point<2>> &poly)
  {
    bool inside = false;
    const std::size_t n = poly.size();
    if (n < 3)
      return false;

    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
      {
        const double xi = poly[i][0], yi = poly[i][1];
        const double xj = poly[j][0], yj = poly[j][1];

        const bool intersect =
          ((yi > p[1]) != (yj > p[1])) &&
          (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-30) + xi);

        if (intersect)
          inside = !inside;
      }
    return inside;
  }

  static double signed_distance_closed_polygon(const Point<2>              &p,
                                               const std::vector<Point<2>> &poly)
  {
    const std::size_t n = poly.size();
    if (n < 2)
      return std::numeric_limits<double>::quiet_NaN();

    double best_d2 = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < n; ++i)
      {
        const std::size_t j = (i + 1) % n;
        double           a  = 0.0;
        const double d2 = dist_point_segment_sq(p, poly[i], poly[j], a);
        best_d2 = std::min(best_d2, d2);
      }

    const double d = std::sqrt(std::max(0.0, best_d2));
    const bool   inside = point_in_polygon(p, poly);
    return inside ? -d : +d;
  }

  class PolygonSignedDistance : public Function<2>
  {
  public:
    explicit PolygonSignedDistance(std::vector<Point<2>> poly_in)
      : Function<2>(1)
      , poly(std::move(poly_in))
    {}

    double value(const Point<2> &p, const unsigned int = 0) const override
    {
      return signed_distance_closed_polygon(p, poly);
    }

  private:
    std::vector<Point<2>> poly;
  };

  template <int dim>
  class PhiFunction : public Function<dim>
  {
  public:
    explicit PhiFunction(const double interfaceY)
      : Function<dim>(1)
      , interfaceY(interfaceY)
    {}

    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return p[1] - interfaceY;
    }

  private:
    const double interfaceY;
  };

  static Tensor<1, 2> unit_normal_from_top_spline(const std::vector<Point<2>> &top,
                                                  const unsigned int           seg)
  {
    Tensor<1, 2> n;
    if (top.size() < 2 || seg + 1 >= top.size())
      return n;

    Tensor<1, 2> t = top[seg + 1] - top[seg];
    const double tn = t.norm();
    if (tn < 1e-14)
      return n;

    t /= tn;
    n[0] = -t[1];
    n[1] =  t[0];

    if (n[1] < 0.0)
      n *= -1.0;

    return n;
  }

  struct SegmentProjection
  {
    double       dist2   = std::numeric_limits<double>::max();
    double       alpha   = 0.0;
    unsigned int seg     = 0;
    double       u_along = 0.0;
  };

  static SegmentProjection project_to_polyline_with_u(const Point<2>                &p,
                                                      const std::vector<Point<2>> &line,
                                                      const std::vector<double>   &u_nodes)
  {
    SegmentProjection best;
    if (line.size() < 2 || u_nodes.size() != line.size())
      return best;
    for (unsigned int k = 0; k + 1 < line.size(); ++k)
      {
        double alpha = 0.0;
        const double d2 = dist_point_segment_sq(p, line[k], line[k + 1], alpha);
        if (d2 < best.dist2)
          {
            best.dist2 = d2;
            best.seg   = k;
            best.alpha = alpha;
            best.u_along = u_nodes[k] + alpha * (u_nodes[k + 1] - u_nodes[k]);
          }
      }
    return best;
  }

  template <int dim>
  
class ImmersedElasticity
  {
  public:
    ImmersedElasticity();
    void run();

  private:
    void make_background_grid();
    void build_geometry_from_gmsh_curve_nodes();
    void setup_level_set();
    void classify_and_distribute_dofs();
    void setup_system();

    void assemble_stiffness(); // K only (volume + bottom Nitsche)

    // Continuous u-parametric load parameters
    void assemble_rhs_top(double u_center,
                          double u_radius,
                          double p_peak);

    void init_preconditioner_K();

    void assemble_scalar_mass_matrix();        // M only once
    void assemble_scalar_rhs_and_cell_avgs();  // RHS for invariants + per-cell avgs
    void solve_vector_system();
    void solve_scalar_systems();

    void build_phi_nodal();
    void fill_cell_outputs_static();

    void initialize_restart_fields();
    std::string tagged_prefix() const;
    std::string restart_filename(const unsigned int id) const;

    template <typename CellIterator>
    bool cell_is_bulk_top(const CellIterator &cell) const;

    std::vector<std::pair<double, double>>
    extract_axis_profile(const Vector<double> &field) const;

    bool find_axis_cuts_for_threshold(
      const std::vector<std::pair<double, double>> &profile,
      const double                                  threshold,
      double                                       &y_left,
      double                                       &y_right) const;

    double compute_bulk_top_area() const;
    double compute_ossified_area_for_threshold(const Vector<double> &field,
                                               const double          threshold) const;
    void compute_area_based_threshold_from_field(const Vector<double> &mi_source);
    void build_center_guided_oss_source_from_reference(const double y_focus_center);
    void build_ossification_from_reference_field(const Vector<double> &mi_source);
    void update_reference_cell_exports();

    void save_restart_bundle(const Vector<double>      &U_avg,
                             const Vector<double>      &vm_avg,
                             const Vector<double>      &os_avg,
                             const Vector<double>      &hd_avg,
                             const Vector<double>      &mi_avg,
                             const std::vector<double> &vm_cell_avg,
                             const std::vector<double> &os_cell_avg,
                             const std::vector<double> &hd_cell_avg,
                             const std::vector<double> &mi_cell_avg,
                             const std::vector<double> &p_cell_avg) const;

    void load_restart_bundle(Vector<double> &mi_loaded,
                             Vector<double> &mi_reference_loaded);

    void output_one(const std::string   &base,
                    const Vector<double> &U_in,
                    const Vector<double> &vm_in,
                    const Vector<double> &os_in,
                    const Vector<double> &hd_in,
                    const Vector<double> &mi_in,
                    const std::vector<double> &vm_cell_in,
                    const std::vector<double> &os_cell_in,
                    const std::vector<double> &hd_cell_in,
                    const std::vector<double> &mi_cell_in,
                    const std::vector<double> &pressure_cell_in) const;

    LameParameters choose_lame(const Point<dim> &p,
                               const double      mi_ref_value,
                               const bool        allow_ossification) const;

    struct StepDef
    {
      double u_center;
      double u_radius;
      double p_peak;
    };

    std::vector<StepDef> build_default_steps() const;

  private:
    enum class RunMode
    {
      export_mi  = 0,
      import_mi  = 1
    };

    // -----------------------------
    // user knobs for chained runs
    // -----------------------------
    const RunMode     run_mode            = RunMode::import_mi;
    const unsigned int iteration_id       = 1; // current output suffix: _0, _1, ...
    const unsigned int import_iteration_id = 0; // used only when run_mode=import_mi

    // GEOMETRY SCALE (you used 2.2)
    const double s = 1.0;

    // target ossified area inside the bulk cartilage region (tag 3)
    // expressed as fraction of the total bulk-cartilage area
    const double oss_cartilage_area_fraction = 0.02;

    // center-guided ossification selection:
    // restrict the thresholded region to a central window around the symmetry axis
    // and around the vertical location of the axis maximum of the reference field.
    const bool   use_center_guided_ossification = false;
    const double oss_focus_x_halfwidth = 0.45;
    const double oss_focus_y_halfwidth = 0.55;

    // degrees
    const unsigned int fe_degree  = 2;
    const unsigned int phi_degree = 1;

    // interface for outputs (scaled)
    const double interfaceY = s * 1.0;

    // material values
    const double youngBelow   = 500.0;
    const double poissonBelow = 0.2;
    const double youngAbove   = 6.0;
    const double poissonAbove = 0.47;

    // stage-1 ossification material = average properties
    const double youngMid     = 0.5 * (youngBelow + youngAbove);
    const double poissonMid   = 0.5 * (poissonBelow + poissonAbove);

    // bottom clamp (scaled in x)
    const double x_bottom_min =  s * -0.5;
    const double x_bottom_max =  s *  0.5;
    const double y_bottom     =  0.0;

    // SOC geometry params (scaled)
    const double p1_geom = s * 0.9;
    const double p2_geom = s * 0.2;

    // background bounds (scaled)
    const Point<dim> bg_p1 = Point<dim>( s * -1.21, 0.0);
    const Point<dim> bg_p2 = Point<dim>( s *  1.21, s * (2.41 + 0.5));

    // solver params (vector)
    const double       cg_rel_tol_vec = 1e-10;
    const unsigned int cg_max_iter_vec = 60000;
    const double       ssor_omega      = 1.2;

    // -----------------------------------------
    // Load tuning knobs (continuous in u)
    // -----------------------------------------
    // center specified as fraction of [u_min, u_max] so you can tune easily
    const double load_center_u_frac = 0.50; // 0 = start of spline, 1 = end

    // reference support radius in u-space (kept for compatibility / tuning)
    const double load_u_radius = s * 0.35;

    // vertex altitude (peak pressure magnitude)
    const double load_p_peak   = 1.0;       // tune: magnitude

    // restart / imported-mi derived state
    bool   enable_ossification_material = false;
    double mi_threshold         = std::numeric_limits<double>::quiet_NaN();
    double mi_axis_max          = std::numeric_limits<double>::quiet_NaN();
    double mi_axis_y_at_max     = std::numeric_limits<double>::quiet_NaN();
    double mi_axis_cut_y0       = std::numeric_limits<double>::quiet_NaN();
    double mi_axis_cut_y1       = std::numeric_limits<double>::quiet_NaN();
    double mi_axis_total_length = std::numeric_limits<double>::quiet_NaN();
    double cartilage_area_total = std::numeric_limits<double>::quiet_NaN();
    double cartilage_area_target = std::numeric_limits<double>::quiet_NaN();
    double cartilage_area_achieved = std::numeric_limits<double>::quiet_NaN();

    // Triangulation + level set
    Triangulation<dim> triangulation;

    FE_Q<dim>        fe_level_set;
    DoFHandler<dim>  level_set_dh;
    Vector<double>   level_set;

    // hp FE collections: vector and scalar
    hp::FECollection<dim> fe_collection_vec;
    hp::FECollection<dim> fe_collection_sca;

    DoFHandler<dim> dof_handler_vec;
    DoFHandler<dim> dof_handler_sca;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    // vector system
    AffineConstraints<double> constraints_vec;
    SparsityPattern           sparsity_vec;
    SparseMatrix<double>      K;
    Vector<double>            U;
    Vector<double>            rhs;

    // preconditioner for K
    mutable PreconditionSSOR<SparseMatrix<double>> precond_K;
    bool precond_K_ready = false;

    // scalar projection system (mass)
    AffineConstraints<double> constraints_sca;
    SparsityPattern           sparsity_sca;
    SparseMatrix<double>      M;

    Vector<double> von_mises, octShearS, hydroD, mi;
    Vector<double> rhs_vm, rhs_os, rhs_hd, rhs_mi;

    mutable PreconditionJacobi<SparseMatrix<double>> precond_M;
    bool precond_M_ready = false;

    // nodal scalar fields
    Vector<double> phi_nodal;
    Vector<double> mi_reference;
    Vector<double> mi_oss_source;
    Vector<double> phi_oss_nodal;

    // static cell data (deal.II)
    Vector<float> cell_id_dealii;
    Vector<float> material_dealii;
    Vector<float> phi_cell_dealii;
    Vector<float> phi_oss_cell_dealii;
    Vector<float> mi_reference_cell_dealii;
    Vector<float> mi_oss_source_cell_dealii;
    Vector<float> oss_material_cell_dealii;

    // per-step cell pressure (deal.II)
    Vector<float> cell_pressure_dealii;

    // per-step deal.II cell invariants
    Vector<float> vm_cell_dealii, os_cell_dealii, hd_cell_dealii, mi_cell_dealii;

    // per-step TRUE values per deal.II active cell (for VTK rewrite)
    std::vector<double> vm_cell, os_cell, hd_cell, mi_cell;
    std::vector<double> pressure_cell;

    // geometry storage
    std::vector<Point<2>> polygon;

    // sampled spline nodes + gmsh param coords (same size)
    std::vector<Point<2>> top_spline;
    std::vector<double>   top_u;

    // output prefix
    const std::string prefix = "growth";
  };

  enum ActiveFEIndex
  {
    lagrange = 0,
    nothing  = 1
  };

  template <int dim>
  ImmersedElasticity<dim>::ImmersedElasticity()
    : fe_level_set(phi_degree)
    , level_set_dh(triangulation)
    , dof_handler_vec(triangulation)
    , dof_handler_sca(triangulation)
    , mesh_classifier(level_set_dh, level_set)
  {}


  template <int dim>
  void ImmersedElasticity<dim>::make_background_grid()
  {
    std::vector<unsigned int> nsub(dim);
    nsub[0] = 60;
    nsub[1] = 60;

    GridGenerator::subdivided_hyper_rectangle(triangulation, nsub, bg_p1, bg_p2, false);

    const unsigned int n_cells = triangulation.n_active_cells();

    cell_id_dealii.reinit(n_cells);
      material_dealii.reinit(n_cells);
      phi_cell_dealii.reinit(n_cells);
      phi_oss_cell_dealii.reinit(n_cells);
      mi_reference_cell_dealii.reinit(n_cells);
      mi_oss_source_cell_dealii.reinit(n_cells);
      oss_material_cell_dealii.reinit(n_cells);
    cell_pressure_dealii.reinit(n_cells);

    vm_cell_dealii.reinit(n_cells);
    os_cell_dealii.reinit(n_cells);
    hd_cell_dealii.reinit(n_cells);
    mi_cell_dealii.reinit(n_cells);

    vm_cell.assign(n_cells, 0.0);
    os_cell.assign(n_cells, 0.0);
    hd_cell.assign(n_cells, 0.0);
    mi_cell.assign(n_cells, 0.0);
    pressure_cell.assign(n_cells, 0.0);
  }

  template <int dim>
  void ImmersedElasticity<dim>::build_geometry_from_gmsh_curve_nodes()
  {
    struct GmshSession
    {
      GmshSession() { gmsh::initialize(); }
      ~GmshSession()
      {
        try { gmsh::finalize(); } catch (...) {}
      }
    } session;

    gmsh::model::add("growth");

    // Geometry scaled by s (2.2)
    gmsh::model::occ::addPoint( s*0.5,  s*0.0, 0.0, 0.1, 1);
    gmsh::model::occ::addPoint( s*0.5,  s*1.0, 0.0, 0.1, 2);
    gmsh::model::occ::addPoint( s*1.0,  p1_geom, 0.0, 0.1, 3);
    gmsh::model::occ::addPoint( s*1.0,  s*2.4, 0.0, 0.1, 4);
    gmsh::model::occ::addPoint( s*0.0,  s*2.4 + p2_geom, 0.0, 0.1, 5);
    gmsh::model::occ::addPoint(-s*1.0,  s*2.4, 0.0, 0.1, 6);
    gmsh::model::occ::addPoint(-s*1.0,  p1_geom, 0.0, 0.1, 7);
    gmsh::model::occ::addPoint(-s*0.5,  s*1.0, 0.0, 0.1, 8);
    gmsh::model::occ::addPoint(-s*0.5,  s*0.0, 0.0, 0.1, 9);

    gmsh::model::occ::addLine(9, 1, 1);
    gmsh::model::occ::addBSpline({1,2,3,4,5,6,7,8,9},
                                 2,
                                 3,
                                 {1.0,2.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0});
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::setTransfiniteCurve(2, 101);
    gmsh::model::mesh::generate(1);

    std::vector<std::size_t> node_tags;
    std::vector<double>      coords;
    std::vector<double>      param_coords;

    gmsh::model::mesh::getNodes(node_tags,
                                coords,
                                param_coords,
                                /*dim=*/1,
                                /*tag=*/2,
                                /*includeBoundary=*/true,
                                /*returnParametricCoord=*/true);

    if (node_tags.empty() || coords.size() != 3 * node_tags.size())
      throw std::runtime_error("Gmsh: getNodes on curve 2 failed.");

    struct Node1D
    {
      std::size_t tag;
      double      u;
      Point<2>    p;
    };

    std::vector<Node1D> nodes;
    nodes.reserve(node_tags.size());

    for (std::size_t i = 0; i < node_tags.size(); ++i)
      {
        const double x = coords[3*i + 0];
        const double y = coords[3*i + 1];
        const double u = (param_coords.empty() ? static_cast<double>(i) : param_coords[i]);
        nodes.push_back({node_tags[i], u, Point<2>(x,y)});
      }

    std::sort(nodes.begin(), nodes.end(),
              [](const Node1D &a, const Node1D &b){ return a.u < b.u; });

    top_spline.clear();
    top_u.clear();
    top_spline.reserve(nodes.size());
    top_u.reserve(nodes.size());

    std::unordered_set<std::size_t> seen;
    for (const auto &n : nodes)
      {
        if (!seen.insert(n.tag).second)
          continue;
        top_spline.push_back(n.p);
        top_u.push_back(n.u);
      }

    if (top_spline.size() < 5)
      throw std::runtime_error("Top spline sampling too small.");

    // Ensure starts near RIGHT endpoint of bottom segment (scaled): (s*0.5, 0)
    const Point<2> right_ref(s*0.5, 0.0);
    if ((top_spline.front() - right_ref).norm() > (top_spline.back() - right_ref).norm())
      {
        std::reverse(top_spline.begin(), top_spline.end());
        std::reverse(top_u.begin(), top_u.end());
      }

    // Build polygon (NOT closed): bottom left->right + spline interior nodes
    const Point<2> left(-s*0.5, 0.0);
    const Point<2> right(s*0.5, 0.0);

    polygon.clear();
    polygon.reserve(top_spline.size() + 2);
    polygon.push_back(left);
    polygon.push_back(right);

    for (unsigned int i = 1; i + 1 < top_spline.size(); ++i)
      polygon.push_back(top_spline[i]);

    std::cout << "Top curve nodes: " << top_spline.size() << " (expected ~101)\n";
    std::cout << "top_u range: [" << top_u.front() << ", " << top_u.back() << "]\n";
  }

  template <int dim>
  void ImmersedElasticity<dim>::setup_level_set()
  {
    level_set_dh.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dh.n_dofs());

    PolygonSignedDistance phi(polygon);
    VectorTools::interpolate(level_set_dh, phi, level_set);

    for (unsigned int i = 0; i < level_set.size(); ++i)
      if (!std::isfinite(level_set[i]))
        throw std::runtime_error("Level set contains NaN/Inf.");
  }

  template <int dim>
  void ImmersedElasticity<dim>::classify_and_distribute_dofs()
  {
    mesh_classifier.reclassify();

    if (fe_collection_vec.size() == 0)
      {
        fe_collection_vec.push_back(FESystem<dim>(FE_Q<dim>(fe_degree), dim));
        fe_collection_vec.push_back(FESystem<dim>(FE_Nothing<dim>(1), dim));
      }
    if (fe_collection_sca.size() == 0)
      {
        fe_collection_sca.push_back(FE_Q<dim>(fe_degree));
        fe_collection_sca.push_back(FE_Nothing<dim>(1));
      }

    for (auto cell = dof_handler_vec.begin_active(); cell != dof_handler_vec.end(); ++cell)
      {
        const auto loc = mesh_classifier.location_to_level_set(cell);
        cell->set_active_fe_index(loc == NonMatching::LocationToLevelSet::outside ?
                                  ActiveFEIndex::nothing :
                                  ActiveFEIndex::lagrange);
      }
    for (auto cell = dof_handler_sca.begin_active(); cell != dof_handler_sca.end(); ++cell)
      {
        const auto loc = mesh_classifier.location_to_level_set(cell);
        cell->set_active_fe_index(loc == NonMatching::LocationToLevelSet::outside ?
                                  ActiveFEIndex::nothing :
                                  ActiveFEIndex::lagrange);
      }

    dof_handler_vec.distribute_dofs(fe_collection_vec);
    dof_handler_sca.distribute_dofs(fe_collection_sca);

    constraints_vec.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_vec, constraints_vec);
    constraints_vec.close();

    constraints_sca.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_sca, constraints_sca);
    constraints_sca.close();

    std::cout << "DoFs (vec): " << dof_handler_vec.n_dofs() << "\n";
    std::cout << "DoFs (sca): " << dof_handler_sca.n_dofs() << "\n";
  }


  template <int dim>
  void ImmersedElasticity<dim>::setup_system()
  {
    // vector system
    {
      DynamicSparsityPattern dsp(dof_handler_vec.n_dofs(), dof_handler_vec.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_vec, dsp, constraints_vec, true);
      sparsity_vec.copy_from(dsp);

      K.reinit(sparsity_vec);
      U.reinit(dof_handler_vec.n_dofs());
      rhs.reinit(dof_handler_vec.n_dofs());
    }

    // scalar system
    {
      DynamicSparsityPattern dsp(dof_handler_sca.n_dofs(), dof_handler_sca.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_sca, dsp, constraints_sca, true);
      sparsity_sca.copy_from(dsp);

      M.reinit(sparsity_sca);

      von_mises.reinit(dof_handler_sca.n_dofs());
      octShearS.reinit(dof_handler_sca.n_dofs());
      hydroD.reinit(dof_handler_sca.n_dofs());
      mi.reinit(dof_handler_sca.n_dofs());

      rhs_vm.reinit(dof_handler_sca.n_dofs());
      rhs_os.reinit(dof_handler_sca.n_dofs());
      rhs_hd.reinit(dof_handler_sca.n_dofs());
      rhs_mi.reinit(dof_handler_sca.n_dofs());

      phi_nodal.reinit(dof_handler_sca.n_dofs());
      mi_reference.reinit(dof_handler_sca.n_dofs());
      mi_oss_source.reinit(dof_handler_sca.n_dofs());
      phi_oss_nodal.reinit(dof_handler_sca.n_dofs());
    }

    initialize_restart_fields();
  }

  template <int dim>
  void ImmersedElasticity<dim>::initialize_restart_fields()
  {
    mi_reference = 0.0;
    mi_oss_source = 0.0;
    phi_oss_nodal = 0.0;
    phi_oss_cell_dealii = 0.0;
    mi_reference_cell_dealii = 0.0;
    mi_oss_source_cell_dealii = 0.0;
    oss_material_cell_dealii = 0.0;
    enable_ossification_material = false;
    mi_threshold = std::numeric_limits<double>::quiet_NaN();
    mi_axis_max = std::numeric_limits<double>::quiet_NaN();
    mi_axis_y_at_max = std::numeric_limits<double>::quiet_NaN();
    mi_axis_cut_y0 = std::numeric_limits<double>::quiet_NaN();
    mi_axis_cut_y1 = std::numeric_limits<double>::quiet_NaN();
    mi_axis_total_length = std::numeric_limits<double>::quiet_NaN();
    cartilage_area_total = std::numeric_limits<double>::quiet_NaN();
    cartilage_area_target = std::numeric_limits<double>::quiet_NaN();
    cartilage_area_achieved = std::numeric_limits<double>::quiet_NaN();
  }

  template <int dim>
  std::string ImmersedElasticity<dim>::tagged_prefix() const
  {
    return "output_growth/" + prefix + "_" + std::to_string(iteration_id);
  }

  template <int dim>
  std::string ImmersedElasticity<dim>::restart_filename(const unsigned int id) const
  {
    return "output_growth/" + prefix + "_restart_" + std::to_string(id) + ".dat";
  }

  template <int dim>
  template <typename CellIterator>
  bool ImmersedElasticity<dim>::cell_is_bulk_top(const CellIterator &cell) const
  {
    double phi_min = std::numeric_limits<double>::max();
    double phi_max = -std::numeric_limits<double>::max();

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const double ph = cell->vertex(v)[1] - interfaceY;
        phi_min = std::min(phi_min, ph);
        phi_max = std::max(phi_max, ph);
      }

    const bool is_cut = (phi_min <= 0.0 && phi_max >= 0.0);
    const bool is_top = (cell->center()[1] >= interfaceY);
    return (!is_cut) && is_top;
  }

  template <int dim>
  double ImmersedElasticity<dim>::compute_bulk_top_area() const
  {
    const QGauss<dim> quadrature(fe_degree + 1);
    FEValues<dim>     fe_values(fe_collection_sca[ActiveFEIndex::lagrange],
                            quadrature,
                            update_JxW_values | update_quadrature_points);

    double area = 0.0;
    for (auto cell = dof_handler_sca.begin_active(); cell != dof_handler_sca.end(); ++cell)
      {
        if (cell->active_fe_index() != ActiveFEIndex::lagrange)
          continue;
        if (!cell_is_bulk_top(cell))
          continue;

        fe_values.reinit(cell);
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const double w = fe_values.JxW(q);
            if (std::isfinite(w) && w > 0.0)
              area += w;
          }
      }

    return area;
  }

  template <int dim>
  double ImmersedElasticity<dim>::compute_ossified_area_for_threshold(
    const Vector<double> &field,
    const double          threshold) const
  {
    const QGauss<dim> quadrature(fe_degree + 1);
    FEValues<dim>     fe_values(fe_collection_sca[ActiveFEIndex::lagrange],
                            quadrature,
                            update_values | update_JxW_values | update_quadrature_points);

    const unsigned int dofs_s =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;
    std::vector<types::global_dof_index> local_s(dofs_s);

    double area = 0.0;
    for (auto cell = dof_handler_sca.begin_active(); cell != dof_handler_sca.end(); ++cell)
      {
        if (cell->active_fe_index() != ActiveFEIndex::lagrange)
          continue;
        if (!cell_is_bulk_top(cell))
          continue;

        fe_values.reinit(cell);

        cell->get_dof_indices(local_s);
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const double w = fe_values.JxW(q);
            if (!std::isfinite(w) || w <= 0.0)
              continue;

            double field_q = 0.0;
            for (unsigned int j = 0; j < dofs_s; ++j)
              field_q += field[local_s[j]] * fe_values.shape_value(j, q);

            if (std::isfinite(field_q) && field_q >= threshold)
              area += w;
          }
      }

    return area;
  }

  template <int dim>
  std::vector<std::pair<double, double>>
  ImmersedElasticity<dim>::extract_axis_profile(const Vector<double> &field) const
  {
    MappingQ1<dim> mapping;
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler_sca, support_points);

    std::vector<std::pair<double, double>> profile;
    profile.reserve(support_points.size());

    double x_tol = std::numeric_limits<double>::max();
    if (!support_points.empty())
    {
      const double span_x = bg_p2[0] - bg_p1[0];
      x_tol = std::max(1e-12, 1e-10 * span_x);
    }

    for (const auto &kv : support_points)
      {
        const auto dof = kv.first;
        const auto &pt = kv.second;
        if (dof >= field.size())
          continue;
        if (!std::isfinite(field[dof]))
          continue;
        if (std::abs(pt[0]) > x_tol)
          continue;

        profile.emplace_back(pt[1], field[dof]);
      }

    std::sort(profile.begin(),
              profile.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<std::pair<double, double>> unique_profile;
    unique_profile.reserve(profile.size());

    for (const auto &yv : profile)
      {
        if (unique_profile.empty() ||
            std::abs(yv.first - unique_profile.back().first) > 1e-12)
          unique_profile.push_back(yv);
        else
          unique_profile.back().second = 0.5 * (unique_profile.back().second + yv.second);
      }

    return unique_profile;
  }

  template <int dim>
  bool ImmersedElasticity<dim>::find_axis_cuts_for_threshold(
    const std::vector<std::pair<double, double>> &profile,
    const double                                  threshold,
    double                                       &y_left,
    double                                       &y_right) const
  {
    if (profile.size() < 3)
      return false;

    auto it_max = std::max_element(
      profile.begin(),
      profile.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

    const std::size_t imax = static_cast<std::size_t>(std::distance(profile.begin(), it_max));

    bool left_found = false;
    for (std::size_t i = imax; i > 0; --i)
      {
        const double y0 = profile[i - 1].first;
        const double y1 = profile[i].first;
        const double f0 = profile[i - 1].second - threshold;
        const double f1 = profile[i].second - threshold;

        if ((f0 <= 0.0 && f1 >= 0.0) || (f0 >= 0.0 && f1 <= 0.0))
          {
            if (std::abs(f1 - f0) < 1e-18)
              y_left = 0.5 * (y0 + y1);
            else
              y_left = y0 + (0.0 - f0) * (y1 - y0) / (f1 - f0);
            left_found = true;
            break;
          }
      }

    bool right_found = false;
    for (std::size_t i = imax; i + 1 < profile.size(); ++i)
      {
        const double y0 = profile[i].first;
        const double y1 = profile[i + 1].first;
        const double f0 = profile[i].second - threshold;
        const double f1 = profile[i + 1].second - threshold;

        if ((f0 >= 0.0 && f1 <= 0.0) || (f0 <= 0.0 && f1 >= 0.0))
          {
            if (std::abs(f1 - f0) < 1e-18)
              y_right = 0.5 * (y0 + y1);
            else
              y_right = y0 + (0.0 - f0) * (y1 - y0) / (f1 - f0);
            right_found = true;
            break;
          }
      }

    return left_found && right_found && (y_right >= y_left);
  }

  template <int dim>
  void ImmersedElasticity<dim>::compute_area_based_threshold_from_field(
    const Vector<double> &mi_source)
  {
    const auto profile = extract_axis_profile(mi_source);
    if (profile.size() < 3)
      throw std::runtime_error("Axis profile too small to determine mi threshold.");

    const auto it_max = std::max_element(
      profile.begin(),
      profile.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

    mi_axis_max          = it_max->second;
    mi_axis_y_at_max     = it_max->first;
    mi_axis_total_length = profile.back().first - profile.front().first;

    cartilage_area_total = compute_bulk_top_area();
    if (!(cartilage_area_total > 0.0))
      throw std::runtime_error("Bulk cartilage area is non-positive.");

    cartilage_area_target = oss_cartilage_area_fraction * cartilage_area_total;

    double fmin = std::numeric_limits<double>::infinity();
    double fmax = -std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < mi_source.size(); ++i)
      if (std::isfinite(mi_source[i]))
        {
          fmin = std::min(fmin, mi_source[i]);
          fmax = std::max(fmax, mi_source[i]);
        }

    if (!std::isfinite(fmin) || !std::isfinite(fmax))
      throw std::runtime_error("mi_source has no finite values.");

    double a = std::nextafter(fmin, -std::numeric_limits<double>::infinity());
    double b = std::nextafter(fmax, +std::numeric_limits<double>::infinity());

    double area_a = compute_ossified_area_for_threshold(mi_source, a);
    double area_b = compute_ossified_area_for_threshold(mi_source, b);

    if (!(area_a >= cartilage_area_target && area_b <= cartilage_area_target))
      throw std::runtime_error("Could not bracket area-based threshold in cartilage.");

    for (unsigned int iter = 0; iter < 80; ++iter)
      {
        const double mid = 0.5 * (a + b);
        const double area_mid = compute_ossified_area_for_threshold(mi_source, mid);

        if (area_mid > cartilage_area_target)
          a = mid;
        else
          b = mid;
      }

    mi_threshold = 0.5 * (a + b);
    cartilage_area_achieved = compute_ossified_area_for_threshold(mi_source, mi_threshold);

    if (!find_axis_cuts_for_threshold(profile, mi_threshold, mi_axis_cut_y0, mi_axis_cut_y1))
      {
        mi_axis_cut_y0 = std::numeric_limits<double>::quiet_NaN();
        mi_axis_cut_y1 = std::numeric_limits<double>::quiet_NaN();
      }
  }


  template <int dim>
  void ImmersedElasticity<dim>::build_center_guided_oss_source_from_reference(
    const double y_focus_center)
  {
    mi_oss_source = 0.0;

    MappingQ1<dim> mapping;
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler_sca, support_points);

    const double sx = std::max(1e-12, 0.45 * oss_focus_x_halfwidth);
    const double sy = std::max(1e-12, 0.45 * oss_focus_y_halfwidth);

    for (const auto &kv : support_points)
      {
        const auto dof = kv.first;
        if (dof >= mi_reference.size())
          continue;

        const Point<dim> &pt = kv.second;

        double gate = 1.0;
        if (use_center_guided_ossification)
          {
            const bool inside_window =
              (std::abs(pt[0]) <= oss_focus_x_halfwidth) &&
              (std::abs(pt[1] - y_focus_center) <= oss_focus_y_halfwidth) &&
              (pt[1] >= interfaceY);

            if (!inside_window)
              gate = 0.0;
            else
              {
                const double gx = pt[0] / sx;
                const double gy = (pt[1] - y_focus_center) / sy;
                gate = std::exp(-0.5 * (gx * gx + gy * gy));
              }
          }

        mi_oss_source[dof] = mi_reference[dof] * gate;
      }

    constraints_sca.distribute(mi_oss_source);
  }

  template <int dim>
  void ImmersedElasticity<dim>::build_ossification_from_reference_field(
    const Vector<double> &mi_source)
  {
    if (mi_source.size() != dof_handler_sca.n_dofs())
      throw std::runtime_error("mi_source size mismatch when building ossification field.");

    mi_reference = mi_source;
    constraints_sca.distribute(mi_reference);

    mi_oss_source = mi_reference;
    constraints_sca.distribute(mi_oss_source);

    compute_area_based_threshold_from_field(mi_reference);

    for (unsigned int i = 0; i < phi_oss_nodal.size(); ++i)
      phi_oss_nodal[i] = mi_reference[i] - mi_threshold;

    constraints_sca.distribute(phi_oss_nodal);
    update_reference_cell_exports();

    std::cout << "Axis mi max = " << mi_axis_max
              << " at y = " << mi_axis_y_at_max
              << ", threshold = " << mi_threshold
              << ", cuts = [" << mi_axis_cut_y0 << ", " << mi_axis_cut_y1 << "]"
              << ", cartilage area target fraction = " << oss_cartilage_area_fraction
              << ", target area = " << cartilage_area_target
              << ", achieved area = " << cartilage_area_achieved << "\n";
  }

  template <int dim>
  void ImmersedElasticity<dim>::update_reference_cell_exports()
  {
    phi_oss_cell_dealii = 0.0;
    mi_reference_cell_dealii = 0.0;
    mi_oss_source_cell_dealii = 0.0;
    oss_material_cell_dealii = 0.0;

    if (dof_handler_sca.n_dofs() == 0)
      return;

    const unsigned int dofs_s =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;
    std::vector<types::global_dof_index> local_s(dofs_s);

    for (auto cell = dof_handler_sca.begin_active(); cell != dof_handler_sca.end(); ++cell)
      {
        const unsigned int cid = cell->active_cell_index();

        if (cell->active_fe_index() != ActiveFEIndex::lagrange || !cell_is_bulk_top(cell))
          {
            phi_oss_cell_dealii[cid] = 0.0f;
            mi_reference_cell_dealii[cid] = 0.0f;
            mi_oss_source_cell_dealii[cid] = 0.0f;
            oss_material_cell_dealii[cid] = 0.0f;
            continue;
          }

        cell->get_dof_indices(local_s);
        double mi_ref_avg = 0.0;
        double mi_oss_avg = 0.0;
        for (const auto idx : local_s)
          {
            mi_ref_avg += mi_reference[idx];
            mi_oss_avg += mi_oss_source[idx];
          }
        mi_ref_avg /= static_cast<double>(dofs_s);
        mi_oss_avg /= static_cast<double>(dofs_s);

        mi_reference_cell_dealii[cid] = static_cast<float>(mi_ref_avg);
        mi_oss_source_cell_dealii[cid] = static_cast<float>(mi_oss_avg);
        phi_oss_cell_dealii[cid]      = static_cast<float>(mi_ref_avg - mi_threshold);
        oss_material_cell_dealii[cid] =
          static_cast<float>((std::isfinite(mi_threshold) && mi_ref_avg >= mi_threshold) ? 1.0 : 0.0);
      }
  }

  template <int dim>
  void ImmersedElasticity<dim>::save_restart_bundle(
    const Vector<double>      &U_avg,
    const Vector<double>      &vm_avg,
    const Vector<double>      &os_avg,
    const Vector<double>      &hd_avg,
    const Vector<double>      &mi_avg,
    const std::vector<double> &vm_cell_avg,
    const std::vector<double> &os_cell_avg,
    const std::vector<double> &hd_cell_avg,
    const std::vector<double> &mi_cell_avg,
    const std::vector<double> &p_cell_avg) const
  {
    (void)U_avg;
    (void)vm_avg;
    (void)os_avg;
    (void)hd_avg;
    (void)vm_cell_avg;
    (void)os_cell_avg;
    (void)hd_cell_avg;
    (void)mi_cell_avg;
    (void)p_cell_avg;

    const std::string filename = restart_filename(iteration_id);
    std::ofstream out(filename);
    if (!out)
      throw std::runtime_error("Could not open restart file for writing: " + filename);

    out << std::setprecision(17);
    out << "SOC15_MI_RESTART_V2\n";
    out << iteration_id << "\n";
    out << triangulation.n_active_cells() << "\n";
    out << dof_handler_sca.n_dofs() << "\n";
    out << oss_cartilage_area_fraction << "\n";
    out << mi_threshold << "\n";
    out << mi_axis_max << "\n";
    out << mi_axis_y_at_max << "\n";
    out << mi_axis_cut_y0 << "\n";
    out << mi_axis_cut_y1 << "\n";
    out << mi_axis_total_length << "\n";

    auto write_vec = [&out](const Vector<double> &v)
    {
      out << v.size() << "\n";
      for (unsigned int i = 0; i < v.size(); ++i)
        out << v[i] << "\n";
    };

    write_vec(mi_avg);
    write_vec(mi_reference);

    std::cout << "Wrote restart bundle " << filename << "";
  }

  template <int dim>
  void ImmersedElasticity<dim>::load_restart_bundle(Vector<double> &mi_loaded,
                                                   Vector<double> &mi_reference_loaded)
  {
    const std::string filename = restart_filename(import_iteration_id);
    std::ifstream in(filename);
    if (!in)
      throw std::runtime_error("Could not open restart file for reading: " + filename);

    std::string magic;
    std::getline(in, magic);
    if (magic != "SOC15_MI_RESTART_V2")
      throw std::runtime_error(
        "Restart file has wrong format/version for this growth build: " + filename +
        ". Re-run export_mi once to regenerate the restart bundle.");

    unsigned int stored_iteration = 0;
    unsigned int n_cells = 0;
    unsigned int n_sca = 0;
    double stored_cartilage_area_fraction = 0.0;

    in >> stored_iteration;
    in >> n_cells;
    in >> n_sca;
    in >> stored_cartilage_area_fraction;
    in >> mi_threshold;
    in >> mi_axis_max;
    in >> mi_axis_y_at_max;
    in >> mi_axis_cut_y0;
    in >> mi_axis_cut_y1;
    in >> mi_axis_total_length;

    if (n_cells != triangulation.n_active_cells())
      throw std::runtime_error("Restart n_active_cells mismatch.");
    if (n_sca != dof_handler_sca.n_dofs())
      throw std::runtime_error("Restart scalar dof count mismatch.");

    auto read_vec = [&in](Vector<double> &v, const std::string &name)
    {
      unsigned int n = 0;
      in >> n;
      if (n != v.size())
        throw std::runtime_error("Restart vector size mismatch for " + name +
                                 ": file=" + std::to_string(n) +
                                 ", current=" + std::to_string(v.size()));
      for (unsigned int i = 0; i < n; ++i)
        in >> v[i];
    };

    read_vec(mi_loaded, "mi_loaded");
    read_vec(mi_reference_loaded, "mi_reference_loaded");

    std::cout << "Loaded restart bundle " << filename
              << " from iteration " << stored_iteration
              << " (area fraction saved=" << stored_cartilage_area_fraction << ")";
  }

  template <int dim>
  LameParameters ImmersedElasticity<dim>::choose_lame(const Point<dim> &p,
                                                      const double      mi_ref_value,
                                                      const bool        allow_ossification) const
  {
    if (allow_ossification &&
        enable_ossification_material &&
        std::isfinite(mi_threshold) &&
        std::isfinite(mi_ref_value) &&
        (mi_ref_value >= mi_threshold))
      return lame_from_E_nu(youngMid, poissonMid);

    const bool isBelow = (p[1] < interfaceY);
    const double E  = isBelow ? youngBelow : youngAbove;
    const double nu = isBelow ? poissonBelow : poissonAbove;
    return lame_from_E_nu(E, nu);
  }

  template <int dim>
  void ImmersedElasticity<dim>::assemble_stiffness()
  {
    K = 0;

    const FEValuesExtractors::Vector u_ex(0);
    const QGauss<1> quadrature_1d(fe_degree + 1);

    NonMatching::RegionUpdateFlags flags_vec;
    flags_vec.inside  = update_gradients | update_JxW_values | update_quadrature_points;
    flags_vec.surface = update_values | update_gradients | update_JxW_values | update_quadrature_points;

    NonMatching::RegionUpdateFlags flags_sca;
    flags_sca.inside  = update_values | update_JxW_values | update_quadrature_points;
    flags_sca.surface = update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> nm_vec(fe_collection_vec,
                                      quadrature_1d,
                                      flags_vec,
                                      mesh_classifier,
                                      level_set_dh,
                                      level_set);

    NonMatching::FEValues<dim> nm_sca(fe_collection_sca,
                                      quadrature_1d,
                                      flags_sca,
                                      mesh_classifier,
                                      level_set_dh,
                                      level_set);

    const unsigned int dofs_per_cell =
      fe_collection_vec[ActiveFEIndex::lagrange].dofs_per_cell;
    const unsigned int dofs_sca =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;

    FullMatrix<double> local_K(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_sca_indices(dofs_sca);

    auto cell_s = dof_handler_sca.begin_active();
    for (auto cell = dof_handler_vec.begin_active(); cell != dof_handler_vec.end(); ++cell, ++cell_s)
      {
        if (cell->active_fe_index() != ActiveFEIndex::lagrange)
          continue;

        local_K = 0;
        const double h = cell->minimum_vertex_distance();

        nm_vec.reinit(cell);
        nm_sca.reinit(cell);

        const bool allow_ossification_here = cell_is_bulk_top(cell);

        cell_s->get_dof_indices(local_sca_indices);

        // volume
        const auto &inside = nm_vec.get_inside_fe_values();
        const auto &inside_sca = nm_sca.get_inside_fe_values();
        if (inside && inside_sca)
          {
            for (const unsigned int q : inside->quadrature_point_indices())
              {
                const Point<dim> &xq = inside->quadrature_point(q);
                const double      w  = inside->JxW(q);
                if (!std::isfinite(w) || w <= 0.0)
                  continue;

                double mi_ref_q = 0.0;
                for (unsigned int j = 0; j < dofs_sca; ++j)
                  mi_ref_q += mi_oss_source[local_sca_indices[j]] * inside_sca->shape_value(j, q);

                const auto lame = choose_lame(xq, mi_ref_q, allow_ossification_here);
                const double lambda = lame.lambda;
                const double mu     = lame.mu;

                for (const unsigned int i : inside->dof_indices())
                  {
                    const SymmetricTensor<2, dim> eps_i =
                      symmetrize(inside->operator[](u_ex).gradient(i, q));
                    const double div_i = inside->operator[](u_ex).divergence(i, q);

                    for (const unsigned int j : inside->dof_indices())
                      {
                        const SymmetricTensor<2, dim> eps_j =
                          symmetrize(inside->operator[](u_ex).gradient(j, q));
                        const double div_j = inside->operator[](u_ex).divergence(j, q);

                        local_K(i, j) +=
                          (2.0 * mu * (eps_i * eps_j) + lambda * div_i * div_j) * w;
                      }
                  }
              }
          }

        // bottom clamp Nitsche (symmetric)
        const auto &surface = nm_vec.get_surface_fe_values();
        const auto &surface_sca = nm_sca.get_surface_fe_values();
        if (surface && surface_sca)
          {
            for (const unsigned int q : surface->quadrature_point_indices())
              {
                const Point<dim> &xq = surface->quadrature_point(q);
                const double ws = surface->JxW(q);
                if (!std::isfinite(ws) || ws <= 0.0)
                  continue;

                double mi_ref_q = 0.0;
                for (unsigned int j = 0; j < dofs_sca; ++j)
                  mi_ref_q += mi_oss_source[local_sca_indices[j]] * surface_sca->shape_value(j, q);

                const auto lame = choose_lame(xq, mi_ref_q, allow_ossification_here);
                const double lambda = lame.lambda;
                const double mu     = lame.mu;

                const double stiffness = (2.0 * mu + lambda);
                const double penalty =
                  20.0 * stiffness * (fe_degree + 1) * (fe_degree + 1) / std::max(1e-16, h);

                const double tol = 1e-10 + 1e-6 * h;
                const bool is_bottom =
                  (std::abs(xq[1] - y_bottom) < tol) &&
                  (xq[0] >= x_bottom_min - tol) && (xq[0] <= x_bottom_max + tol);

                if (!is_bottom)
                  continue;

                Tensor<1, dim> n;
                n[0] = 0.0;
                n[1] = -1.0;

                for (const unsigned int i : surface->dof_indices())
                  {
                    const Tensor<1, dim> v_i = surface->operator[](u_ex).value(i, q);

                    const Tensor<2, dim> grad_v_i = surface->operator[](u_ex).gradient(i, q);
                    const SymmetricTensor<2, dim> eps_v_i = symmetrize(grad_v_i);
                    const double div_v_i = trace(grad_v_i);

                    const Tensor<1, dim> sigma_vn =
                      (2.0 * mu * eps_v_i +
                       lambda * div_v_i * unit_symmetric_tensor<dim>()) * n;

                    for (const unsigned int j : surface->dof_indices())
                      {
                        const Tensor<1, dim> u_j = surface->operator[](u_ex).value(j, q);

                        const Tensor<2, dim> grad_u_j = surface->operator[](u_ex).gradient(j, q);
                        const SymmetricTensor<2, dim> eps_u_j = symmetrize(grad_u_j);
                        const double div_u_j = trace(grad_u_j);

                        const Tensor<1, dim> sigma_un =
                          (2.0 * mu * eps_u_j +
                           lambda * div_u_j * unit_symmetric_tensor<dim>()) * n;

                        local_K(i, j) +=
                          (-(sigma_un * v_i)
                           -(sigma_vn * u_j)
                           + penalty * (u_j * v_i)) * ws;
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints_vec.distribute_local_to_global(local_K, local_dof_indices, K);
      }

    constraints_vec.condense(K);
    std::cout << "Stiffness: Kf=" << K.frobenius_norm() << "\n";
  }

  // Continuous u-parametric load assembly.
  template <int dim>
  void ImmersedElasticity<dim>::assemble_rhs_top(const double u_center,
                                                const double u_radius,
                                                const double p_peak)
  {
    rhs = 0;
    cell_pressure_dealii = 0;
    std::fill(pressure_cell.begin(), pressure_cell.end(), 0.0);

    if (top_spline.size() < 2 || top_u.size() != top_spline.size())
      throw std::runtime_error("Top spline storage invalid.");

    const FEValuesExtractors::Vector u_ex(0);
    const QGauss<1> quadrature_1d(fe_degree + 1);

    NonMatching::RegionUpdateFlags flags;
    flags.surface = update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> nm(fe_collection_vec,
                                  quadrature_1d,
                                  flags,
                                  mesh_classifier,
                                  level_set_dh,
                                  level_set);

    const unsigned int dofs_per_cell =
      fe_collection_vec[ActiveFEIndex::lagrange].dofs_per_cell;

    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double R = std::max(1e-14, u_radius);
    const double ua = u_center - R;
    const double ub = u_center + R;

    for (auto cell = dof_handler_vec.begin_active(); cell != dof_handler_vec.end(); ++cell)
      {
        if (cell->active_fe_index() != ActiveFEIndex::lagrange)
          continue;

        local_rhs = 0;
        nm.reinit(cell);

        const auto &surface = nm.get_surface_fe_values();
        if (!surface)
          continue;

        for (const unsigned int q : surface->quadrature_point_indices())
          {
            const Point<dim> &xq = surface->quadrature_point(q);
            const double ws = surface->JxW(q);
            if (!std::isfinite(ws) || ws <= 0.0)
              continue;

            // Project once, keep only the local active window,
            // and build a local normalized parabola on that window.
            const auto proj =
              project_to_polyline_with_u(Point<2>(xq[0], xq[1]), top_spline, top_u);
            if (proj.seg + 1 >= top_spline.size())
              continue;

            const double h = cell->minimum_vertex_distance();
            if (std::sqrt(proj.dist2) > 3.0 * h)
              continue;

            const double us0 = top_u[proj.seg];
            const double us1 = top_u[proj.seg + 1];
            const double useg_min = std::min(us0, us1);
            const double useg_max = std::max(us0, us1);

            // Segment gating: only segments overlapping the active window matter.
            if (useg_max < ua || useg_min > ub)
              continue;

            const double tq0 = (us0 - u_center) / R;
            const double tq1 = (us1 - u_center) / R;
            const double tq  = tq0 + proj.alpha * (tq1 - tq0);
            if (std::abs(tq) >= 1.0)
              continue;

            Tensor<1, dim> n = unit_normal_from_top_spline(top_spline, proj.seg);
            if (n.norm() < 1e-14)
              continue;

            const double p = p_peak * (1.0 - tq * tq);
            const Tensor<1, dim> traction = -p * n;

            const unsigned int cid = cell->active_cell_index();
            pressure_cell[cid] = std::max(pressure_cell[cid], p);
            cell_pressure_dealii[cid] =
              std::max(cell_pressure_dealii[cid], static_cast<float>(p));

            for (const unsigned int i : surface->dof_indices())
              {
                const Tensor<1, dim> v_i = surface->operator[](u_ex).value(i, q);
                local_rhs(i) += (traction * v_i) * ws;
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints_vec.distribute_local_to_global(local_rhs, local_dof_indices, rhs);
      }

    constraints_vec.condense(rhs);

    std::cout << "RHS: ||rhs||=" << rhs.l2_norm()
              << " (u_center=" << u_center
              << ", u_radius=" << R
              << ", p_peak=" << p_peak << ")\n";
  }

  template <int dim>
  void ImmersedElasticity<dim>::init_preconditioner_K()
  {
    precond_K.initialize(K, ssor_omega);
    precond_K_ready = true;
  }

  template <int dim>
  void ImmersedElasticity<dim>::solve_vector_system()
  {
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm == 0.0)
      {
        U = 0.0;
        return;
      }

    const double tol = std::max(1e-14, cg_rel_tol_vec * rhs_norm);

    SolverControl solver_control(cg_max_iter_vec, tol);
    solver_control.log_result(true);

    SolverCG<Vector<double>> solver(solver_control);

    if (!precond_K_ready)
      throw std::runtime_error("Preconditioner for K not ready.");

    solver.solve(K, U, rhs, precond_K);
    constraints_vec.distribute(U);

    std::cout << "CG finished at step " << solver_control.last_step()
              << " residual " << solver_control.last_value()
              << " (target " << tol << ")\n";
  }

  template <int dim>
  void ImmersedElasticity<dim>::assemble_scalar_mass_matrix()
  {
    M = 0;

    const QGauss<1> quadrature_1d(fe_degree + 1);

    NonMatching::RegionUpdateFlags flags;
    flags.inside = update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> nm(fe_collection_sca,
                                  quadrature_1d,
                                  flags,
                                  mesh_classifier,
                                  level_set_dh,
                                  level_set);

    const unsigned int dofs_per_cell =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;

    FullMatrix<double> cellM(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_s(dofs_per_cell);

    for (auto cell = dof_handler_sca.begin_active(); cell != dof_handler_sca.end(); ++cell)
      {
        if (cell->active_fe_index() != ActiveFEIndex::lagrange)
          continue;

        cellM = 0;
        nm.reinit(cell);

        const auto &inside = nm.get_inside_fe_values();
        if (!inside)
          continue;

        for (const unsigned int q : inside->quadrature_point_indices())
          {
            const double w = inside->JxW(q);
            if (!std::isfinite(w) || w <= 0.0)
              continue;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const double phi_i = inside->shape_value(i, q);
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const double phi_j = inside->shape_value(j, q);
                    cellM(i, j) += phi_i * phi_j * w;
                  }
              }
          }

        cell->get_dof_indices(local_s);
        constraints_sca.distribute_local_to_global(cellM, local_s, M);
      }

    constraints_sca.condense(M);

    precond_M.initialize(M);
    precond_M_ready = true;

    std::cout << "Mass: Mf=" << M.frobenius_norm() << "\n";
  }


  template <int dim>
  void ImmersedElasticity<dim>::assemble_scalar_rhs_and_cell_avgs()
  {
    rhs_vm = 0;
    rhs_os = 0;
    rhs_hd = 0;
    rhs_mi = 0;

    std::fill(vm_cell.begin(), vm_cell.end(), 0.0);
    std::fill(os_cell.begin(), os_cell.end(), 0.0);
    std::fill(hd_cell.begin(), hd_cell.end(), 0.0);
    std::fill(mi_cell.begin(), mi_cell.end(), 0.0);

    const FEValuesExtractors::Vector u_ex(0);
    const QGauss<1> quadrature_1d(fe_degree + 1);

    NonMatching::RegionUpdateFlags flags_u;
    flags_u.inside = update_gradients | update_JxW_values | update_quadrature_points;

    NonMatching::RegionUpdateFlags flags_s;
    flags_s.inside = update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> nm_u(fe_collection_vec,
                                    quadrature_1d,
                                    flags_u,
                                    mesh_classifier,
                                    level_set_dh,
                                    level_set);

    NonMatching::FEValues<dim> nm_s(fe_collection_sca,
                                    quadrature_1d,
                                    flags_s,
                                    mesh_classifier,
                                    level_set_dh,
                                    level_set);

    const unsigned int dofs_u =
      fe_collection_vec[ActiveFEIndex::lagrange].dofs_per_cell;
    const unsigned int dofs_s =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;

    Vector<double> cellR_vm(dofs_s), cellR_os(dofs_s), cellR_hd(dofs_s), cellR_mi(dofs_s);
    std::vector<types::global_dof_index> local_u(dofs_u);
    std::vector<types::global_dof_index> local_s(dofs_s);

    auto cellU = dof_handler_vec.begin_active();
    auto cellS = dof_handler_sca.begin_active();

    for (; cellU != dof_handler_vec.end(); ++cellU, ++cellS)
      {
        if (cellU->active_fe_index() != ActiveFEIndex::lagrange)
          continue;

        nm_u.reinit(cellU);
        nm_s.reinit(cellU);

        const auto &insideU = nm_u.get_inside_fe_values();
        const auto &insideS = nm_s.get_inside_fe_values();
        if (!insideU || !insideS)
          continue;

        cellR_vm = 0;
        cellR_os = 0;
        cellR_hd = 0;
        cellR_mi = 0;

        cellU->get_dof_indices(local_u);
        cellS->get_dof_indices(local_s);

        const bool allow_ossification_here = cell_is_bulk_top(cellU);

        double area = 0.0;
        double sum_vm = 0.0, sum_os = 0.0, sum_hd = 0.0, sum_mi_ = 0.0;

        for (const unsigned int q : insideU->quadrature_point_indices())
          {
            const Point<dim> &xq = insideU->quadrature_point(q);
            const double w = insideU->JxW(q);
            if (!std::isfinite(w) || w <= 0.0)
              continue;

            double mi_ref_q = 0.0;
            for (unsigned int j = 0; j < dofs_s; ++j)
              mi_ref_q += mi_oss_source[local_s[j]] * insideS->shape_value(j, q);

            const auto lame = choose_lame(xq, mi_ref_q, allow_ossification_here);

            Tensor<2, dim> grad_u;
            for (unsigned int j = 0; j < dofs_u; ++j)
              grad_u += U[local_u[j]] * insideU->operator[](u_ex).gradient(j, q);

            const SymmetricTensor<2, dim> eps2d = symmetrize(grad_u);

            double vmq=0.0, hdq=0.0, osq=0.0, miq=0.0;
            compute_invariants_plane_strain(eps2d, lame.lambda, lame.mu, vmq, hdq, osq, miq);

            area   += w;
            sum_vm += vmq * w;
            sum_os += osq * w;
            sum_hd += hdq * w;
            sum_mi_+= miq * w;

            for (unsigned int i = 0; i < dofs_s; ++i)
              {
                const double phi_i = insideS->shape_value(i, q);
                cellR_vm(i) += vmq * phi_i * w;
                cellR_os(i) += osq * phi_i * w;
                cellR_hd(i) += hdq * phi_i * w;
                cellR_mi(i) += miq * phi_i * w;
              }
          }

        const unsigned int cid = cellU->active_cell_index();
        if (area > 1e-30)
          {
            const double invA = 1.0 / area;
            vm_cell[cid] = sum_vm * invA;
            os_cell[cid] = sum_os * invA;
            hd_cell[cid] = sum_hd * invA;
            mi_cell[cid] = sum_mi_ * invA;
          }

        constraints_sca.distribute_local_to_global(cellR_vm, local_s, rhs_vm);
        constraints_sca.distribute_local_to_global(cellR_os, local_s, rhs_os);
        constraints_sca.distribute_local_to_global(cellR_hd, local_s, rhs_hd);
        constraints_sca.distribute_local_to_global(cellR_mi, local_s, rhs_mi);
      }

    constraints_sca.condense(rhs_vm);
    constraints_sca.condense(rhs_os);
    constraints_sca.condense(rhs_hd);
    constraints_sca.condense(rhs_mi);

    // fill deal.II cell vectors for export
    for (unsigned int c = 0; c < triangulation.n_active_cells(); ++c)
      {
        vm_cell_dealii[c] = static_cast<float>(vm_cell[c]);
        os_cell_dealii[c] = static_cast<float>(os_cell[c]);
        hd_cell_dealii[c] = static_cast<float>(hd_cell[c]);
        mi_cell_dealii[c] = static_cast<float>(mi_cell[c]);
      }
  }

  template <int dim>
  void ImmersedElasticity<dim>::solve_scalar_systems()
  {
    if (!precond_M_ready)
      throw std::runtime_error("Preconditioner for M not ready.");

    auto solve_mass = [&](Vector<double> &x, const Vector<double> &b)
    {
      const double tol = 1e-12 * (b.l2_norm() + 1.0);
      SolverControl sc(20000, tol);
      SolverCG<Vector<double>> solver(sc);
      solver.solve(M, x, b, precond_M);
      constraints_sca.distribute(x);
    };

    solve_mass(von_mises, rhs_vm);
    solve_mass(octShearS, rhs_os);
    solve_mass(hydroD, rhs_hd);
    solve_mass(mi, rhs_mi);
  }

  template <int dim>
  void ImmersedElasticity<dim>::build_phi_nodal()
  {
    PhiFunction<dim> phi_fun(interfaceY);
    VectorTools::interpolate(dof_handler_sca, phi_fun, phi_nodal);
    constraints_sca.distribute(phi_nodal);
  }


  template <int dim>
  void ImmersedElasticity<dim>::fill_cell_outputs_static()
  {
    update_reference_cell_exports();

    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
      {
        const unsigned int idx = cell->active_cell_index();

        cell_id_dealii[idx]  = static_cast<float>(idx);
        phi_cell_dealii[idx] = static_cast<float>(cell->center()[1] - interfaceY);

        double phi_min = std::numeric_limits<double>::max();
        double phi_max = -std::numeric_limits<double>::max();
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            const double ph = cell->vertex(v)[1] - interfaceY;
            phi_min = std::min(phi_min, ph);
            phi_max = std::max(phi_max, ph);
          }

        const bool is_cut = (phi_min <= 0.0 && phi_max >= 0.0);

        unsigned int mat_id = 0;
        if (!is_cut)
          {
            const bool is_top = (cell->center()[1] >= interfaceY);
            mat_id = is_top ? 3u : 0u;
          }
        else
          {
            const bool cell_is_top = (cell->center()[1] >= interfaceY);
            mat_id = cell_is_top ? 2u : 1u;
          }

        if (mat_id == 3u && enable_ossification_material && oss_material_cell_dealii[idx] > 0.5f)
          mat_id = 4u;

        cell->set_material_id(mat_id);
        material_dealii[idx] = static_cast<float>(cell->material_id());
      }
  }

  // NEW: steps are defined in u-space (continuous)
  template <int dim>
  std::vector<typename ImmersedElasticity<dim>::StepDef>
  ImmersedElasticity<dim>::build_default_steps() const
  {
    if (top_u.size() < 2)
      throw std::runtime_error("top_u not ready when building steps.");

    const double u0 = top_u.front();
    const double u1 = top_u.back();
    const double ur = u1 - u0;

    // center from fraction
    const double load_du_frac       = 0.08; // shift per step as fraction of u-range
    const double uc = u0 + load_center_u_frac * ur;
    const double du = load_du_frac * ur;

    std::vector<StepDef> steps;
    // Tripled radiuses (4.5 * step) and x5 relative peak magnitudes
    const double overlap_radius = du;
    steps.push_back({uc + 2.0*du, overlap_radius, 0.5 * load_p_peak});
    steps.push_back({uc + du, overlap_radius, 0.75 * load_p_peak});
    steps.push_back({uc, overlap_radius, load_p_peak});
    steps.push_back({uc - du, overlap_radius, 0.75 * load_p_peak});
    steps.push_back({uc - 2.0*du, overlap_radius, 0.5 * load_p_peak});

    // clamp u centers
    for (auto &st : steps)
      st.u_center = std::min(std::max(st.u_center, u0), u1);

    return steps;
  }


  template <int dim>
  void ImmersedElasticity<dim>::output_one(const std::string &base,
                                          const Vector<double> &U_in,
                                          const Vector<double> &vm_in,
                                          const Vector<double> &os_in,
                                          const Vector<double> &hd_in,
                                          const Vector<double> &mi_in,
                                          const std::vector<double> &vm_cell_in,
                                          const std::vector<double> &os_cell_in,
                                          const std::vector<double> &hd_cell_in,
                                          const std::vector<double> &mi_cell_in,
                                          const std::vector<double> &pressure_cell_in) const
  {
    const std::string raw_file = base + ".raw.vtu";
    const std::string out_file = base + ".vtu";

    Vector<float> export_material_tag_local(triangulation.n_active_cells());
    for (unsigned int i = 0; i < export_material_tag_local.size(); ++i)
      export_material_tag_local[i] = 0.0f;

    const unsigned int dofs_per_cell_sca =
      fe_collection_sca[ActiveFEIndex::lagrange].dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices_sca(dofs_per_cell_sca);

    unsigned int c_tag = 0;
    auto cell_sca = dof_handler_sca.begin_active();
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell, ++cell_sca, ++c_tag)
      {
        const auto loc = mesh_classifier.location_to_level_set(cell);

        if (loc == NonMatching::LocationToLevelSet::outside)
          {
            export_material_tag_local[c_tag] = 0.0f;
            continue;
          }

        if (cell_sca->active_fe_index() != ActiveFEIndex::lagrange)
          {
            export_material_tag_local[c_tag] = 0.0f;
            continue;
          }

        cell_sca->get_dof_indices(local_dof_indices_sca);

        const bool is_cut_by_phi =
          (loc == NonMatching::LocationToLevelSet::intersected);

        double phi_oss_min = std::numeric_limits<double>::max();
        double phi_oss_max = -std::numeric_limits<double>::max();

        for (const auto dof : local_dof_indices_sca)
          {
            const double v = phi_oss_nodal[dof];
            phi_oss_min = std::min(phi_oss_min, v);
            phi_oss_max = std::max(phi_oss_max, v);
          }

        const bool is_cut_by_phi_oss =
          (phi_oss_min < 0.0 && phi_oss_max > 0.0);

        const bool is_any_cut = is_cut_by_phi || is_cut_by_phi_oss;

        if (is_any_cut)
          export_material_tag_local[c_tag] = 4.0f;
        else if (oss_material_cell_dealii[c_tag] > 0.5f)
          export_material_tag_local[c_tag] = 5.0f;
        else
          export_material_tag_local[c_tag] = 3.0f;
      }

    // deal.II output (raw)
    {
      DataOut<dim> data_out;

      std::vector<std::string> names = {"ux", "uy"};
      std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
        dim, DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_vec, U_in, names, interp);

      data_out.add_data_vector(dof_handler_sca, vm_in, "von_mises");
      data_out.add_data_vector(dof_handler_sca, os_in, "octShearS");
      data_out.add_data_vector(dof_handler_sca, hd_in, "hydroD");
      data_out.add_data_vector(dof_handler_sca, mi_in, "mi");

      data_out.add_data_vector(dof_handler_sca, phi_nodal, "phi_nodal");
      data_out.add_data_vector(dof_handler_sca, mi_reference, "mi_reference");
      data_out.add_data_vector(dof_handler_sca, mi_oss_source, "mi_oss_source");
      data_out.add_data_vector(dof_handler_sca, phi_oss_nodal, "phi_oss_nodal");
      data_out.add_data_vector(level_set_dh, level_set, "level_set");

      // static cell exports
      data_out.add_data_vector(material_dealii,          "material_dealii",          DataOut<dim>::type_cell_data);
      data_out.add_data_vector(cell_id_dealii,           "cell_id_dealii",           DataOut<dim>::type_cell_data);
      data_out.add_data_vector(phi_cell_dealii,          "phi_cell_dealii",          DataOut<dim>::type_cell_data);
      data_out.add_data_vector(phi_oss_cell_dealii,      "phi_oss_cell_dealii",      DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_reference_cell_dealii, "mi_reference_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_oss_source_cell_dealii,"mi_oss_source_cell_dealii",DataOut<dim>::type_cell_data);
      data_out.add_data_vector(oss_material_cell_dealii, "oss_material_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(export_material_tag_local, "export_material_tag_dealii", DataOut<dim>::type_cell_data);

      // per-run cell pressure
      data_out.add_data_vector(cell_pressure_dealii, "cell_pressure_dealii", DataOut<dim>::type_cell_data);

      // per-run deal.II cell invariants
      data_out.add_data_vector(vm_cell_dealii, "von_mises_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(os_cell_dealii, "octShearS_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(hd_cell_dealii, "hydroD_cell_dealii",    DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_cell_dealii, "mi_cell_dealii",        DataOut<dim>::type_cell_data);

      data_out.set_cell_selection(
        [this](const typename Triangulation<dim>::cell_iterator &cell_it) {
          return cell_it->is_active() &&
                 mesh_classifier.location_to_level_set(cell_it) !=
                   NonMatching::LocationToLevelSet::outside;
        });

      data_out.build_patches();
      std::ofstream out(raw_file);
      data_out.write_vtu(out);
    }

    // TRUE CellData rewrite
    std::vector<double> material_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> phi_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> phi_oss_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> mi_ref_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> mi_oss_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> oss_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> export_material_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> cartilage_indicator_cell(triangulation.n_active_cells(), 0.0);
    std::vector<double> vm_cartilage_cell(triangulation.n_active_cells(),
                                          std::numeric_limits<double>::quiet_NaN());
    std::vector<double> os_cartilage_cell(triangulation.n_active_cells(),
                                          std::numeric_limits<double>::quiet_NaN());
    std::vector<double> hd_cartilage_cell(triangulation.n_active_cells(),
                                          std::numeric_limits<double>::quiet_NaN());
    std::vector<double> mi_cartilage_cell(triangulation.n_active_cells(),
                                          std::numeric_limits<double>::quiet_NaN());

    unsigned int c = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell, ++c)
      {
        material_cell[c] = static_cast<double>(material_dealii[c]);
        phi_cell[c]      = static_cast<double>(phi_cell_dealii[c]);
        phi_oss_cell[c]  = static_cast<double>(phi_oss_cell_dealii[c]);
        mi_ref_cell[c]   = static_cast<double>(mi_reference_cell_dealii[c]);
        mi_oss_cell[c]   = static_cast<double>(mi_oss_source_cell_dealii[c]);
        oss_cell[c]      = static_cast<double>(oss_material_cell_dealii[c]);
        export_material_cell[c] = static_cast<double>(export_material_tag_local[c]);

        const double export_tag = static_cast<double>(export_material_tag_local[c]);

        if (export_tag == 3.0 || export_tag == 4.0 || export_tag == 5.0)
          cartilage_indicator_cell[c] = 1.0;

        if (export_tag == 3.0 || export_tag == 4.0)
          {
            vm_cartilage_cell[c] = vm_cell_in[c];
            os_cartilage_cell[c] = os_cell_in[c];
            hd_cartilage_cell[c] = hd_cell_in[c];
            mi_cartilage_cell[c] = mi_cell_in[c];
          }
      }

    vtk_postprocess_add_true_celldata(
      raw_file,
      out_file,
      "cell_id_dealii",
      {
        {"material_cell",          material_cell},
        {"phi_cell",               phi_cell},
        {"phi_oss_cell",           phi_oss_cell},
        {"mi_reference_cell",      mi_ref_cell},
        {"mi_oss_source_cell",     mi_oss_cell},
        {"oss_material_cell",      oss_cell},
        {"export_material_cell",   export_material_cell},
        {"cartilage_indicator_cell", cartilage_indicator_cell},
        {"cell_pressure",          pressure_cell_in},
        {"von_mises_cell",         vm_cell_in},
        {"octShearS_cell",         os_cell_in},
        {"hydroD_cell",            hd_cell_in},
        {"mi_cell",                mi_cell_in},
        {"von_mises_cartilage_cell", vm_cartilage_cell},
        {"octShearS_cartilage_cell", os_cartilage_cell},
        {"hydroD_cartilage_cell",    hd_cartilage_cell},
        {"mi_cartilage_cell",        mi_cartilage_cell}
      });

    std::remove(raw_file.c_str());
    std::cout << "Wrote " << out_file << "\n";

    const std::string raw_cart_file = base + "_cartilage.raw.vtu";
    const std::string out_cart_file = base + "_cartilage.vtu";

    {
      DataOut<dim> data_out;

      std::vector<std::string> names = {"ux", "uy"};
      std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
        dim, DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(dof_handler_vec, U_in, names, interp);

      data_out.add_data_vector(dof_handler_sca, vm_in, "von_mises");
      data_out.add_data_vector(dof_handler_sca, os_in, "octShearS");
      data_out.add_data_vector(dof_handler_sca, hd_in, "hydroD");
      data_out.add_data_vector(dof_handler_sca, mi_in, "mi");

      data_out.add_data_vector(dof_handler_sca, phi_nodal, "phi_nodal");
      data_out.add_data_vector(dof_handler_sca, mi_reference, "mi_reference");
      data_out.add_data_vector(dof_handler_sca, mi_oss_source, "mi_oss_source");
      data_out.add_data_vector(dof_handler_sca, phi_oss_nodal, "phi_oss_nodal");
      data_out.add_data_vector(level_set_dh, level_set, "level_set");

      data_out.add_data_vector(material_dealii,          "material_dealii",          DataOut<dim>::type_cell_data);
      data_out.add_data_vector(cell_id_dealii,           "cell_id_dealii",           DataOut<dim>::type_cell_data);
      data_out.add_data_vector(phi_cell_dealii,          "phi_cell_dealii",          DataOut<dim>::type_cell_data);
      data_out.add_data_vector(phi_oss_cell_dealii,      "phi_oss_cell_dealii",      DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_reference_cell_dealii, "mi_reference_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_oss_source_cell_dealii,"mi_oss_source_cell_dealii",DataOut<dim>::type_cell_data);
      data_out.add_data_vector(oss_material_cell_dealii, "oss_material_cell_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(export_material_tag_local, "export_material_tag_dealii", DataOut<dim>::type_cell_data);
      data_out.add_data_vector(cell_pressure_dealii,     "cell_pressure_dealii",     DataOut<dim>::type_cell_data);
      data_out.add_data_vector(vm_cell_dealii,           "von_mises_cell_dealii",    DataOut<dim>::type_cell_data);
      data_out.add_data_vector(os_cell_dealii,           "octShearS_cell_dealii",    DataOut<dim>::type_cell_data);
      data_out.add_data_vector(hd_cell_dealii,           "hydroD_cell_dealii",       DataOut<dim>::type_cell_data);
      data_out.add_data_vector(mi_cell_dealii,           "mi_cell_dealii",           DataOut<dim>::type_cell_data);

      data_out.set_cell_selection(
        [this, &export_material_tag_local](const typename Triangulation<dim>::cell_iterator &cell_it) {
          if (!cell_it->is_active())
            return false;

          const unsigned int cid = cell_it->active_cell_index();
          return (export_material_tag_local[cid] == 3.0f ||
                  export_material_tag_local[cid] == 4.0f);
        });

      data_out.build_patches();
      std::ofstream out(raw_cart_file);
      data_out.write_vtu(out);
    }

    vtk_postprocess_add_true_celldata(
      raw_cart_file,
      out_cart_file,
      "cell_id_dealii",
      {
        {"material_cell",          material_cell},
        {"phi_cell",               phi_cell},
        {"phi_oss_cell",           phi_oss_cell},
        {"mi_reference_cell",      mi_ref_cell},
        {"mi_oss_source_cell",     mi_oss_cell},
        {"oss_material_cell",      oss_cell},
        {"export_material_cell",   export_material_cell},
        {"cartilage_indicator_cell", cartilage_indicator_cell},
        {"cell_pressure",          pressure_cell_in},
        {"von_mises_cell",         vm_cell_in},
        {"octShearS_cell",         os_cell_in},
        {"hydroD_cell",            hd_cell_in},
        {"mi_cell",                mi_cell_in},
        {"von_mises_cartilage_cell", vm_cartilage_cell},
        {"octShearS_cartilage_cell", os_cartilage_cell},
        {"hydroD_cartilage_cell",    hd_cartilage_cell},
        {"mi_cartilage_cell",        mi_cartilage_cell}
      });

    std::remove(raw_cart_file.c_str());
    std::cout << "Wrote " << out_cart_file << "\n";
  }


  template <int dim>
  void ImmersedElasticity<dim>::run()
  {
    std::filesystem::create_directories("output_growth");

    make_background_grid();
    build_geometry_from_gmsh_curve_nodes();
    setup_level_set();
    classify_and_distribute_dofs();
    setup_system();

    fill_cell_outputs_static();
    build_phi_nodal();

    if (run_mode == RunMode::import_mi)
      {
        Vector<double> mi_loaded(dof_handler_sca.n_dofs());
        Vector<double> mi_reference_loaded(dof_handler_sca.n_dofs());
        load_restart_bundle(mi_loaded, mi_reference_loaded);
        if (mi_reference_loaded.l2_norm() == 0.0 && mi_loaded.l2_norm() > 0.0)
          mi_reference_loaded = mi_loaded;
        build_ossification_from_reference_field(mi_reference_loaded);
        enable_ossification_material = true;
        fill_cell_outputs_static();
      }

    // assemble once
    assemble_stiffness();
    init_preconditioner_K();
    assemble_scalar_mass_matrix();

    // build steps (5)
    const auto steps = build_default_steps();

    // averaging accumulators (point fields)
    Vector<double> U_sum(U.size()); U_sum = 0.0;
    Vector<double> vm_sum(von_mises.size()); vm_sum = 0.0;
    Vector<double> os_sum(octShearS.size()); os_sum = 0.0;
    Vector<double> hd_sum(hydroD.size());    hd_sum = 0.0;
    Vector<double> mi_sum(mi.size());        mi_sum = 0.0;

    // averaging accumulators (cell fields)
    std::vector<double> vm_cell_sum(triangulation.n_active_cells(), 0.0);
    std::vector<double> os_cell_sum(triangulation.n_active_cells(), 0.0);
    std::vector<double> hd_cell_sum(triangulation.n_active_cells(), 0.0);
    std::vector<double> mi_cell_sum(triangulation.n_active_cells(), 0.0);
    std::vector<double> p_cell_sum (triangulation.n_active_cells(), 0.0);

    // step loop
    for (unsigned int sidx = 0; sidx < steps.size(); ++sidx)
      {
        assemble_rhs_top(steps[sidx].u_center, steps[sidx].u_radius, steps[sidx].p_peak);
        solve_vector_system();

        assemble_scalar_rhs_and_cell_avgs();
        solve_scalar_systems();

        // output step
        const std::string step_base =
          tagged_prefix() + "-step-" + std::to_string(sidx + 1);
        output_one(step_base,
                   U, von_mises, octShearS, hydroD, mi,
                   vm_cell, os_cell, hd_cell, mi_cell,
                   pressure_cell);

        // accumulate averages
        U_sum  += U;
        vm_sum += von_mises;
        os_sum += octShearS;
        hd_sum += hydroD;
        mi_sum += mi;

        for (unsigned int c = 0; c < triangulation.n_active_cells(); ++c)
          {
            vm_cell_sum[c] += vm_cell[c];
            os_cell_sum[c] += os_cell[c];
            hd_cell_sum[c] += hd_cell[c];
            mi_cell_sum[c] += mi_cell[c];
            p_cell_sum[c]  += pressure_cell[c];
          }
      }

    // averaged fields
    const double invN = 1.0 / static_cast<double>(steps.size());

    Vector<double> U_avg  = U_sum;  U_avg  *= invN;
    Vector<double> vm_avg = vm_sum; vm_avg *= invN;
    Vector<double> os_avg = os_sum; os_avg *= invN;
    Vector<double> hd_avg = hd_sum; hd_avg *= invN;
    Vector<double> mi_avg = mi_sum; mi_avg *= invN;

    constraints_vec.distribute(U_avg);
    constraints_sca.distribute(vm_avg);
    constraints_sca.distribute(os_avg);
    constraints_sca.distribute(hd_avg);
    constraints_sca.distribute(mi_avg);

    std::vector<double> vm_cell_avg(vm_cell_sum.size());
    std::vector<double> os_cell_avg(os_cell_sum.size());
    std::vector<double> hd_cell_avg(hd_cell_sum.size());
    std::vector<double> mi_cell_avg(mi_cell_sum.size());
    std::vector<double> p_cell_avg (p_cell_sum.size());

    for (unsigned int c = 0; c < triangulation.n_active_cells(); ++c)
      {
        vm_cell_avg[c] = vm_cell_sum[c] * invN;
        os_cell_avg[c] = os_cell_sum[c] * invN;
        hd_cell_avg[c] = hd_cell_sum[c] * invN;
        mi_cell_avg[c] = mi_cell_sum[c] * invN;
        p_cell_avg[c]  = p_cell_sum[c]  * invN;
      }

    // for averaged output, also refresh deal.II cell vectors used in raw export
    for (unsigned int c = 0; c < triangulation.n_active_cells(); ++c)
      {
        vm_cell_dealii[c] = static_cast<float>(vm_cell_avg[c]);
        os_cell_dealii[c] = static_cast<float>(os_cell_avg[c]);
        hd_cell_dealii[c] = static_cast<float>(hd_cell_avg[c]);
        mi_cell_dealii[c] = static_cast<float>(mi_cell_avg[c]);
        cell_pressure_dealii[c] = static_cast<float>(p_cell_avg[c]);
      }

    // In export mode, the averaged mi becomes the reference for the next stage.
    // In import mode, keep the imported reference frozen so phi_oss_nodal and the
    // ossification mask remain based on the previous simulation, not on the newly
    // recomputed mi_avg of the current run.
    if (run_mode == RunMode::export_mi)
      {
        const bool old_flag = enable_ossification_material;
        enable_ossification_material = false;
        build_ossification_from_reference_field(mi_avg);
        enable_ossification_material = old_flag;
      }
    else
      {
        enable_ossification_material = true;
        update_reference_cell_exports();
      }

    fill_cell_outputs_static();

    output_one(tagged_prefix(),
               U_avg, vm_avg, os_avg, hd_avg, mi_avg,
               vm_cell_avg, os_cell_avg, hd_cell_avg, mi_cell_avg,
               p_cell_avg);

    save_restart_bundle(U_avg, vm_avg, os_avg, hd_avg, mi_avg,
                        vm_cell_avg, os_cell_avg, hd_cell_avg, mi_cell_avg, p_cell_avg);
  }

} // namespace SOC

int main()
{
  try
    {
      SOC::ImmersedElasticity<2> prob;
      prob.run();
    }
  catch (std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << "\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception.\n";
      return 1;
    }
  return 0;
}
