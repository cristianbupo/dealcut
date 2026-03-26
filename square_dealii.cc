/**
 * @brief Bimaterial linear elasticity on unit square [0,1]x[0,1].
 *
 * Simplified reference problem using deal.II (no CutFEM, no Gmsh, no VTK).
 *
 * - Plane strain
 * - Bottom (y=0): clamped (zero Dirichlet)
 * - Top (y=1): parabolic Neumann load p(x) = p_peak * 4x(1-x)
 * - Material interface at y=0.5 (aligned with mesh)
 *   Below: E=500, nu=0.2 (bone)
 *   Above: E=6, nu=0.47 (cartilage)
 * - Single solve
 * - Output: displacement + stress invariants (von Mises, hydrostatic,
 *   octahedral shear, Miner index)
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Square
{
  using namespace dealii;

  // ============================================================
  // Material (plane strain)
  // ============================================================
  static constexpr double E_bone = 500.0, nu_bone = 0.2;
  static constexpr double E_cart = 6.0, nu_cart = 0.47;
  static constexpr double interface_y = 0.5;
  static constexpr double p_peak = 1.0;
  static constexpr double k_mi = 0.5;

  struct LameParams
  {
    double lambda, mu;
  };

  static LameParams
  lame_from_E_nu(double E, double nu)
  {
    return {E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))};
  }

  static LameParams
  lame_at(const Point<2> &p)
  {
    return (p[1] < interface_y) ? lame_from_E_nu(E_bone, nu_bone)
                                : lame_from_E_nu(E_cart, nu_cart);
  }

  // ============================================================
  // Stress invariants (plane strain)
  // ============================================================
  struct Invariants
  {
    double von_mises, hydrostatic, oct_shear, miner;
  };

  static Invariants
  compute_invariants(const SymmetricTensor<2, 2> &eps, double lam, double mu)
  {
    const SymmetricTensor<2, 2> sigma =
      2.0 * mu * eps + lam * trace(eps) * unit_symmetric_tensor<2>();
    const double s_zz = lam * trace(eps);

    const double hydro = (sigma[0][0] + sigma[1][1] + s_zz) / 3.0;

    const double d_xx = sigma[0][0] - hydro;
    const double d_yy = sigma[1][1] - hydro;
    const double d_zz = s_zz - hydro;

    const double J2 =
      d_xx * d_xx + d_yy * d_yy + d_zz * d_zz + 2.0 * sigma[0][1] * sigma[0][1];

    const double vm  = std::sqrt(1.5 * J2);
    const double oct = std::sqrt(2.0 / 3.0) * vm;

    return {vm, hydro, oct, oct + k_mi * std::min(hydro, 0.0)};
  }

  // ============================================================
  // Run
  // ============================================================
  void
  run(const unsigned int n_cells)
  {
    const unsigned int fe_degree = 1;

    // ---- Mesh ----
    // subdivided_hyper_rectangle with even n_cells ensures y=0.5 is a mesh line
    Triangulation<2>             tria;
    const std::vector<unsigned int> reps = {n_cells, n_cells};
    GridGenerator::subdivided_hyper_rectangle(
      tria, reps, Point<2>(0.0, 0.0), Point<2>(1.0, 1.0), /*colorize=*/true);
    // colorize: boundary_id 0=left, 1=right, 2=bottom, 3=top

    // ---- FE spaces ----
    const FESystem<2> fe(FE_Q<2>(fe_degree), 2);
    DoFHandler<2>     dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    std::cout << "Mesh: " << n_cells << "x" << n_cells
              << ", DOFs=" << dof_handler.n_dofs() << "\n";

    // ---- Dirichlet: u=0 on bottom (id=2) ----
    AffineConstraints<double> constraints;
    VectorTools::interpolate_boundary_values(
      dof_handler, 2, Functions::ZeroFunction<2>(2), constraints);
    constraints.close();

    // ---- Sparsity & system ----
    SparsityPattern      sparsity;
    SparseMatrix<double> K;
    Vector<double>       rhs, U;
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
      sparsity.copy_from(dsp);
    }
    K.reinit(sparsity);
    rhs.reinit(dof_handler.n_dofs());
    U.reinit(dof_handler.n_dofs());

    // ---- Assemble stiffness ----
    const QGauss<2>                    quad(fe_degree + 1);
    const FEValuesExtractors::Vector   u_ex(0);
    FEValues<2>                        fe_values(fe, quad,
                            update_values | update_gradients |
                              update_JxW_values | update_quadrature_points);

    const unsigned int                          dpc = fe.dofs_per_cell;
    FullMatrix<double>                          cell_K(dpc, dpc);
    Vector<double>                              cell_rhs(dpc);
    std::vector<types::global_dof_index>        local_dofs(dpc);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_K   = 0;
        cell_rhs = 0;
        cell->get_dof_indices(local_dofs);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const auto  &xq  = fe_values.quadrature_point(q);
            const double w   = fe_values.JxW(q);
            const auto   lp  = lame_at(xq);

            for (unsigned int i = 0; i < dpc; ++i)
              {
                const auto   eps_i = symmetrize(fe_values[u_ex].gradient(i, q));
                const double div_i = fe_values[u_ex].divergence(i, q);

                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    const auto   eps_j = symmetrize(fe_values[u_ex].gradient(j, q));
                    const double div_j = fe_values[u_ex].divergence(j, q);

                    cell_K(i, j) +=
                      (2.0 * lp.mu * (eps_i * eps_j) + lp.lambda * div_i * div_j) * w;
                  }
              }
          }

        constraints.distribute_local_to_global(cell_K, cell_rhs, local_dofs, K, rhs);
      }

    // ---- Assemble Neumann RHS on top (id=3) ----
    // Parabolic traction: t = (0, -p_peak * 4x(1-x))
    const QGauss<1> face_quad(fe_degree + 1);
    FEFaceValues<2> fe_face(fe, face_quad,
                            update_values | update_JxW_values |
                              update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(local_dofs);

        for (const unsigned int f : cell->face_indices())
          {
            if (!cell->face(f)->at_boundary() || cell->face(f)->boundary_id() != 3)
              continue;

            fe_face.reinit(cell, f);
            cell_rhs = 0;

            for (const unsigned int q : fe_face.quadrature_point_indices())
              {
                const auto  &xq = fe_face.quadrature_point(q);
                const double w  = fe_face.JxW(q);
                const double p  = p_peak * 4.0 * xq[0] * (1.0 - xq[0]);

                Tensor<1, 2> traction;
                traction[0] = 0.0;
                traction[1] = -p; // downward

                for (unsigned int i = 0; i < dpc; ++i)
                  cell_rhs(i) += (traction * fe_face[u_ex].value(i, q)) * w;
              }

            constraints.distribute_local_to_global(cell_rhs, local_dofs, rhs);
          }
      }

    // ---- Solve ----
    {
      const double tol = std::max(1e-14, 1e-10 * rhs.l2_norm());
      SolverControl              control(30000, tol);
      SolverCG<Vector<double>>   solver(control);
      PreconditionSSOR<SparseMatrix<double>> precond;
      precond.initialize(K, 1.2);
      solver.solve(K, U, rhs, precond);
      constraints.distribute(U);
      std::cout << "Solved in " << control.last_step() << " CG iterations.\n";
    }

    // ---- Post-process: cell-average stress invariants ----
    const unsigned int n_active = tria.n_active_cells();

    Vector<float> vm_cell(n_active), hd_cell(n_active),
                  oct_cell(n_active), mi_cell(n_active), mat_cell(n_active);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dofs);
        const unsigned int idx = cell->active_cell_index();

        double vm_sum = 0, hd_sum = 0, oct_sum = 0, mi_sum = 0, area = 0;

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const auto  &xq = fe_values.quadrature_point(q);
            const double w  = fe_values.JxW(q);
            const auto   lp = lame_at(xq);

            Tensor<2, 2> grad_u;
            for (unsigned int j = 0; j < dpc; ++j)
              grad_u += U[local_dofs[j]] * fe_values[u_ex].gradient(j, q);

            const auto eps = symmetrize(grad_u);
            const auto inv = compute_invariants(eps, lp.lambda, lp.mu);

            vm_sum  += inv.von_mises * w;
            hd_sum  += inv.hydrostatic * w;
            oct_sum += inv.oct_shear * w;
            mi_sum  += inv.miner * w;
            area    += w;
          }

        if (area > 0.0)
          {
            const double inv_a = 1.0 / area;
            vm_cell[idx]  = static_cast<float>(vm_sum * inv_a);
            hd_cell[idx]  = static_cast<float>(hd_sum * inv_a);
            oct_cell[idx] = static_cast<float>(oct_sum * inv_a);
            mi_cell[idx]  = static_cast<float>(mi_sum * inv_a);
          }

        mat_cell[idx] = (cell->center()[1] < interface_y) ? 1.0f : 0.0f;
      }

    // ---- Output ----
    std::filesystem::create_directories("output/square_dealii");

    DataOut<2> data_out;

    const std::vector<std::string> names = {"ux", "uy"};
    const std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
      2, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(dof_handler, U, names, interp);
    data_out.add_data_vector(vm_cell, "von_mises", DataOut<2>::type_cell_data);
    data_out.add_data_vector(hd_cell, "hydrostatic", DataOut<2>::type_cell_data);
    data_out.add_data_vector(oct_cell, "oct_shear", DataOut<2>::type_cell_data);
    data_out.add_data_vector(mi_cell, "miner_index", DataOut<2>::type_cell_data);
    data_out.add_data_vector(mat_cell, "material", DataOut<2>::type_cell_data);

    data_out.build_patches();

    const std::string outfile = "output/square_dealii/square_dealii.vtu";
    std::ofstream     out(outfile);
    data_out.write_vtu(out);
    std::cout << "Output: " << outfile << "\n";
  }

} // namespace Square

int
main()
{
  try
    {
      Square::run(40); // 40x40 mesh
    }
  catch (std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << "\n";
      return 1;
    }
  return 0;
}
