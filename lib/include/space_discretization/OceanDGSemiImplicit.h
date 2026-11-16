/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version. The full text of the license can be found in the file
 * LICENSE.md at the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia, 2026
 *         Giuseppe Orlando, 2026
 */
#ifndef OCEANDGSEMIIMPLICIT_H
#define OCEANDGSEMIIMPLICIT_H

// The following files include the oceano libraries
#include <space_discretization/OceanDG.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>

/**
 * Namespace containing the spatial Operator
 */

namespace SpaceDiscretization
{

  using namespace dealii;

  using Number = double;



  // @sect3{The OceanoOperator with semi-implicit integrator}

  // This class implements the assembly of the evaluators for the ocean
  // problem using a semi-implicit scheme. The partitioning of the right-hand
  // side of the shallow water equations into stiff and non-stiff components
  // follows the requirement of using a large time step that is not restricted
  // by gravity wave speed or bottom friction coefficient. This is obtained by
  // treating semi-implicitly the mass flux term in the continuity equation,
  // the pressure gradient term and the bottom friction term in the momentum
  // equation.
  template <int dim, int n_tra, int degree, int n_points_1d>
  // Class OceanoOperatorSemiImplicit inherits from the base class OceanoOperator, which is a template class that takes:
  // the spatial dimension (dim), the number of tracers (n_tra), the polynomial degree (degree), and the number of quadrature points per spatial direction (n_points_1d) as template parameters.
  class OceanoOperatorSemiImplicit : public OceanoOperator<dim, n_tra, degree, n_points_1d>
  {
  public:
    double factor_matrix;

    OceanoOperatorSemiImplicit(
      IO::ParameterHandler      &param,
      ICBC::BcBase<dim, 1+dim+n_tra> *bc,
      TimerOutput               &timer_output,
      const unsigned int         max_iteration_height);
    ~OceanoOperatorSemiImplicit() = default;

    void local_apply_cell_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_stiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_pressure_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_face_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_face_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_face_stiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_stiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_inverse_modified_mass_matrix_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;
    
    void local_apply_linearized_cell_mass_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const LinearAlgebra::distributed::Vector<Number>              &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    // vmult method to work with GMRES solver, which is used to solve the linear system arising from the semi-implicit time integration scheme.
    void vmult(
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const LinearAlgebra::distributed::Vector<Number> &src) const;
    
    // Application of the mass matrix
    void vmult_mass_matrix(
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const LinearAlgebra::distributed::Vector<Number> &src) const;
    
    void
    perform_stage_hydro(
      const unsigned int                                             cur_stage,
      const Number                                                   cur_time,
      const Number                                                  *factor_residual,
      const Number                                                  *factor_tilde_residual,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &solution_height,
      LinearAlgebra::distributed::Vector<Number>                    &solution_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_height,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge) const;

    using Base = OceanoOperator<dim, n_tra, degree, n_points_1d>;
    using Base::bc;
    using Base::model;

    using Base::data_quadrature_cell_0;
    using Base::data_quadrature_cell_2;
    using Base::data_quadrature_face;
    using Base::data_quadrature_boundary;

  protected:
    using Base::data;
    using Base::timer;
    using Base::num_flux;
  };



  // The constructor simply initialize the base classes of the Ocean Operator.
  template <int dim, int n_tra, int degree, int n_points_1d>
  OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::OceanoOperatorSemiImplicit(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, 1+dim+n_tra>   *bc,
    TimerOutput                      &timer,
    const unsigned int                max_iteration_height)
    : OceanoOperator<dim, n_tra, degree, n_points_1d>(
      param, bc, timer, max_iteration_height)
  {}


  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::local_apply_linearized_cell_mass_height(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const LinearAlgebra::distributed::Vector<Number>              &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, degree + 2, 1, Number> phi_height(data, cell_range, 0, 1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q);
            // const auto zb_q = data_quadrature_cell_1.get_data(cell, q);

            phi_height.submit_value(
              z_q, // we should have + zb_q, but we are considering the linearized operator
                   // so the bathymetry goes into the right side
              q);
          }

        phi_height.integrate_scatter(EvaluationFlags::values,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  vmult(
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const LinearAlgebra::distributed::Vector<Number> &src
  ) const
  {
    // We apply the mass matrix: dst = M * src, where M is the mass matrix associated with the free-surface height equation.
    vmult_mass_matrix(dst, src);
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  vmult_mass_matrix(
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const LinearAlgebra::distributed::Vector<Number> &src
  ) const
  {
    // solution_height.zero_out_ghost_values();
    data.cell_loop(
        &OceanoOperatorSemiImplicit::local_apply_linearized_cell_mass_height,
        this,
        dst,
        src,
        true);
  }

  // For an implicit-explicit (IMEX) scheme, the right-hand side must be
  // partitioned into a stiff part and a non-stiff part. The stiff part
  // appears only in the momentum equation, namely the bottom friction term;
  // the non-stiff part is what is left. Since the bottom friction term is a
  // local operator, only a local cell-wise apply is needed for the stiff
  // part.
  // For the continuity equation, we copies the apply the cell apply
  // function of the base class, since nothing changes there. Neverthless we
  // have to reimplement it, to be used in the cell_loop.

  // CONTINUITY EQUATION - MASS FLUX
  // Computes the volume contribution of the discrete divergence operator B.
  // When applied to the known intermediate discharge Q*, it contributes
  // to the right-hand side F of the free-surface equation.
  // Mathematically, the volume contribution corresponds to:
  // integral_K q_h · grad(psi_i) dOmega,
  // where psi_i is a test function for the free surface.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_height(
    const MatrixFree<dim, Number> &, // information regarding the matrix-free data structure, via a dealii::MatrixFree object
    LinearAlgebra::distributed::Vector<Number>                    &dst, // dst = destination vector, where the result of the operator is stored,
                                                                        // is a distributed vector, which is a vector that is distributed across multiple processors in a parallel computing environment
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src, // vector of distributed vectors, which contains the input data for the operator, in this case the discharge vector Q*
                                                                        // for example here we use src[1] to access the discharge vector Q*.
                                                                        // but src[0] is not used, because the height equation does not depend on the height vector H.
                                                                        // so src contains the prognostic variables, non necessarily evaluated at the current time step, but at the time step of the Runge-Kutta stage.
    const std::pair<unsigned int, unsigned int>                   &cell_range) const // std::pair defines the range of cell batches to process:
                                                                                     // first is the starting index (included), second is the ending index (excluded).
  {
    // FEEvaluation is a class that provides the evaluation of finite element functions at quadrature points.
    // FEEvaluation<dim, fe_degree, n_points_1d, n_components, Number>
    //   dim          : spatial dimension of the problem
    //   fe_degree    : polynomial degree (-1 means determined at runtime), indeed we choose the poly degree at runtime.
    //   n_points_1d  : number of quadrature points per spatial direction
    //   n_components : number of components (1 for scalar, dim for vector)
    //   Number       : numerical type (e.g., double)
    //
    // Constructor (data, cell_range, dof_index):
    //   data         : MatrixFree object containing FE and geometric information
    //   cell_range   : range of cell batches to process
    //   dof_index    : index of the DoFHandler registered in MatrixFree

    // Here we create two FEEvaluation objects, one for the height (scalar) and one for the discharge (vector).
    // Indeed, note the difference in the n_components template parameter: 1 for the height and dim for the discharge.
    // Notice also the dof_index parameter in the constructor: 0 for the height and 1 for the discharge,
    // which correspond to the different DoFHandlers registered in the MatrixFree object.
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1);
    
    // Now we loop over the cell batches in the specified range. For each cell batch, we reinitialize the FEEvaluation objects
    // and gather the discharge values from the src vector.
    // Gathering means to gather the local DoF values from src[1] (discharge vector)
    // and evaluate the discrete discharge at quadrature points.
    // This is a local FE operation, not an MPI_Gather.
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        // Reinitializing is the equivalent of selecting the current cell batch for the FEEvaluation objects.
        phi_height.reinit(cell); // reinitialize the height evaluator for the current cell batch
        phi_discharge.reinit(cell); // reinitialize the discharge evaluator for the current cell batch
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values); // actual evaluation of the discharge at quadrature points, gathering from src[1] (discharge vector)
        
        
        // Here we loop over the quadrature points of the cell batch
        // and compute the mass flux at each quadrature point.
        // The order of operations is:
        // 1. Get the discharge value using phi_discharge.get_value(q).
        // 2. Compute the mass flux using model.mass_flux(), which takes
        //    the discharge as input (and returns the discharge itself).
        // 3. Submit the mass flux to phi_height using submit_gradient().
        //    This contribution will be integrated later by integrate_scatter().
        //
        // Since we are working with the continuity equation, the test
        // functions belong to the free-surface space. Therefore, we use
        // phi_height.submit_gradient() to integrate against their gradients.
        //
        // Mathematically, the volume contribution corresponds to:
        // integral_K q_h · grad(psi_i) dOmega,
        // where psi_i is a test function for the free surface.

        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          phi_height.submit_gradient(
            model.mass_flux(phi_discharge.get_value(q)),
            q);
        
        // integrate_scatter() performs two operations:
        //
        // 1. integrate:
        //    Computes the local integral over the current cell (or cell batch)
        //    using the submitted values, quadrature weights and Jacobian.
        //
        // 2. scatter:
        //    Accumulates the local contributions into the destination
        //    vector dst, at the corresponding DoF indices.
        //
        // The summation over all cells is managed by the external data.loop(),
        // which calls this method for each cell batch.
        //
        // In DG, DoFs are local to each cell, so contributions from
        // different cells do not necessarily share the same DoF indices.


        // A Degree of Freedom (DoF) is an independent coefficient of the
        // finite element approximation, not necessarily a mesh node.
        //
        // In nodal finite elements, DoFs represent solution values at nodes.
        // In other bases, they represent coefficients of basis functions.
        //
        // In DG, each cell has its own independent DoFs. Therefore, two
        // adjacent cells can share the same geometric point but have
        // different DoF indices and different solution values.
        // This allows the numerical solution to be discontinuous across cells.

        phi_height.integrate_scatter(EvaluationFlags::gradients,
                                       dst);
      }
  }

  // MOMENTUM EQUATION - NON-STIFF CONTRIBUTIONS
  // Computes the volume contributions of the explicit (non-stiff)
  // part of the momentum equation, associated with the operator G.
  // In particular, it includes the advective and diffusive fluxes,
  // while pressure and bottom friction are treated separately.
  //
  // The free-surface height zeta is used to reconstruct the water
  // depth h, while the discharge q and the velocity gradient
  // are used to compute the advective and diffusive fluxes.
  //
  // Mathematically, the advective volume contribution corresponds to:
  // integral_K (q_h tensor u_h) : grad(varphi_i) dOmega,
  // where varphi_i is a vector-valued test function for the discharge.
  //
  // The complete volume contribution may also include diffusion
  // and non-stiff source terms.

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_nonstiff_discharge(
    const MatrixFree<dim, Number> &, // information regarding the matrix-free data structure, via a dealii::MatrixFree object
    LinearAlgebra::distributed::Vector<Number>                    &dst, // dst = destination vector, where the result of the operator is stored,
                                                                        // is a distributed vector, which is a vector that is distributed across multiple processors in a parallel computing environment
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src, // vector of distributed vectors, which contains the input data for the operator,
                                                                        // check the comment in local_apply_cell_height for more details.
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0); // FEEvaluation for the free-surface height (scalar)
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1); // FEEvaluation for the discharge (vector)
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_velocity(data,cell_range,1); // FEEvaluation for the velocity (vector), which is computed from the discharge and depth
                                                                                     // since is not a prognostic variable, we do not have a DoFHandler for it, but we can compute it from the prognostic variables.
    // Compute 1/p^2, where p is the polynomial degree.
    // Use zero for p = 0 to avoid division by zero.
    // This factor can be used to scale numerical coefficients depending
    // on the polynomial degree, e.g., diffusion or stabilization parameters.
    const auto inv_degree = degree > 0 ? 1./(degree*degree) : 0.; // the ? operator is a ternary operator, which is a shorthand for an if-else statement.
                                                                  // If degree > 0, then inv_degree = 1/(degree^2), otherwise inv_degree = 0.

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) // loop over the cell batches in the specified range
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values); // gather the free-surface height values from src[0] (height vector) and evaluate at quadrature points
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values); // gather the discharge values from src[1] (discharge vector) and evaluate at quadrature points
        phi_velocity.reinit(cell);
        phi_velocity.gather_evaluate(src.back(), EvaluationFlags::gradients); // gather the velocity gradients from src.back() (velocity vector) and evaluate at quadrature points
                                                                              // src.back() accesses the last element of the src vector
        VectorizedArray<Number> area_cell; // declare a vectorized array to store the cell areas for each quadrature point in the current cell batch
        // data is a MatrixFree object that contains information about the mesh and the finite element spaces.
        // data.n_active_entries_per_cell_batch(cell) returns the number of active cells in the current SIMD batch, excluding unused entries.
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
          area_cell[v] = inv_degree * data.get_cell_iterator(cell,v)->measure(); // data.get_cell_iterator(cell,v) returns an iterator to the v-th active cell in the current SIMD batch,
                                                                                 // and measure() returns its area (or volume in 3D).
        // Now we loop over the quadrature points of the cell batch and compute the non-stiff contributions to the momentum equation.
        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q); // free-surface
            const auto q_q = phi_discharge.get_value(q); // discharge
            const auto du_q = phi_velocity.get_gradient(q); // velocity gradient for the viscous flux computation
            const auto point_q = phi_discharge.quadrature_point(q); // quadrature_point(q)returns the physical coordinates of the
                                                                    // q-th quadrature point in the current cell batch, which is 
                                                                    // needed to evaluate the problem data (e.g., wind stress, Coriolis force) at that point.

            const auto zb_q = data_quadrature_cell_0.get_data(cell, q)[0]; // we get the bathymetry value at the q-th quadrature point in the current cell batch
                                                                           // from the data_quadrature_cell_0 object, which stores the precomputed bathymetry values at quadrature points for each cell batch.
            const auto data_onthefly_q =
              evaluate_function<dim, Number, dim+1>(*bc->problem_data, point_q, 2); // we evaluate the external problem data (e.g., wind stress,Coriolis force)
		        // The operations performed here are:
            // 1. Compute the water depth h from the free-surface height zeta and bathymetry zb.
            // 2. Compute the advective and diffusive fluxes using the discharge q,
            //    the velocity gradient du, and the water depth h.
            // 3. Submit the computed fluxes to phi_discharge for integration, i.e multiply by the test functions and integrate over the cell.
            phi_discharge.submit_gradient(
              model.advective_diffusive_flux(
                z_q, q_q, model.depth(z_q, zb_q)*du_q, zb_q, area_cell),
              q);
            // Here we also compute the non-stiff source term, which includes the wind stress and Coriolis force contributions.
            // The source term is computed using the discharge q, the water depth h, and the external problem data (e.g., wind stress, Coriolis force) evaluated at the quadrature point.
            // Pressure and friction are not included in the non-stiff source term, since they are treated separately in the stiff part of the operator.
            // submit_value() is used to submit the computed source term to phi_discharge for integration, i.e multiply by the test functions and integrate over the cell.
            phi_discharge.submit_value(
              model.source_no_pressure_no_friction(q_q, data_onthefly_q),
              q);
          }
        // Finally, as noted in the previous function, integrate_scatter() performs two operations: integration and scattering.
        // Notice that in this case we are integrating both the values and the gradients, since we have submitted both contributions to phi_discharge.
        // The symbol | is an OR operator (not the logical OR operator), and it combines both the integrals over values and gradients into a single evaluation flag,
        // so it computes the sum of the two integral contributions and scatters them into the destination vector dst.
        phi_discharge.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                       dst);
      }
  }

  // MOMENTUM EQUATION - STIFF CONTRIBUTIONS
  // Computes the cell-volume contribution of the stiff operator L,
  // including the pressure-gradient and bottom-friction source terms.
  //
  // The free-surface elevation zeta, its gradient, and the discharge q
  // are evaluated at quadrature points, together with the bathymetry
  // and friction coefficient.
  //
  // Mathematically, the volume contribution corresponds to:
  // integral_K S_stiff(zeta_h, q_h, grad(zeta_h)) · varphi_i dOmega,
  // where varphi_i is a vector-valued test function for the discharge.
  //
  // The resulting contributions are integrated and scattered into dst.

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_stiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        // In this operator we need both the values and the gradients of the free-surface height zeta, since they are used in the pressure-gradient term.
        phi_height.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);
        // Now we loop over the quadrature points of the cell batch and compute the stiff contributions to the momentum equation.
        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q); // free-sruface height zeta at the q-th quadrature point
            const auto dz_q = phi_height.get_gradient(q); // gradient of the free-surface height zeta at the q-th quadrature point
            const auto q_q = phi_discharge.get_value(q); // discharge q at the q-th quadrature point

            const auto zb_q = data_quadrature_cell_0.get_data(cell, q)[0]; // bathymetry at the q-th quadrature point
            const auto cf_q = data_quadrature_cell_0.get_data(cell, q)[1]; // friction coefficient at the q-th quadrature point
            // The operations performed here are:
            // 1. Compute the pressure-gradient source term using model.source_pressure_and_friction(),
            //    which is g * h * grad(zeta) + bottom_friction(q, cf)
            // 2. Submit the computed source term to phi_discharge value for integration, i.e multiply by the test functions to be integrated over the cell.
            phi_discharge.submit_value(
              model.source_pressure_and_friction(z_q, q_q, dz_q, zb_q, cf_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values,
                                       dst);
      }
  }

  // MOMENTUM EQUATION - PRESSURE GRADIENT CONTRIBUTION
  // Computes the cell-volume contribution of the pressure operator E,
  // excluding the bottom-friction term.
  //
  // The free-surface elevation zeta and its gradient are evaluated
  // at quadrature points, together with the bathymetry, to reconstruct
  // the water depth h.
  //
  // Mathematically, the volume contribution corresponds to:
  // integral_K (-g * h * grad(zeta_h)) · varphi_i dOmega,
  // where varphi_i is a vector-valued test function for the discharge.
  //
  // The resulting contributions are integrated and scattered into dst.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_pressure_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        // In this operator we need both the values and the gradients of the free-surface height zeta, since they are used in the pressure-gradient term.
        phi_height.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_discharge.reinit(cell);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q); // free-surface height zeta at the q-th quadrature point
            const auto dz_q = phi_height.get_gradient(q); // gradient of the free-surface height zeta at the q-th quadrature point

            const auto zb_q = data_quadrature_cell_0.get_data(cell, q)[0]; // bathymetry at the q-th quadrature point
            // The operations performed here are:
            // 1. Compute the pressure-gradient source term using model.source_pressure(),
            //    which is g * h * grad(zeta)
            // 2. Submit the computed source term to phi_discharge value for integration, i.e multiply by the test functions to be integrated over the cell.

            
            phi_discharge.submit_value(
              model.source_pressure(z_q, dz_q, zb_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values,
                                       dst);
      }
  }

  // Within this class we also reimplement the face operators, which differ
  // from those of the base class. In fact, unlike the base class, we need
  // the numerical mass flux computed with a centered flux. The centered flux
  // is so simple that it is hard-coded in the body of the apply operator,
  // together with its hydrostatic reconstruction. Since the pressure term is
  // treated implicitly, we define both stiff and non-stiff face and boundary
  // operators: the non-stiff operator contains the numerical advective
  // fluxes, while the stiff one implements the pressure fluxes.

  // So here are implemented all the face operators, which are similar to those of the base class, but with some differences in the numerical fluxes.
  // In particular we can notice them as the sums over the faces EPS in the semi-discrete formulation of the shallow water equations.
  
  // CONTINUITY EQUATION - FACE CONTRIBUTIONS TO F
  // Computes the interior-face contribution to the known term F,
  // using the intermediate discharge Q*.
  //
  // The free-surface elevation and discharge are evaluated on both
  // sides of each face (minus and plus). The discharge is reconstructed
  // using the local velocities and a common bathymetry.
  //
  // Mathematically, the face contribution in Eq. (9b) is:
  // - integral_F ( {{Q*}} + (lambda/2) [[zeta*]] ) · [[psi_i]] dGamma,
  // where {{Q*}} denotes the average discharge and [[.]] the DG jump.
  //
  // IMPORTANT DIFFERENCE FROM THE PAPER:
  // The current implementation includes only the centered average
  // of the reconstructed discharge. The stabilization term
  // (lambda/2) [[zeta*]] appearing in Eq. (9b) is not implemented here.
  //
  // The numerical contribution is submitted with opposite signs
  // on the minus and plus sides, then integrated and scattered into dst.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_face_height(
    const MatrixFree<dim, Number> &, // information regarding the matrix-free data structure, via a dealii::MatrixFree object
    LinearAlgebra::distributed::Vector<Number>                    &dst, // dst = destination vector, where the result of the operator is stored,
                                                                        // is a distributed vector, which is a vector that is distributed across multiple processors in a parallel computing environment
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src, // vector of distributed vectors, which contains the input data for the operator,
                                                                        // check the comment in local_apply_cell_height for more details.
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    // FEFaceEvaluation evaluates finite element quantities at face quadrature points.
    // Unlike FEEvaluation, which operates on cell interiors, FEFaceEvaluation
    // operates on the faces shared by neighboring cells.
    //
    // Template parameters:
    // dim        : spatial dimension.
    // -1         : polynomial degree determined at runtime.
    // degree + 2 : number of quadrature points per face direction.
    // 1 or dim    : number of components (scalar height / vector discharge).
    // Number     : numerical type used for computations.
    //
    // Constructor arguments:
    // data       : MatrixFree object containing the finite element information.
    // face_range : range of faces to process.
    // true/false : interior (-) or exterior (+) side of the face.
    // 0 / 1      : DoFHandler index (0 = height, 1 = discharge).
    // 1          : quadrature formula index.
    //
    // The minus (-) and plus (+) evaluators provide the values and gradients
    // from both sides of an interface, needed to compute numerical fluxes.
    // phi_height_m = evaluator for the height on the interior side of the face, i.e. minus side
    // phi_height_p = evaluator for the height on the exterior side of the face, i.e. plus side
    // etc for the discharge.
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_m(data, face_range,
                                                                      true, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_p(data, face_range,
                                                                      false, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_m(data, face_range,
                                                                      true, 1, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_p(data, face_range,
                                                                      false, 1, 1);
    // Now we loop over the faces in the specified range
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height_p.reinit(face);
        phi_height_p.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the plus side of the face
        phi_discharge_p.reinit(face);
        phi_discharge_p.gather_evaluate(src[1], EvaluationFlags::values); // discharge on the plus side of the face

        phi_height_m.reinit(face);
        phi_height_m.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the minus side of the face
        phi_discharge_m.reinit(face);
        phi_discharge_m.gather_evaluate(src[1], EvaluationFlags::values); // discharge on the minus side of the face
        // Now we loop over the quadrature points of the face and compute the numerical fluxes for the height equation.
        for (unsigned int q = 0; q < phi_height_m.n_q_points; ++q)
          {
            const auto z_m    = phi_height_m.get_value(q); // free-surface height on the minus side of the face at the q-th quadrature point
            const auto z_p    = phi_height_p.get_value(q); // free-surface height on the plus side of the face at the q-th quadrature point
            const auto normal = phi_discharge_m.normal_vector(q); // normal vector at the q-th quadrature point, pointing from the minus side to the plus side of the face
            // The bathymetry values on both sides of the face are obtained from the precomputed data_quadrature_face object. And are stored
            // in an even-odd fashion, where the even indices correspond to the minus side and the odd indices to the plus side of the face.
            const auto zb_m   = data_quadrature_face.get_data(face, 2*q); // bathymetry on the minus side of the face at the q-th quadrature point (even index)
            const auto zb_p   = data_quadrature_face.get_data(face, 2*q+1); // bathymetry on the plus side of the face at the q-th quadrature point (odd index)
            // Computing the velocities on both sides using the model.velocity() function
            const auto v_m = model.velocity(z_m, phi_discharge_m.get_value(q), zb_m);
            const auto v_p = model.velocity(z_p, phi_discharge_p.get_value(q), zb_p);
            const auto zb_star = std::min(zb_m, zb_p); // taking the minimum bathymetry, to ensure well balancing
            // So zb_star is a reconstruction of the bathymetry at the interface, which is used to compute the water depth on both sides of the face.
            // Now we compute the numerical fluxes for the height equation using the reconstructed water depths and velocities on both sides of the face.
            // In the continuity equation, the mass flux is the discharge q = h * v, where h = max(height + bathymetry, 0) 
            // Here, we reconstruct the discharge on both sides of the face using
            // the original velocities and the water depths computed with zb_star.
            const auto flux_m = model.depth(z_m, zb_star) * v_m; // computing the reconstructed discharge on the minus side
            const auto flux_p = model.depth(z_p, zb_star) * v_p; // computing the reconstructed discharge on the plus side
            
            // Compute the average reconstructed discharge {{q}} and project it
            // onto the face normal to obtain the scalar normal flux.
            auto numerical_flux_p = 0.5 * (flux_m * normal + flux_p * normal); 
            // Account for the jump of the test function:
            // [[psi_i]] = psi_i_minus * n_minus + psi_i_plus * n_plus,
            // with n_plus = -n_minus.
            // So the opposite signs implement the term -{{q}} · [[psi_i]].
            phi_height_m.submit_value(-numerical_flux_p, q);
            phi_height_p.submit_value(numerical_flux_p, q);
          }

        phi_height_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_height_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // MOMENTUM EQUATION - NON-STIFF FACE CONTRIBUTIONS TO G
  // Computes the interior-face contribution of the non-stiff operator G,
  // associated with the numerical advective flux of the momentum equation.
  //
  // The free-surface elevation and discharge are evaluated on both sides
  // of each face (minus and plus), together with the local bathymetry.
  //
  // The modified Lax-Friedrichs (Rusanov) numerical flux consists of:
  // - The centered average of the advective momentum fluxes.
  // - A stabilization term proportional to lambda and the difference
  //   between the hydrostatically reconstructed discharges.
  //
  // Mathematically, the numerical flux is:
  // 0.5 * (F_minus + F_plus) * n + 0.5 * lambda * (q_minus - q_plus),
  // where lambda is the maximum characteristic wave speed across the face.
  //
  // The resulting numerical flux is submitted with opposite signs
  // on the minus and plus sides, then integrated and scattered into dst.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_face_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_m(data, face_range,
                                                                      true, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_p(data, face_range,
                                                                      false, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_m(data, face_range,
                                                                      true, 1, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_p(data, face_range,
                                                                      false, 1, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height_p.reinit(face);
        phi_height_p.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the plus side of the face
        phi_discharge_p.reinit(face);
        phi_discharge_p.gather_evaluate(src[1], EvaluationFlags::values); // discharge on the plus side of the face

        phi_height_m.reinit(face);
        phi_height_m.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the minus side of the face
        phi_discharge_m.reinit(face);
        phi_discharge_m.gather_evaluate(src[1], EvaluationFlags::values); // discharge on the minus side of the face
        // Now we loop over the quadrature points of the face and compute the numerical fluxes for the momentum equation.
        for (unsigned int q = 0; q < phi_discharge_m.n_q_points; ++q)
          {
            const auto z_m    = phi_height_m.get_value(q); // free-surface height on the minus side of the face at the q-th quadrature point
            const auto z_p    = phi_height_p.get_value(q); // free-surface height on the plus side of the face at the q-th quadrature point
            const auto normal = phi_discharge_m.normal_vector(q); // normal vector at the q-th quadrature point, pointing from the minus side to the plus side of the face
            const auto zb_m   = data_quadrature_face.get_data(face, 2*q); // bathymetry on the minus side of the face at the q-th quadrature point (even index)
            const auto zb_p   = data_quadrature_face.get_data(face, 2*q+1);  // bathymetry on the plus side of the face at the q-th quadrature point (odd index)
            // Compute the numerical flux for the momentum equation using the modified Lax-Friedrichs (Rusanov) flux.
            // hence computing: {{q^(*,(m))}} otimes v^{(m)} + lambda^{(m)}/2 * <<q^(*,(m))>> 
            auto numerical_flux_p =
              num_flux.numerical_advflux_weak(z_m,
                                              z_p,
                                              phi_discharge_m.get_value(q),
                                              phi_discharge_p.get_value(q),
                                              normal,
                                              zb_m,
                                              zb_p);
            // The numerical flux for the minus side is simply the negative of the plus side, since we are computing the jump across the face.
            // Everything gets then submitted to the respective FEFaceEvaluation object of discharge for integration and scattering into the destination vector dst.
            phi_discharge_m.submit_value(-numerical_flux_p, q);
            phi_discharge_p.submit_value(numerical_flux_p, q);
          }

        phi_discharge_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_discharge_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // MOMENTUM EQUATION - STIFF FACE CONTRIBUTIONS
  // Computes the interior-face contribution of the stiff operator L,
  // associated with the pressure term in the momentum equation.
  //
  // The free-surface elevation is evaluated on both sides of each face
  // (minus and plus), together with the corresponding bathymetry.
  //
  // The numerical pressure flux is computed using
  // numerical_presflux_strong(), separately for the minus and plus sides,
  // with reversed input states and bathymetry on the plus side.
  //
  // Unlike the cell-volume stiff contribution, this face operator
  // contains only pressure terms; bottom friction is not included.
  //
  // The numerical fluxes are submitted to the discharge test functions
  // on both sides, then integrated and scattered into dst.
  //
  // This face operator is also reused in the pressure correction step
  // following the solution of the free-surface equation.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_face_stiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_m(data, face_range,
                                                                      true, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_p(data, face_range,
                                                                      false, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_m(data, face_range,
                                                                      true, 1, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_p(data, face_range,
                                                                      false, 1, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height_p.reinit(face);
        phi_height_p.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the plus side of the face
        phi_discharge_p.reinit(face);

        phi_height_m.reinit(face);
        phi_height_m.gather_evaluate(src[0], EvaluationFlags::values); // free-surface height on the minus side of the face
        phi_discharge_m.reinit(face);

        for (unsigned int q = 0; q < phi_discharge_m.n_q_points; ++q)
          {
            const auto z_m    = phi_height_m.get_value(q); // free-surface height on the minus side of the face at q-th quadrature point
            const auto z_p    = phi_height_p.get_value(q); // free-surface height on the plus side of the face at q-th quadrature point
            const auto normal = phi_discharge_m.normal_vector(q); // normal vector at the interface pointing from plus face to minus face
            const auto zb_m   = data_quadrature_face.get_data(face, 2*q); // bathymetry on the minus side of the face at the q-th quadrature point (even index)
            const auto zb_p   = data_quadrature_face.get_data(face, 2*q+1); // bathymetry on the plus side of the face at the q-th quadrature point (odd index)
            // The function numerical_presflux_strong, computes the numerical pressure flux corresponding to
            // g * {{h*}} * {{phi_i}} · [[zeta*]], excluding the test-function
            // contribution, which is handled by submit_value().
            auto numerical_flux_m =
              -num_flux.numerical_presflux_strong(z_m,
                                                 z_p,
                                                 normal,
                                                 zb_m,
                                                 zb_p); // numerical pressure flux on the minus side of the face
            auto numerical_flux_p =
               num_flux.numerical_presflux_strong(z_p,
                                                 z_m,
                                                 normal,
                                                 zb_p,
                                                 zb_m); // numerical pressure flux on the plus side of the face
            // The numerical flux for the minus side is simply the negative of the plus side, since we are computing the jump across the face.
            // Everything gets then submitted to the respective FEFaceEvaluation object of discharge for integration and scattering into the destination vector dst.
            phi_discharge_m.submit_value(numerical_flux_m, q);
            phi_discharge_p.submit_value(numerical_flux_p, q);
          }

        phi_discharge_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_discharge_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // Similar argument applies to the boundary operators, which must be
  // reimplemented.


  // CONTINUITY EQUATION - BOUNDARY FACE CONTRIBUTIONS
  // Computes the boundary-face contribution of the discrete divergence
  // operator B, accounting for the numerical mass flux across the
  // external boundaries of the computational domain.
  //
  // Unlike interior faces, where the minus and plus states belong to
  // two neighboring cells, only the interior (minus) state is directly
  // available. The exterior (plus) state is reconstructed according
  // to the prescribed boundary condition.
  //
  // The numerical mass flux is computed using the interior and
  // reconstructed exterior states, together with the bathymetry.
  //
  // Mathematically, the boundary contribution corresponds to:
  // - integral_boundaryK (flux · n) * psi_i dGamma,
  // where psi_i is a scalar test function for the free surface.
  //
  // The resulting contributions are integrated and scattered into dst.
  // When applied to the known intermediate discharge Q*, this operator
  // contributes to the right-hand side F of the free-surface equation.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_boundary_face_height(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    // We are at the boundary so only the interior face value is known, hence we evaluate it (true = interior face)
    // The exterior (plus) state is obtained from the boundary condition.
    FEFaceEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data, face_range, true, 0);
    FEFaceEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data, face_range, true, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values); // free-surface in the interior face (minus side)
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values); // discharge in the interior face (minus side)

        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q); // free-surface in the interior face (minus side) evaluated at q-th point of quad
            const auto q_m    = phi_discharge.get_value(q); // discharge in the interior face (minus side) evaluated at q-th point of quad
            const auto normal = phi_height.normal_vector(q); // normal vector at the interface pointing from plus face to minus face
            const auto point  = phi_height.quadrature_point(q); // returns the q-th quadrature point on the face in real coordinates stored in MappingInfo
            // Bathymetry at the boundary quadrature point; subtract n_inner_face_batches()
            // to convert the global face-batch index into a boundary-local index.
            const auto zb_m   =
              data_quadrature_boundary.get_data(face-data.n_inner_face_batches(), q);
            // declaring the values to be constructed in the "ghost" plus side of the face
            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            bool at_outflow;
            
            const auto boundary_id = data.get_boundary_id(face); // get the corresponding boundary_id of the face, to decide the applied boundary cond
            // Construct the exterior (plus) state from the prescribed boundary condition,
            // using the interior state, outward normal, position, and bathymetry.
            Base::get_boundary_value(
              boundary_id, z_m, q_m, normal, point, zb_m, &z_p, q_p, &at_outflow);
            // Compute velocities on both sides of the boundary face (interior and reconstructed exterior).
            // Reconstruct the discharges using the common bathymetry zb_star = zb_m.
            // Recall that the mass flux is F(q) = q = h * v.
            const auto v_m = model.velocity(z_m, q_m, zb_m);
            const auto v_p = model.velocity(z_p, q_p, zb_m);
            const auto zb_star = std::min(zb_m, zb_m);
            const auto flux_m = model.depth(z_m, zb_star) * v_m;
            const auto flux_p = model.depth(z_p, zb_star) * v_p;
            // Compute the centered numerical mass flux projected onto the outward normal.
            // The minus sign comes from the boundary term of the weak continuity equation,
            // after moving the spatial contributions to the right-hand side.
            auto flux = 0.5 * (flux_m * normal + flux_p * normal);

            phi_height.submit_value(-flux, q);
          }

        phi_height.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // MOMENTUM EQUATION - NON-STIFF BOUNDARY FACE CONTRIBUTIONS
  // Computes the boundary-face contribution of the non-stiff operator G,
  // associated with the numerical advective momentum flux.
  //
  // The free-surface elevation and discharge are evaluated on the interior
  // (minus) side, while the exterior (plus) state is reconstructed using
  // the prescribed boundary condition.
  //
  // The numerical advective flux is computed using the local
  // Lax-Friedrichs (Rusanov) formulation with hydrostatic reconstruction.
  //
  // Mathematically, the boundary contribution corresponds to:
  // - integral_boundaryK F_adv_num · varphi_i dGamma,
  // where F_adv_num is the numerical advective flux projected onto
  // the outward normal, and varphi_i is the vector-valued discharge test function.
  //
  // The resulting contributions are integrated and scattered into dst.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_boundary_face_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    // Only the interior (minus) side is evaluated (true).
    // The exterior (plus) state is obtained from the boundary condition.
    FEFaceEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data, face_range, true, 0);
    FEFaceEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data, face_range, true, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q);
            const auto q_m    = phi_discharge.get_value(q);
            const auto normal = phi_discharge.normal_vector(q);
            const auto point  = phi_height.quadrature_point(q);
            const auto zb_m   =
              data_quadrature_boundary.get_data(face-data.n_inner_face_batches(), q);

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            bool at_outflow;

            const auto boundary_id = data.get_boundary_id(face);
            // Construct the exterior (plus) state from the prescribed boundary condition,
            // using the interior state, outward normal, position, and bathymetry.
            Base::get_boundary_value(
              boundary_id, z_m, q_m, normal, point, zb_m, &z_p, q_p, &at_outflow);
            // Compute the numerical advective momentum flux using the interior
            // and reconstructed exterior states, with the same bathymetry on both sides.
            // (advective flux for the momentum eq + Rusanov stabilization)
            auto flux =
              num_flux.numerical_advflux_weak(z_m, z_p, q_m, q_p, normal, zb_m, zb_m);

            // OUTFLOW BOUNDARY CORRECTION
            // If the boundary is classified as outflow, suppress the numerical
            // advective flux in SIMD lanes where the interior discharge points inward
            // (q_m · n < -1e-12), preventing an incoming advective contribution.
            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  auto rho_u_dot_n = q_m * normal; // computes the discharge along the outward normal
                  // if the discharge along the outward normal is negative (< 0) then it means that it's going INSIDE the domain
                  // so we nullify the flux
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d][v] = 0.;
                }
            // The minus sign comes from the boundary term of the weak continuity equation,
            // after moving the spatial contributions to the right-hand side.
            phi_discharge.submit_value(-flux, q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // MOMENTUM EQUATION - STIFF BOUNDARY FACE CONTRIBUTIONS
  // Computes the boundary-face contribution of the stiff operator L,
  // associated with the numerical pressure flux in the momentum equation.
  //
  // The free-surface elevation and discharge are evaluated on the interior
  // (minus) side, while the exterior (plus) state is reconstructed using
  // the prescribed boundary condition.
  //
  // The numerical pressure flux is computed using
  // numerical_presflux_strong() and the boundary bathymetry.
  //
  // Mathematically, the boundary contribution corresponds to:
  // - integral_boundaryK F_pres_num · varphi_i dGamma,
  // where F_pres_num is the numerical pressure flux and
  // varphi_i is the vector-valued discharge test function.
  //
  // Bottom friction is not included in this boundary contribution.
  // The resulting contributions are integrated and scattered into dst.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_boundary_face_stiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    // Only the interior (minus) side is evaluated (true).
    // The exterior (plus) state is obtained from the boundary condition.
    FEFaceEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data, face_range, true, 0);
    FEFaceEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data, face_range, true, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q);
            const auto q_m    = phi_discharge.get_value(q);
            const auto normal = phi_discharge.normal_vector(q);
            const auto point  = phi_height.quadrature_point(q);
            const auto zb_m   =
              data_quadrature_boundary.get_data(face-data.n_inner_face_batches(), q);

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            bool at_outflow;

            const auto boundary_id = data.get_boundary_id(face);
            // Construct the exterior (plus) state from the prescribed boundary condition,
            // using the interior state, outward normal, position, and bathymetry.
            Base::get_boundary_value(
              boundary_id, z_m, q_m, normal, point, zb_m, &z_p, q_p, &at_outflow);
            // Compute the numerical pressure flux using the interior and
            // reconstructed exterior free-surface elevations.
            // The same boundary bathymetry is used on both sides.
            auto flux =
              num_flux.numerical_presflux_strong(z_m, z_p, normal, zb_m, zb_m);
            // OUTFLOW BOUNDARY CORRECTION
            // If the boundary is classified as outflow, suppress the numerical
            // advective flux in SIMD lanes where the interior discharge points inward
            // (q_m · n < -1e-12), preventing an incoming advective contribution.
            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  auto rho_u_dot_n = q_m * normal;
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d][v] = 0.;
                }
            // Submit the negative numerical pressure flux, consistently with
            // the boundary contribution of the stiff momentum equation.
            phi_discharge.submit_value(-flux, q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // Implements the inverse mass matrix for the discharge equation. The
  // mass matrix includes a friction term, which is handled implicitly and this
  // why we named "modified". As in the explicit case, we use the fast inversion.
  // An important difference here is that, due to the presence of the non-polynomial
  // friction term, the integration cannot be exact. We assume that the number of
  // quadrature points used for the fast inversion, equal to the number of degrees
  // of freedom, is accurate enough.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_inverse_modified_mass_matrix_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    // phi_discharge: discharge field to which A^(-1) is applied to compute Q* at stage l.
    // phi_discharge_ri: reference discharge used to evaluate the coefficients of A^(-1).
    // phi_height_ri: reference free-surface elevation used to evaluate the coefficients of A^(-1).
    FEEvaluation<dim, -1, degree + 1, dim, Number> phi_discharge(data, cell_range, 1, 2);
    FEEvaluation<dim, -1, degree + 1, 1, Number> phi_height_ri(data, cell_range, 0, 2);
    FEEvaluation<dim, -1, degree + 1, dim, Number> phi_discharge_ri(data, cell_range, 1, 2);
    // INVERSE MODIFIED MASS MATRIX - DISCHARGE
    // Define two operators to apply the inverse of the local mass matrix.
    //
    // inverse: standard deal.II operator, applying the full cellwise inverse
    // mass matrix to the discharge DoFs.
    //
    // inverse_dry: OCEANO-specific lumped approximation, using only the
    // diagonal entries for efficient SIMD operations on dry cells.
    //
    // Both operators are initialized with phi_discharge and will be used
    // to apply A^(-1) in the semi-implicit momentum equation.
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim, Number>
      inverse(phi_discharge);
    MatrixFreeOperatorsOceano::CellwiseInverseMassMatrixLumped<dim, 0, dim, Number>
      inverse_dry(phi_discharge);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        // read_dof_values() reads the coefficients q_j (DoFs) directly.
        // gather_evaluate() reconstructs the field at quadrature points:
        // q_h(x_q) = sum_j q_j * phi_j(x_q).
        // Here, src[0] provides the DoFs for A^(-1), while src[1] and src[2]
        // provide the reference fields needed to evaluate its coefficients.
        phi_discharge.reinit(cell);
        phi_discharge.read_dof_values(src[0]);

        phi_height_ri.reinit(cell);
        phi_height_ri.gather_evaluate(src[1], EvaluationFlags::values);
        phi_discharge_ri.reinit(cell);
        phi_discharge_ri.gather_evaluate(src[2], EvaluationFlags::values);
        // Select the inverse mass-matrix method based on the active FE index.
        // Index 1: standard polynomial degree, using the full cellwise inverse.
        // Index 0: degree-zero element, typically used for dry cells in the hp strategy,
        //          using the lumped inverse mass matrix (exact for degree zero).
        // Note: FE index 0 identifies the polynomial degree, not necessarily a dry cell.
        if (phi_discharge.get_active_fe_index())
          {
            // deal.II performs FE computations on a reference cell rather than directly
            // on each physical cell, allowing the same basis functions and quadrature
            // rules to be reused for cells with different geometries.
            //
            // The geometric Jacobian J maps the reference cell to the physical cell,
            // and |det(J)| accounts for the change in volume during integration.
            //
            // The inverse JxW = 1 / (|det(J)| * w_q), where w_q is the quadrature weight,
            // is used to apply the local inverse mass matrix.
            AlignedVector<VectorizedArray<Number>> inverse_jxw(phi_discharge.n_q_points);
	    inverse.fill_inverse_JxW_values(inverse_jxw);

            for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
              {
                const auto z_q = phi_height_ri.get_value(q); // Reference free-surface elevation zeta, used to compute the reference water depth h.
                const auto q_q = phi_discharge_ri.get_value(q); // Reference discharge q, used to compute the reference velocity.
                const auto zb_q = data_quadrature_cell_2.get_data(cell, q)[0]; // Bathymetry at the q-th quadrature point.
                const auto cf_q = data_quadrature_cell_2.get_data(cell, q)[1]; // Bottom-friction (drag) coefficient at the q-th quadrature point.
                // Modify the inverse JxW coefficients by the reciprocal of the
                // friction-dependent factor defining A, evaluated at the reference state.
                // These modified coefficients are then used to apply A^(-1).

                // Compute the analytically defined Jacobian of the bottom-friction term,
                // evaluated at the reference velocity v = q/h and water depth h.
                // velocity() reconstructs v from zeta, q, and bathymetry,
                // while depth() computes h = zeta + zb (with a minimum depth threshold).
                
                // Modify inverse_jxw by the reciprocal of the coefficient defining A^(l):
                // 1 / (1 + a_tilde_ll * dt * gamma(q_ref, h_ref)).
                //
                // Here, factor_matrix = a_tilde_ll * dt, while bottom_friction.jacobian()
                // evaluates gamma at the reference state.
                //
                // The resulting coefficient is:
                // inverse_jxw[q] = 1 / (JxW_q * (1 + a_tilde_ll * dt * gamma)).
                //
                // These coefficients are then used by inverse.apply() to apply A^(-1).
                inverse_jxw[q] *= 1. / ( 1. + factor_matrix
                  * model.bottom_friction.jacobian(model.velocity(z_q, q_q, zb_q),
                                                   cf_q,
                                                   model.depth(z_q, zb_q))
                                       );
              }
            // The basis functions do not appear explicitly because the standard
            // deal.II CellwiseInverseMassMatrix uses the FE shape information
            // stored in phi_discharge to apply the inverse mass matrix.
            // It internally accounts for the products phi_i * phi_j, while
            // inverse_jxw provides the inverse geometric and friction coefficients.  

            // Now we apply the local inverse of the modified mass matrix A^(l) to the discharge DoFs.
            //
            // inverse: deal.II CellwiseInverseMassMatrix object, initialized with
            //          phi_discharge to perform the local matrix inversion.
            //
            // inverse_jxw: contains 1 / (JxW_q * (1 + factor_matrix * gamma_q)),
            //              accounting for cell geometry and bottom friction.
            //
            // dim: number of discharge components.
            //
            // phi_discharge.begin_dof_values(): pointer to the local discharge
            //          coefficients q_j, previously loaded using read_dof_values().
            //
            // The first pointer provides the input DoFs, while the second stores
            // the result A^(-1) * src[0] in the same array (in-place operation).
            //
            // This branch uses the standard inverse mass matrix, not the lumped
            // approximation used for degree-zero cells.
            inverse.apply(inverse_jxw, dim, phi_discharge.begin_dof_values(),
              phi_discharge.begin_dof_values());
          }
        else // Index 0: degree-zero element, typically used for dry cells in the hp strategy,
             // using the lumped inverse mass matrix (exact for degree zero).
          {
            for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
              {
                // Same as the non lumped case, we get the values of zeta, discharge, bathymetry and friction coeff
                const auto z_q = phi_height_ri.get_value(q);
                const auto q_q = phi_discharge_ri.get_value(q);
                const auto zb_q = data_quadrature_cell_2.get_data(cell, q)[0];
                const auto cf_q = data_quadrature_cell_2.get_data(cell, q)[1];

                // For the degree-zero element, assemble the single entry of the
                // modified mass matrix A_00 = integral_K (1 + alpha * gamma) dOmega.
                //
                // submit_value() and integrate() compute the integral against the
                // height basis function psi_0 = 1. Since phi_0 * phi_0 = 1,
                // this gives the exact 1x1 mass matrix, without assembling a full matrix.
                //
                // The lumped inverse is therefore exact for this degree-zero element.
                phi_height_ri.submit_value(1. + factor_matrix
                    * model.bottom_friction.jacobian(model.velocity(z_q, q_q, zb_q),
                                                     cf_q,
                                                     model.depth(z_q, zb_q)),
                                              q);
              }
            phi_height_ri.integrate(EvaluationFlags::values);
            // We create an array of dimension 1, since we have only one DoF for each cell, i.e. the matrix is 1x1
            AlignedVector<VectorizedArray<Number>> cell_matrix(1);
            // Extract the DoF value from phi_height_ri, recall that the DoF of phi_height_ri now is:
            // A_00 = integral_K (1 + alpha * gamma) dOmega
            cell_matrix[0] = phi_height_ri.get_dof_value(0);
            // Apply the lumped inverse of the degree-zero modified mass matrix A.
            // The single coefficient A_00 is provided by cell_matrix[0].
            // Each discharge DoF is divided by A_00, directly in SIMD format.
            // The same DoF array is used for input and output (in-place operation). 
            inverse_dry.apply(&cell_matrix[0], phi_discharge.begin_dof_values(),
              phi_discharge.begin_dof_values());
          }
        // Now we write in the global dsetination vector (dst) the local DoF vaalues of the discharge
        // containing the results of the application of the inverse A^(-1)
        phi_discharge.set_dof_values(dst);
      }
  }



  // The partitioning of the right-hand side of the shallow water equations into
  // stiff and non-stiff components follows the requirement of using a large time
  // step that is not restricted by gravity wave speed or bottom friction coefficient.
  // This is obtained by treating semi-implicitly the mass flux term in the
  // continuity equation, the pressure gradient term and the bottom friction term
  // in the momentum equation.
  //
  // The hydrodynamic system is solved using the Schur complement tecnique
  // which is accomplished in three steps: 1/ an explicit update of the
  // discharge 2/ the solution of the discrete wave equation. 3/ the correction
  // of the discharge.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::perform_stage_hydro(
    const unsigned int                                             current_stage, // Index of the current IMEX RK stage (l)
    const Number                                                   current_time, // Physical time associated with the current stage
    const Number                                                  *factor_residual, // Explicit RK coefficients for combining non-stiff residuals (a,b,c)
    const Number                                                  *factor_tilde_residual, // Implicit RK coefficients for combining stiff residuals (a-tilde,b-tilde,c-tilde)
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri, // Know hydrodynamic state (zeta and discharge) used as input
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height, // Stage residual vectors for the free-surface equation. (later combined with b/b-tilde coefficients)
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge, // Stage residual vectors for the discharge equation. (later combined with b/b-tilde coefficients)
    LinearAlgebra::distributed::Vector<Number>                    &solution_height, // Free-surface solution vector.
    LinearAlgebra::distributed::Vector<Number>                    &solution_discharge, // Discharge solution vector.
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_height, // Output free-surface state for the current stage.
                                                                  // Output discharge state for the current stage.
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge ) const // const here indicates that the method cannot change members of the object
                                                                                             // aside from those passed as reference (&)
  {
    // vec_ki is a vector of distributed vectors, so vec_ki_height.size is referring to the number of vectors contained
    const unsigned int n_stages = vec_ki_height.size()-1; // -1 because the code consider one stage more
    // Check whether the current RK stage requires an implicit solve,
    // i.e., whether the diagonal implicit coefficient dt * a_tilde_ll is nonzero.
    const bool is_last_stage_implicit = factor_matrix > 0.;

    {
      // After updating the current time, we compute the current residuals.
      // The residual of the continuity equation is treated implicitly, and we
      // reuse the base-class operators. The residual of the momentum equation
      // is evaluated for both the explicit and the implicit parts.
      // Note that, in the case of a final explicit stage, the explicit and
      // implicit residuals are evaluated with the same weights, so we can store
      // a single residual and save some memory. This explains the different
      // code path for a time integrator whose last stage is explicit.

      // Compute the DG residuals F, G and L required by the IMEX RK scheme.
      // These contributions are stored for the RK stage combinations,
      // while the operators B, E and D are handled in the implicit solve.
      TimerOutput::Scope t(timer, "rk_stage hydro - integrals L_h"); // Start a timer to measure the execution time of this scope.

      // Update all time-dependent boundary conditions to the current RK stage time.
      // 'auto' automatically deduces the type of each map element 'i', where
      // 'i.second' is a pointer to the boundary condition object.
      // The '->' operator accesses the set_time() method through this pointer.
      // This updates the evaluation time without applying the boundary conditions yet.
      for (auto &i : bc->supercritical_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->height_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->discharge_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->absorbing_outflow_boundaries)
        i.second->set_time(current_time);
      bc->problem_data->set_time(current_time);

      /*
      *
      * 'data' is a MatrixFree<dim, Number> object declared as a protected
      * member of OceanoOperator in OceanDG.h and inherited by this class
      * through 'using Base::data'.
      *
      * MatrixFree::loop() executes the specified local operators over
      * all cells, interior faces and boundary faces, assembling their
      * contributions into the destination vector.
      * 
      * MatrixFree::cell_loop() executes only cell-volume operators,
      * without evaluating interior-face or boundary-face contributions.
      * It can also use additional callbacks (lambda functions) to
      * perform vector operations, such as combining RK residuals.
      * 
      * We explicitly use Base::local_apply_* from OceanDG.h to evaluate some
      * the operators(rather than the corresponding methods defined
      * in OceanoOperatorSemiImplicit).
      */

      // 'data' manages the matrix-free DG computations but does not store the residual.
      // The computed cell, interior-face and boundary-face contributions are accumulated
      // in vec_ki_height[current_stage + 1], while current_ri remains the input state.

      // Compute the DG residual F for the free-surface equation.
      data.loop(&Base::local_apply_cell_height, // Volume contributions of F
                &Base::local_apply_face_height, // Interior-face contributions of F
                &Base::local_apply_boundary_face_height, // Boundary-face contributions of F
                static_cast<const Base *>(this), // Cast the current (this) object to its Base type
                                                 // possible since OceanoOperatorSemiImplicit derives from Base.
                vec_ki_height[current_stage+1], // Destination: DG residual of the current RK stage.
                current_ri, // Source: intermediate RK state (zeta and discharge).
                true, // Initialize the destination vector to zero before accumulating contributions
                      // Recall that vec_ki is a std::vector of distributed vectors, and only the vector with index
                      // [current_stage+1] get set to zero for initialization
                MatrixFree<dim, Number>::DataAccessOnFaces::values, // Required face data access for the source vector (current_ri).
                MatrixFree<dim, Number>::DataAccessOnFaces::values); // Required face data access for the destination vector (vec_ki_height).


      // 'current_stage' corresponds to the theoretical RK stage l (with zero-based indexing),
      // while m denotes the previous stages (m < l), whose contributions are already known.
      // Check whether we are at the final RK stage and whether it has no implicit
      // diagonal contribution (factor_matrix = dt * a_tilde_ll = 0).
      if (current_stage == n_stages-1 && !is_last_stage_implicit)
        {
          // Compute the DG residual G (non-stiff discharge operator) in the case of FULLY-EXPLICIT (even the friction)
          // Store G^(l) at index 2*current_stage+1.
          // Each RK stage uses two slots: one for G (non-stiff) and one for L (stiff).
          // Index 0 is reserved as a temporary vector for RHS assembly.
          // Here we expect only the use of non-stiff operators since we are in explicit RK
          data.loop(
                &Base::local_apply_cell_discharge, // Volume cotnributions of G
                &Base::local_apply_face_discharge, // Interior-face contributions of G
                &Base::local_apply_boundary_face_discharge, // Boundary-face contributions of G
                static_cast<const Base *>(this), // Cast the current (this) object to its Base type
                                                 // possible since OceanoOperatorSemiImplicit derives from Base.
                vec_ki_discharge[2*current_stage+1], // Destination: STIFF-DG residual of the current RK stage in odd indexes
                current_ri, // Source: intermediate RK state (zeta and discharge).
                true, // Initialize the destination vector to zero before accumulating contributions
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
      else // We are either NOT at the last stage OR the last stage is implicit (not exclusive or) 
        {
          /*
          * Compute the non-stiff discharge residual G using the operators
          * defined in OceanoOperatorSemiImplicit.
          * Unlike the Base implementation, these exclude the stiff contributions
          * (pressure and friction), which are handled separately by L.
          */
          data.loop(
                &OceanoOperatorSemiImplicit::local_apply_cell_nonstiff_discharge, // Volume cotnributions of G
                &OceanoOperatorSemiImplicit::local_apply_face_nonstiff_discharge, // Interior-face contributions of G
                &OceanoOperatorSemiImplicit::local_apply_boundary_face_nonstiff_discharge, // Boundary-face contributions of G
                this, // This object, doesn't need upcasting since it's for the class SemiImplicit
                vec_ki_discharge[2*current_stage+1], // Destination: NON-STIFF-DG residual of the current RK stage in odd indexes
                current_ri, // Source: intermediate RK state (zeta and discharge).
                true, // Initialize the destination vector to zero before accumulating contributions
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
          /*
          * Compute the stiff discharge residual L using the operators
          * defined in OceanoOperatorSemiImplicit.
          * These include the pressure and friction contributions, treated
          * implicitly in the IMEX Runge-Kutta scheme.
          */      
          data.loop(
                &OceanoOperatorSemiImplicit::local_apply_cell_stiff_discharge, // Volume contributions of L
                &OceanoOperatorSemiImplicit::local_apply_face_stiff_discharge, // Interior-face contributions of L
                &OceanoOperatorSemiImplicit::local_apply_boundary_face_stiff_discharge, // Boundary-face contributions of L
                this, // This object, doesn't need upcasting since it's for the class SemiImplicit
                vec_ki_discharge[2*current_stage+2], // Destination: STIFF-DG residual of the current RK stage in even indexes
                current_ri, // Source: intermediate RK state (zeta and discharge).
                true, // Initialize the destination vector to zero before accumulating contributions
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
    }

    // Now we combine the residuals and apply the inverse mass matrix to update the solution.
    {
      TimerOutput::Scope t(timer, "rk_stage hydro - inv mass + vec upd");
      // Fully explicit RK stage: no implicit solve is required, so we use a different code path.
      if (current_stage == n_stages-1 && !is_last_stage_implicit)
        {          
          // As anticipated, If the last stage is explicit, we use a completely
          // different code path, as we do not have to invert any sparse matrix.
          // We simply accumulate the residuals in a temporary vector, invert the
          // local mass matrix, and add the result to the old solution vector.

          // First, we assemble the right-hand side of the free-surface equation.
          // The RK stage residuals are weighted by their coefficients and accumulated
          // in vec_ki_height.front().
          // The local_apply_cell_mass_height operator then adds H(zeta^n),
          // which includes the bathymetry contribution in the general case.
          data.cell_loop(
            &Base::local_apply_cell_mass_height,
            static_cast<const Base *>(this),
            vec_ki_height.front(),
            solution_height,
            [&](const unsigned int start_range, const unsigned int end_range) {
              // DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i           = vec_ki_height[1].local_element(i);
                  vec_ki_height.front().local_element(i)  = factor_residual[0]  * k_i;
                  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
		      k_i              = vec_ki_height[j+1].local_element(i);
		      vec_ki_height.front().local_element(i) += factor_residual[j]  * k_i;
		    }
                }
	    },
            std::function<void(const unsigned int, const unsigned int)>(),
            0);
          // Apply the inverse mass operator to obtain the updated free-surface elevation.
          // For the GMRES test, this operation will be replaced by an iterative solve.

          /*
          solution_height.zero_out_ghost_values();
          data.cell_loop(
            &Base::local_apply_inverse_mass_matrix_height,
            static_cast<const Base *>(this),
            solution_height,
            vec_ki_height.front());
          */

          // Solve the free-surface equation using GMRES.
          // For the current test, the operator reduces to the mass matrix.

          solution_height.zero_out_ghost_values(); // mette a 0 i valori sulle celle ghost sugli altri processori

          SolverControl solver_control(
              1000,
              1e-8);
            // 1e-10 * vec_ki_height.front().l2_norm()

          SolverGMRES<LinearAlgebra::distributed::Vector<Number>>
              gmres(solver_control);

          gmres.solve(
              *this,
              solution_height,
              vec_ki_height.front(),
              PreconditionIdentity());

          //std::cout << "GMRES iterations: "
          //<< solver_control.last_step() << std::endl;

          // Assemble the right-hand side of the discharge equation by accumulating
          // the RK stage residuals weighted by their corresponding coefficients.
          // Apply the inverse mass matrix to compute the discharge increment,
          // which is then added to the previous solution.
          data.cell_loop(
            &Base::local_apply_inverse_mass_matrix_discharge,
            static_cast<const Base *>(this),
            next_ri_discharge,
            vec_ki_discharge.front(),
            [&](const unsigned int start_range, const unsigned int end_range) {
              // DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number kex_i           = vec_ki_discharge[1].local_element(i);
                  vec_ki_discharge.front().local_element(i)  = factor_residual[0] * kex_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      kex_i              = vec_ki_discharge[2*j+1].local_element(i);
                      const Number kim_i = vec_ki_discharge[2*j].local_element(i);
                      vec_ki_discharge.front().local_element(i) += factor_residual[j]   * kex_i
                                                                 + factor_residual[j-1] * kim_i;
                    }
                }
            },
            [&](const unsigned int start_range, const unsigned int end_range) {
              // DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number sol_i     = next_ri_discharge.local_element(i);
                  solution_discharge.local_element(i)  += sol_i;
                }
            },
            1);
        }
      else // We are either NOT at the last stage OR the last stage is implicit (not exclusive or) 
        {
          /*
          * Assemble and combine the previously computed DG residuals using
          * the IMEX Runge-Kutta coefficients, then update the stage solutions.
          *
          * For implicit stages, the procedure follows equations (17a)-(17c):
          * 1. Compute the intermediate discharge Q* by applying A^(-1).
          * 2. Assemble and solve the free-surface equation for zeta
          *    (the Schur complement solve is still to be implemented).
          * 3. Correct Q* using the pressure contribution to obtain Q.
          *
          * For a final explicit stage, update the solution directly using
          * the accumulated residuals and the inverse mass matrices.
          */

          // For the inner stages we solve the system using the Schur complement.
          // The first step updates the discharge with the explicit contributions
          // only. A cell loop assembles the right-hand side into the
          // `front()` vector, summing the stiff and non-stiff residuals computed
          // so far, and then adds the mass term at time level `n` within a
          // `local_apply` operator. A second cell loop computes the intermediate
          // discharge, denoted as "star" discharge, by inverting the mass matrix
          //  and stores it into the `next_ri` vector.

          /*
          * First step of Eq. (17a): assemble the RHS for the intermediate discharge Q*.
          * Combine the previously computed non-stiff (G) and stiff (L) DG residuals,
          * weighted by the corresponding IMEX RK coefficients, and add the mass
          * contribution M*Q^n through Base::local_apply_cell_mass_discharge.
          *
          * In the paper, M*Q^n is denoted by Q_tilde^n.
          *
          * The resulting RHS is stored in vec_ki_discharge.front() and will be
          * used in the next cell_loop to compute Q* by applying A^(-1).
          */

          // The last three arguments of data.cell_loop() are:
          // 1. operation_before_loop: lambda executed before the cell operator,
          //    initializing dst with the weighted RK residuals G and L.
          // 2. operation_after_loop: optional callback executed after the cell operator;
          //    here an empty std::function, so no additional operation is performed.
          // 3. dof_handler_index_pre_post: index of the DoFHandler used by the callbacks;
          //    here 1, corresponding to the discharge field.
          //
          // The cell operator adds M * Q^n to the RHS initialized by the first lambda,
          // through integrate_scatter(), completing Eq. (17a).
          data.cell_loop(
            &Base::local_apply_cell_mass_discharge, // mass matrix applied to discharge
            static_cast<const Base *>(this), // Cast the current (this) object to its Base type
                                             // possible since OceanoOperatorSemiImplicit derives from Base.
            vec_ki_discharge.front(), // Destination: temporary vector storing the assembled RHS of Eq. (17a).
                                      // .front() access to the first index of the vector i.e. [0]
            solution_discharge, // Source: discharge solution Q^n, used to compute M * Q^n = Q_tilde^n.
            // Now we define Lambda function specifying the operations to be executed by data.cell_loop().
            // In this case, it combines the previously computed G (non-stiff) and L (stiff)
            // residuals using the corresponding IMEX RK coefficients.
            // The capture [&] allows access by reference to external variables used
            // inside the lambda, such as vec_ki_discharge, factor_residual,
            // factor_tilde_residual and current_stage.
            // start_range and end_range are provided by cell_loop() and define
            // the range of local vector entries to process.
            [&](const unsigned int start_range, const unsigned int end_range) { // start/end_range are both given by data.loop
              /* DEAL_II_OPENMP_SIMD_PRAGMA */

              // start_range and end_range define the range of local DoF indices
              // to be processed, provided automatically by data.cell_loop().
              // The loop iterates over vector entries, not cells or SIMD lanes.
              for (unsigned int i = start_range; i < end_range; ++i) // cycle on the local DoF
                {
                  // Now we compute the explicit and implicit contributions for the i-th DoF
                  // Recall that vec_ki_discharge is a std::vector of distributed vectors,
                  // storing the stiff (L) and non-stiff (G) residuals of the RK stages.
                  // local_element(i) accesses the scalar value associated with the i-th
                  // locally owned DoF of the selected distributed vector.
                  Number kex_i           = vec_ki_discharge[1].local_element(i); // Non-stiff residual G of the first RK stage, at local DoF i.
                  Number kim_i           = vec_ki_discharge[2].local_element(i); // Stiff residual L of the first RK stage, at local DoF i.
                  
                  // Initialize the temporary RHS vector (front() = index 0) with the weighted
                  // sum of the first-stage explicit and implicit residuals at DoF i.
                  // Subsequent RK stage contributions will be accumulated using +=.
                  vec_ki_discharge.front().local_element(i)  = factor_residual[0]       * kex_i
                                                             + factor_tilde_residual[0] * kim_i;
                  // Loop over the RK stages up to the current one, retrieving the
                  // corresponding non-stiff (G) and stiff (L) residuals at local DoF i.
                  // Accumulate their weighted contributions into vec_ki_discharge.front()
                  // using +=, preserving the contributions from previous stages.                                          
                  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
		      kex_i              = vec_ki_discharge[2*j+1].local_element(i);
	              kim_i              = vec_ki_discharge[2*j+2].local_element(i);
		      vec_ki_discharge.front().local_element(i) += factor_residual[j]        * kex_i
		                                                 + factor_tilde_residual[j]  * kim_i;
		    }
                }
	    },
            std::function<void(const unsigned int, const unsigned int)>(), // No additional vector-update callback (void second Lambda function)
            1); // DoFHandler index for the callbacks: 1 corresponds to the discharge field.
            // Finished arguments of the first data.cell_loop


          // Second step of Eq. (17a): compute the intermediate discharge Q*
          // by applying the inverse of the friction-modified mass matrix A
          // to the previously assembled RHS.
          // Since A is block-diagonal across DG cells, its inverse can be
          // applied locally without solving a global linear system.
          // The degree-zero case uses an exact lumped inverse.
          data.cell_loop(
            &OceanoOperatorSemiImplicit::local_apply_inverse_modified_mass_matrix_discharge, // inversion of the mass matrix + friction term
            this,
            next_ri_discharge, // Destination: intermediate discharge Q* of Eq. (17a).
            {vec_ki_discharge.front(), current_ri[0], current_ri[1]}, // RHS (that we just computed), reference zeta and discharge for friction.
            true); // Clear the destination before applying A^(-1).

          // The second step consists of updating the free surface. A cell/face
          // loop computes the mass flux term related to the "star" discharge.
          // To do this, specific `local_apply` operators are created for the
          // star discharge.
          // The right-hand side is then assembled into the `front()` vector,
          // summing the residuals computed so far multiplied by proper weights,
          // The mass term at time level `n` is added within a `local_apply`
          // operator, as in step 1.
          //
          // The mass matrix inversion is left to Mattia's master thesis.

          // Assemble the RHS of Eq. (17b) for the free-surface elevation.
          // First, data.loop() computes the mass-flux contribution B*Q* using
          // the intermediate discharge Q* and the reference free surface zeta
          // (also needed to evaluate the numerical fluxes at cell faces).
          data.loop(&OceanoOperatorSemiImplicit::local_apply_cell_height, // Local operator defining the cell-volume contribution. (B contribution)
                    &OceanoOperatorSemiImplicit::local_apply_face_height, // Local operator defining the interior-face contribution.
                    &OceanoOperatorSemiImplicit::local_apply_boundary_face_height, // Local operator defining the boundary-face contribution.
                    this, // Pointer to the current semi-implicit operator.
                    vec_ki_height.front(), // Destination: temporary vector storing B*Q* (index 0).
                    {current_ri[0], next_ri_discharge}, // Source: reference zeta (zeta(l)) and intermediate discharge Q*.
                    true, // Clear dst before assembling the mass-flux contribution.
                    MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values);
          // Then, data.cell_loop() combines this contribution (kf_i) with the
          // previously computed RK residuals F (k_i), weighted by the corresponding
          // IMEX coefficients. The lambda assembles these terms in
          // vec_ki_height.front(), while the cell operator adds M*zeta^n.
          data.cell_loop(
            &Base::local_apply_cell_mass_height, // Apply the standard height mass matrix M.
            static_cast<const Base *>(this), // Use the Base class implementation.
            vec_ki_height.front(), // Destination: RHS of Eq. (17b).
            solution_height, // Source: zeta^n, used to compute M*zeta^n.

            // Before-loop callback: combine B*Q* and the stored RK residuals F.
            [&](const unsigned int start_range, const unsigned int end_range) {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  // Extract the mass-flux contribution and first-stage residual at local DoF i.
                  Number kf_i         = vec_ki_height.front().local_element(i); // Extracting (B*Q*)_i.
                  Number k_i          = vec_ki_height[1].local_element(i); // First stored F residual.
                  // Initialize the RHS contribution with the weighted B*Q*
                  // and first-stage F residual.
                  // factor_matrix = dt * a_tilde_ll, where a_tilde_ll is the diagonal
                  // coefficient of the implicit Butcher tableau for the current RK stage.
                  // It weights the implicit contribution B*Q* in Eq. (17b) and is also
                  // used to construct the friction-modified mass matrix A in Eq. (17a).
                  vec_ki_height.front().local_element(i)  = factor_matrix * kf_i + factor_tilde_residual[0]  * k_i;
                  // Accumulate the F residuals from the remaining RK stages.
                  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
		      k_i              = vec_ki_height[j+1].local_element(i); // F residual of stage j+1
		      vec_ki_height.front().local_element(i) += factor_tilde_residual[j]  * k_i;  // Add weighted contribution
		    }
                }
	    },
            std::function<void(const unsigned int, const unsigned int)>(), // No after-loop callback.
            0); // DoFHandler index for the callbacks: 0 corresponds to the height field.

          // The mass matrix inversion is left to Mattia master thesis. It should go here
          // with output vector the `solution` vector for last stage and `next_ri` vector
          // for inner stages.
          /* TODO */
          // Guaarda rigaa 824 di Atmospheric flow di orlndo
	  // Clsse di DEALII = SolverGMRES, solvercontrol dealii (fine tuning)
          // Noi abbimo già la rhs, pres_fix è l solution, che per noi è next_ri_height perchè solution_height è zeta_n e euler_matrix è la nostra lhs. Per creare la lhs us l MatrixDFreeOperator, è una derived clss e va reimplementta un tot di roba, tipo pply_add, per noi apply_add è tutto l'operatore dellaa lefthand side, dove ovviamente avro dei data_cell loop da faare e dei dta_face_loop da fare e guardare anche i vmult, apply_add probabilmente prende in input direttamente tutt la rhs, va tutto scritto  come matrice*vettore, senza salvare mtrici, nota bene che non è detto che bisogna costruire le mtrici perchè maaagaari sono dei fake vmult, che in realtà sono solo dei daata_loop
          if (current_stage == n_stages-1)
            {
              solution_height.zero_out_ghost_values();
              data.cell_loop(
                &Base::local_apply_inverse_mass_matrix_height,
                static_cast<const Base *>(this),
                solution_height,
                vec_ki_height.front());
            }
          else
            {
              next_ri_height = solution_height;
              next_ri_height.zero_out_ghost_values();
              data.cell_loop(
                &Base::local_apply_inverse_mass_matrix_height,
                static_cast<const Base *>(this),
                next_ri_height,
                vec_ki_height.front());
            }


          // The last step is the correction of the "star" discharge. With the
          // updated free surface at hand, we compute the pressure terms. For face
          // loops, we can reuse the stiff operators, which are equal to the pressure
          // face operators.
          // In principle, the discharge vector should not enter the pressure terms.
          // However, for some boundary conditions the boundary free-surface value
          // depends on the inner discharge value, which is why we need to pass the
          // discharge vector.
          // A final inversion of the discharge mass matrix completes the step and
          // updates the discharge.
          std::vector<LinearAlgebra::distributed::Vector<Number>> input_vectors =
            (current_stage == n_stages - 1)
              ? std::vector<LinearAlgebra::distributed::Vector<Number>>{solution_height, current_ri[1]}
              : std::vector<LinearAlgebra::distributed::Vector<Number>>{next_ri_height, current_ri[1]};
          data.loop(
                &OceanoOperatorSemiImplicit::local_apply_cell_pressure_discharge,
                &OceanoOperatorSemiImplicit::local_apply_face_stiff_discharge,
                &OceanoOperatorSemiImplicit::local_apply_boundary_face_stiff_discharge,
                this,
                vec_ki_discharge.front(),
                input_vectors,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);

          data.cell_loop(
            &OceanoOperatorSemiImplicit::local_apply_inverse_modified_mass_matrix_discharge,
            this,
            vec_ki_discharge.front(),
            {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
              if (current_stage == n_stages-1)
                {
                  // DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      Number k_i           = vec_ki_discharge.front().local_element(i);
                      const Number sol_i   = next_ri_discharge.local_element(i);
                      solution_discharge.local_element(i)  = sol_i + factor_matrix * k_i;
                    }
                }
              else
                {
                  // DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      Number k_i            = vec_ki_discharge.front().local_element(i);
                      const Number sol_i    = next_ri_discharge.local_element(i);
                      next_ri_discharge.local_element(i) = sol_i + factor_matrix  * k_i;
                    }
                }
            },
            1);
        }
    }
  }

} // namespace SpaceDiscretization

#endif //OCEANDGSEMIIMPLICIT_H
