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
 * Author: Martin Kronbichler (copied from), 2020
 *         Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
 */
#ifndef ICBCBASE_H
#define ICBCBASE_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>

/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  using namespace dealii;



  // @sect3{Equation data}

  // We now define a base class for the boundary conditions. Given that
  // the governing equations are a problem with $n_vars$ equations in $d$
  // dimensions, we template the icbc class.
  // The base class is overridden by derived classes which contain the
  // boundary conditions specific to each test case.

  // For example, we willd define a bc class of shallow water votrex as:
  // class BcShallowWaterVortex : public BcBase<dim, n_vars>
  // i.e. a boundary condition class specific to the SW vortex case, derived from the general BcBase
  // So the BcBase class is a base class to define the interface of how to set the boundary conditions,
  // while the derived classes will implement the specific boundary conditions for each test case.
  // But the operative part is implemented elsewhere, in the `OceanoProblem` class

  template <int dim, int n_vars>
  class BcBase
  {
  public:

    BcBase() = default;
    virtual ~BcBase() = default;

    // The next members are for the functions that appears into the boundary
    // conditions or into the forcing terms. They are defined with the help of the
    // Deal.II `Function<dim>` which can design vector functions of the space and time.
    // The first four members associate, with the aid of a map, each boundary id with the
    // corresponding function for the boundary condition. The last member is used
    // for the problem spatially and time varying parameters, such as friction, bathymetry
    // or wind.
    // Actually we use pointers to Function, but note that we do not use regular
    // pointers but `unique_ptr`. These are particular pointers that
    // destroy the object to which they point, after use. This avoids
    // memory leaks with dynamic allocation. No other pointer should point
    // to its managed object. For this reason we will see that these particular
    // pointers cannot be copied but they can only be moved with `std::move()`

    // map: standard class and associates a key with a value. The key is the boundary_id and the value is a pointer to a Function
    // Function: class of deal.II that represents a function of space and time, with a given number of components
    // boundary_id: class of deal.II that represents a boundary id, which is an unsigned integer. Used to indentify different kinds of boundaries

    // Boundary conditions depend on the sign of the characteristic speeds
    // (eigenvalues of the flux Jacobian along the outward normal):
    //
    // lambda < 0: Incoming characteristic. Information enters the domain
    //             and must be prescribed through boundary conditions.
    //
    // lambda > 0: Outgoing characteristic. Information leaves the domain
    //             and is determined by the interior solution.
    //
    // lambda = 0: Characteristic tangent to the boundary (special case).
    //
    // The number of boundary conditions to prescribe equals the number
    // of incoming characteristics (for a non-characteristic boundary).

    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   supercritical_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   supercritical_outflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   height_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   discharge_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   absorbing_outflow_boundaries;
    // The wall boundaries are the only ones that do not require a Function, because we do not use
    // a function to impose the boundary condition. Imposing the normal velocity at 0 (no passage) resulting
    // in a normal inverted velocity.
    std::set<types::boundary_id>   wall_boundaries;

    std::unique_ptr<Function<dim>> problem_data; // problem_data is a pointer to a Function, representing the problem data
                                                 // such as bathymetry, friction, wind, etc.

    // The subsequent four member functions are the ones that fill the boundary
    // containers. They must be called from outside to specify the various types
    // of boundaries. For an inflow boundary, we must specify all components in
    // terms of free-surface $\zeta$, momentum $h\mathbf{u}$ and, eventually tracers.
    // Given this information, we then store the
    // function alongside the respective boundary id in a map member variable of
    // this class. Likewise, we proceed for the subcritical/supercritical
    // outflow boundaries (where we request a function as well,
    // which we use to retrieve the far-field state or the energy). For the
    // wall (no-penetration) boundaries we impose zero normal velocity, no
    // function necessary, so we only request the boundary id. For the present
    // DG code where boundary conditions are solely applied as part of the weak
    // form (during time integration), the call to set the boundary conditions
    // can appear both before or after the `reinit()` call to this class. This
    // is different from continuous finite element codes where the boundary
    // conditions determine the content of the AffineConstraints object that is
    // sent into MatrixFree for initialization, thus requiring to be set before
    // the initialization of the matrix-free data structures.

    // Declaration of the methods of the class.
    void set_supercritical_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_function);

    void set_supercritical_outflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> outflow_energy);

    void set_height_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_function);

    void set_discharge_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_function);

    void set_absorbing_outflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> outflow_function);

    void set_wall_boundary(const types::boundary_id boundary_id);

    void set_problem_data(std::unique_ptr<Function<dim>> problem_data);

    // The next member composes the different boundary conditions for each test case.
    // It is overridden by the derived classes specific to each test case.
    // In the boundary condition function the user defines the
    // composition of the boundary, that is for each boundary id a boundary condition
    // among the members defined above must be provided.

    // Meaning that the user will define via an object set_boundary_conditions() the boundary conditions for each boundary id
    // the function itself is virtual and it is overridden in the derived classes, so if it's not overridden
    // it will print the written error message.
    virtual void set_boundary_conditions()
    {
      std::cout << "ERROR in set_boundary_conditions()"
            << "The function is not written in the icbc derived class"
            << std::endl;
    }

  };



  // The checks added in each of the four function are used to
  // ensure that boundary conditions are mutually exclusive on the various
  // parts of the boundary, i.e., that a user does not accidentally designate a
  // boundary as both an inflow and say a supercritical outflow boundary.
  template <int dim, int n_vars> // template allows to define the class for a generic dim and n_vars
                                 // which are set in compile time.
  void BcBase<dim, n_vars>::set_supercritical_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    // We assert the the boundary_id is not already used for another type of boundary condition.
    // This is done by looking for the boundary_ide in the other boundary conditions maps and checking if what
    // we get is the end of the map, meaning that the boundary_id is not present in the map.
    AssertThrow(supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical inflow"));
    // We assert the the number of components (attribute n_components) of the function is equal to n_vars
    // object->attribute is the syntax to access the attribute of an object in C++.
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));
    // We transfer the ownership of the pointer to the map `supercritical_inflow_boundaries` with the key `boundary_id`.
    supercritical_inflow_boundaries[boundary_id] = std::move(inflow_function); // move is used for the unique_ptr
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_supercritical_outflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> outflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical outflow"));
    AssertThrow(outflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    supercritical_outflow_boundaries[boundary_id] = std::move(outflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_height_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as height inflow"));
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    height_inflow_boundaries[boundary_id] = std::move(inflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_discharge_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as height inflow"));
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    discharge_inflow_boundaries[boundary_id] = std::move(inflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_absorbing_outflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> outflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical outflow"));
    AssertThrow(outflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    absorbing_outflow_boundaries[boundary_id] = std::move(outflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_wall_boundary(
    const types::boundary_id boundary_id)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as wall boundary"));

    wall_boundaries.insert(boundary_id);
  }

  // problem_data is a case dependent function that contains the problem data, such as bathymetry, friction, wind, etc.
  // The next member composes all the external data that we need to solve the
  // problem. Problem data is case dependent; for this reason it appears inside
  // the `ICBC` namespace. The data in general depends on both time and space.
  // Deal.II has a class `Function` which returns function of space and time,
  // thus we simply create a derived class. The next member set this class. The size
  // of the data is fixed to `dim+3=5` scalar quantities.
  // First we need to handle the problem surface data. The number of
  // components for the data is fixed to `dim+3=5` scalar quantities.
  // We list them here:
  // begin{itemize}
  // \item the first component is the bathymetry.
  // \item the second is the bottom friction coefficient.
  // \item the third and fourth components are the cartesian components
  // of the wind velocity (in order, eastward and northward).
  // \item the fifth one is the Coriolis parameter.
  // end{itemize}
  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_problem_data(
    std::unique_ptr<Function<dim>> problem_data)
  {
    AssertDimension(problem_data->n_components, dim+3); // we assert its right size, i.e. dim+3=5

    this->problem_data = std::move(problem_data);
  }

} // namespace ICBC
#endif //ICBCBASE_H
