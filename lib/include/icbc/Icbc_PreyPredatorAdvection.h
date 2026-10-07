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
 * Author: Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
 */
#ifndef ICBC_PREYPREDATORADVECTION_H
#define ICBC_PREYPREDATORADVECTION_H

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // The test is for the biological module. A background flow in solid rotation
  // transports a square blob in which the prey and predator concentrations are
  // at equilibrium. The time scale of the simulation matches the biological
  // time scale.

  using namespace dealii;

  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. In the initial test case, the useful parameter is
  // the angular velocity
  constexpr double omega = 0.5; // T = 2*pi/|OMEGA| = 2pi where |OMEGA| = 2 * omega in our case



  // @sect3{Equation data}
  //
  // We do not have any source term and no associated data values.
  // This syntax describes a family of classes ProblemData depending on the value of "dim"
  template <int dim>
  class ProblemData : public Function<dim> // ProblemData is a class derived from
  					   // the public class Function of dealii
  {
  public:
    ProblemData(IO::ParameterHandler &prm); // Constructor, handles .prm via a type
    					    // ParameterHandler
    ~ProblemData() = default;               // Destructor
    // Methods
    // value method: gives back a double value, by reciving as input, vector value (p, component)
    // where p = point of space <dim>, component = component number in the vector
    // (for ex. component = 0 means h in SW)
    // So the value method, tells us the value of a certain component in a point in space
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override; // if component is not
                         // defined, then use = 0, and in general override the value method in 
                         // Function
  };
  // Syntax to call methods from a class: NomeClasse :: NomeMethod
  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/) // Here we are calling the 
  // constructor, notice that the syntax "/*prm*/" indicates that we need to call this due to the 
  // ProblemData class, but we do not use it in this call
    : Function<dim>(dim+3) // perchè dim + 3?
  {}


  // Here we are calling the method value, but we are not using nor the point or the component and 
  // we simply return 0
  // Notice that in this case we are not using component, but we are still giving it as parameter, 
  // so it's not considered as component = 0
  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & /*x*/,
                                 const unsigned int /*component*/) const
  {
    return 0.0;
  }



  // The exact solution is obtained by considering a solid-body rotation
  // around the center of the domain (L_x/2, L_y/2), so just imagine a circle equation
  // around that center, multiplied by $/omega$ angular velocity. The corresponding
  // velocity field is:
  // \begin{equation*}
  // \begin{aligned}
  // u(x,y) &= -2\omega\left(y-\frac{L_y}{2}\right),
  // v(x,y) &= +2\omega\left(x-\frac{L_x}{2}\right).
  // \end{aligned}
  // \end{equation*}
  // with $\omega$ denoting the angular velocity of the rotation, $L_x$ and $L_y$ represent the 
  // length of the domain along x and y directions (anticlock-wise rotation)
  // Notice that this solution is constructed in a way to have a period of 2pi, so we choose
  // $/omega$ accordingly
  
  // WE construct the exact solution class as a derivation of Function class
  template <int dim, int n_vars>
  class ExactSolution : public Function<dim>
  {
  public:
    // Takes in input (time, parameters red from the .prm)
    ExactSolution(const double time,
                  IO::ParameterHandler &prm)
      : Function<dim>(n_vars, time)
    {
      prm.enter_subsection("Physical constants");
      g = prm.get_double("g");
      prm.leave_subsection();

      prm.enter_subsection("Biological constants");
      r = prm.get_double("prey_growth_rate");
      K = prm.get_double("prey_capacity");
      e = prm.get_double("predator_efficiency");
      a = prm.get_double("predator_attack_rate");
      tau = prm.get_double("predator_handling_time");
      d = prm.get_double("predator_death_rate");
      prm.leave_subsection();
    }
    ~ExactSolution() = default; // default destructor

    // Usual value method
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    double g;
    double r;    /*!<prey intrinsic growth rate */
    double K;    /*!<prey carrying capacity */
    double e;    /*!<predator conversion efficiency */
    double a;    /*!<predator attack rate */
    double tau;  /*!<handling time */
    double d;    /*!<predator mortality rate */
  };

  // We call the value method, evaluated at point "x" and "component".
  // The exact solution is the solid body rotation of a square blob. Inside
  // the square prey and predator are computed from the (stable for now)
  // equilibrium solution of the Rosenzweig-MacArthur model.
  // We take advantage of deal.ii BoundingBox which works only for quadrilaterals
  // aligned along the coordinate axis. Thus the point's coordinate "x" are
  // transported back along the characterstic line. We ispect if the transported
  // point follows within the initial square blob.
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    const double t = this->get_time();
    Assert(dim == 2, ExcNotImplemented());
    Assert(n_vars == dim+3, ExcNotImplemented());

    Point<dim> x0;
    x0[0] = 2;
    x0[1] = 2;
    const double depth = 0.4;
    const double u     = -2 * omega * (x[1] - x0[1]);
    const double v     = +2 * omega * (x[0] - x0[0]);

    const double theta = 2 * omega * t;
    const double dx    = x[0] - x0[0];
    const double dy    = x[1] - x0[1];
    const double xr    = x0[0] + cos(theta) * dx + sin(theta) * dy;
    const double yr    = x0[1] - sin(theta) * dx + cos(theta) * dy;

    Point<2> lower_left(1., 1.);
    Point<2> upper_right(1.6, 1.6);
    BoundingBox<2> bbox(std::make_pair(lower_left, upper_right));
    if (component == 0)
      return depth;
    else if (component == 1)
      return u;
    else if (component == 2)
      return v;
    else if (component == 3)
      if (bbox.point_inside(Point<2>(xr, yr), 1e-6))
        return d / ( a * (e-d*tau) );
      else
        return 0.;
    else
      if (bbox.point_inside(Point<2>(xr, yr), 1e-6))
        return e*r * (a*K*(e-d*tau)-d) / ( a*a*K*(e-d*tau)*(e-d*tau) );
      else
        return 0.;
}



  // Dirichlet boundary conditions (inflow) are specified on the left boundary of the domain.
  // The right boundary is for outflow. Top and bottom boundaries are wall. Please note that,
  // for the vortex parameters given above, the flow is supercritical and the choice of
  // boundary conditions seems appropriate.
  
  // He we construct the InitialCondition using the ExactSolution class, in our case the solution
  // is stationary
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0.,prm)
    {}
    ~Ic() = default;
  };



  template <int dim, int n_vars>
  class BcPreyPredatorAdvection : public BcBase<dim, n_vars>
  {
  public:

    BcPreyPredatorAdvection(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcPreyPredatorAdvection() = default;

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  template <int dim, int n_vars>
  void BcPreyPredatorAdvection<dim, n_vars>::set_boundary_conditions()
  {
    this->set_supercritical_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_supercritical_inflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    //this->set_supercritical_outflow_boundary(
    //  1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    //this->set_supercritical_outflow_boundary(
    //  2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    // Teoricamente posso commentare il wall boundary
    this->set_wall_boundary(0);
  }
} // namespace ICBC
#endif //ICBC_PREYPREDATORADVECTION_H
