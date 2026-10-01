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
#ifndef ICBC_SHALLOWWATERVORTEX_H
#define ICBC_SHALLOWWATERVORTEX_H

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // The first test case is a compact shallow water travelling vortex (shortly RB-vortex)
  // (Ricchiuto and Bollerman, 2008) which fulfills the shallow water equations
  // with zero force term on the right hand side.
  // The RB-vortex is $C^4$ in the depth but only $C^1$
  // in the velocity. According to the classical interpolation estimate
  // of finite element theory, we can expect only second order of convergence for
  // the variables, included momentum. This vortex is thus suited to test second order schemes.
  // We have also coded an extension of the RB-vortex to an arbitrary degree of smoothness,
  // see (Ricchiuto and Torlo, 2021 arXiv:2109.10183v1). The implementation followed here
  // is the same provided in the last reference for the degree of smoothness $p=2$.
  // Other iterative corrections to improve the vortex smoothness and
  // test higher then third order schemes can be readily implemented.
  //
  // Additionally you can add a tracer which is initalized as a linear function
  // of $y$.

  using namespace dealii;

  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. In the initial test case, the useful parameter is the angular vel
  constexpr double omega = 2; // T = 2*pi/OMEGA = 2pi where OMEGA = 2 * omega in our case



  // @sect3{Equation data}
  //
  // We do not have any source term and no associated data values.
  // This syntax describes a family of classes ProblemData depending on the value of "dim"
  template <int dim>
  class ProblemData : public Function<dim> // ProblemData is a class derived from
  					   // the public class Function of dealii
  {
  public:
    ProblemData(IO::ParameterHandler &prm); // Costruttore, handles .prm via a type
    					    // ParameterHandler
    ~ProblemData() = default; // Distruttore
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
      // omega = prm.get_double("omega"); // getter for angular velocity $/omega$
      prm.leave_subsection();
      // std :: cout << "Helloworld!" << std :: endl;
    }
    ~ExactSolution() = default; // default destroyer

    // Usual value method
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    double g;
    // double omega;
  };

  // We call the value method, evaluated at point "x" and "component"
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    // const double t = this->get_time();
    // const double t = 0; // we do not need time since the vf is stationary
    Assert(dim == 2, ExcNotImplemented());
    
    // Here we define the center (L_x,L_y) of the solid body rotation in [0,1]x[0,1]
    Point<dim> x0;
    // x0[0] = 0.5;
    // x0[1] = 0.5;
    x0[0] = 2;
    x0[1] = 2;
    // We found h to solve SW with this vector field:
    // \begin{equation*}
    // h(x,y) = h_c + \frac{2\omega^2}{g}
    // \left[(x-x_0)^2 + (y-y_0)^2\right],
    // \end{equation*}
    // where h_c is the water depth at the center of the solid-body rotation.
    // To choose h in supercritical, recall:
    // Frud's number = Fr = ||u||/sqrt(gh), Fr < 1 sub-crit, Fr > 1 super-crit, Fr = 1 crit
    // insert then the equalities for u and v.
    // To have supercritical at the boundary, we find h < (2*omega^2*r^2)/g for r = 1/2 and so we
    // find h_c < 1/(8*g)
    const double h_c = 0.4;
    const double depth = h_c + (2. * omega * omega / g)
                     * ((x[0] - x0[0]) * (x[0] - x0[0])
                        + (x[1] - x0[1]) * (x[1] - x0[1]));
                        // h chosen to be in supercritical regime
    // const double depth = 1;
    const double u     = -2 * omega * (x[1] - x0[1]);
    const double v     = +2 * omega * (x[0] - x0[0]);
    const double r2 = (x[0]-1)*(x[0]-1) + (x[1]-1)*(x[1]-1);

    if (component == 0)
      return depth;
    else if (component == 1)
      return u;
    else if (component == 2)
      return v;
    else
      // return 0.1 * x[1];
      /*if (((x[0]-1)*(x[0]-1)+(x[1]-1)*(x[1]-1)) < 0.2*0.2)
      	// return 1;
	
	return std::exp(-r2/(2.*0.2*0.2));
      else
      	return 0;
      */
      return std::exp(-r2/(2.*0.2*0.2));
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
  class BcShallowWaterVortex : public BcBase<dim, n_vars>
  {
  public:

    BcShallowWaterVortex(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcShallowWaterVortex() = default;

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  template <int dim, int n_vars>
  void BcShallowWaterVortex<dim, n_vars>::set_boundary_conditions()
  {
    this->set_supercritical_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_supercritical_outflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    // Teoricamente posso commentare il wall boundary
    this->set_wall_boundary(0);
  }
} // namespace ICBC
#endif //ICBC_SHALLOWWATERVORTEX_H
