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
 * Author: Luca Arpaia, 2025
           Giuseppe Orlando, 2026
 */
#ifndef OPERATORS_H
#define OPERATORS_H

#include <deal.II/matrix_free/fe_evaluation.h>

/**
 * The following classes implement the operation of the action of the inverse of a
 * @ref GlossMassMatrix "mass matrix" for a DGQ element.
 * It is useful because the original class `CellwiseInverseMassMatrix` provided by
 * the library is valid only for the special case of an evaluation object with as
 * many quadrature points as there are cell degrees of freedom. Here you can use a
 * generic quadrature formula. It is thus more flexible. Moreover we tailor the
 * singular case specificly to wet-dry applications. That is track fully dry cells
 * and we set the output array to zero (nothing moves in fully dry cells).
 *
*/

// Here we construct two operators classes, then we construct the methods "apply"
// for each one and that method is the local applycation of the inverse matrix to the input vector
namespace MatrixFreeOperatorsOceano
{

  using namespace dealii;

 /**
  * The first class produces the exact mass matrix for DGQ elements. This algorithm
  * uses a LU factorization, implemented by the deal.ii Gauss-Jordan algorithm. For
  * the small matrices needed in Discontinuous Galerkin, the inversion cost is totally
  * acceptable. Indeed, the main problem is that the Gauss-Jordan routine does not
  * support vectorization and, at the end, it gives a noticeable overhead with respect
  * to the original deal.ii `CellwiseInverseMassMatrix`.
  *
  */
  template <int dim,
            int fe_degree,
            int n_components             = 1,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class CellwiseInverseMassMatrix
  {
  public:
    /**
     * Default constructor.
     */
    CellwiseInverseMassMatrix(
      const FEEvaluationBase<dim,
                             n_components,
                             Number,
                             false,
                             VectorizedArrayType> &fe_eval);

    /**
     * Default destructor.
     */
    ~CellwiseInverseMassMatrix() = default;

    /**
     * Applies the inverse @ref GlossMassMatrix "mass matrix" operation on an input array and
     * assign it to an output array. It is assumed that the passed input and output pointers are
     * of correct size, namely FEEvaluation::dofs_per_cell long for input/output and
     * FEEvaluation::dofs_per_cell power two for the mass matrix. We track the singular case
     * with a simple if statement that checks a mask passed as argument. The default behaviour
     * does not check the mask.
     * To call this function vectorized lanes must be unrolled, as we pass the cell index in the
     * batch lane in the last argument.
     *
     * @param mass_array mass matrix
     * @param in_array input vector
     * @param out_array output vector
     * @param cell_in_lane cell index in lane
     * @param mask cell is masked?
     */
    void apply(const VectorizedArrayType *mass_array,
               const VectorizedArrayType *in_array,
               VectorizedArrayType       *out_array,
               const unsigned int         cell_in_lane,
               const VectorizedArrayType  mask = 0) const;

  private:
    /**
     * A reference to the FEEvaluation object for getting info on the cell.
     */
    const FEEvaluationBase<dim,
                           n_components,
                           Number,
                           false,
                           VectorizedArrayType> &fe_eval;
  };



  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  CellwiseInverseMassMatrix<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::CellwiseInverseMassMatrix(const FEEvaluationBase<dim,
                                                    n_components,
                                                    Number,
                                                    false,
                                                    VectorizedArrayType> &fe_eval)
    : fe_eval(fe_eval)
  {}


  // CELLWISE INVERSE MASS MATRIX - FULL MATRIX
  // Applies the inverse of the local DG mass matrix on each cell.
  //
  // The full local mass matrix is explicitly constructed and inverted
  // using Gauss-Jordan elimination, then applied to each component
  // of the input vector.
  //
  // Unlike the standard deal.II implementation, this version supports
  // general quadrature rules and handles fully dry cells by setting
  // the output to zero.
  // By working locally, the important parts are the DoFs per cell.
  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  CellwiseInverseMassMatrix<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::
      apply(const VectorizedArrayType *mass_array, // Local mass-matrix entries stored in SIMD format (vector)
            const VectorizedArrayType *in_array, // Input vector to which the inverse mass matrix is applied.
            VectorizedArrayType       *out_array, // Output vector containing the result of the inversion.
            const unsigned int         cell_in_lane, // Index of the cell within the SIMD batch.
            const VectorizedArrayType  mask) const // Mask used to identify active/non-dry cells.
  {
    // If fe_degree >= 0, then given_degree = fe_degree. Else we get the degree from the object fe_eval
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;
    // Compute the degree of freedom per component, assuming tensorial elements we have:
    // DoF = (1+p)^dim
    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);
    // Read the total number of quadrature points in fe_eval
    const unsigned int n_q_points =
        fe_eval.get_shape_info().n_q_points;

    // The inversion uses scalar FullMatrix and Vector objects,
    // so each cell is extracted from the SIMD arrays and processed separately.
    // The computed result is then stored back into the corresponding SIMD lane.

    // Allocate the local mass matrix and input/output vectors.
    // Their size is determined by the number of scalar DoFs per cell.
    FullMatrix<Number> cell_matrix(dofs_per_component, dofs_per_component);
    Vector<Number> cell_src(dofs_per_component),
                   cell_dst(dofs_per_component);
    // Apply the inverse only to unmasked cells.
    // Set the output to zero for fully dry (masked) cells.
    if (mask[cell_in_lane] < n_q_points)
      {
        for (unsigned int j=0; j<dofs_per_component; ++j)
          for (unsigned int i=0; i<dofs_per_component; ++i)
            // the local mass matrix is stored as a vector, and here it gets copied into a matrix
            // Extract the (i,j) mass-matrix entry for the current SIMD lane.
            cell_matrix(i,j) = mass_array[i + j * dofs_per_component][cell_in_lane];
        // Extract the mass matrix for the current SIMD lane
        // and invert it using Gauss-Jordan elimination.
        cell_matrix.gauss_jordan();
        for (unsigned int d = 0; d < n_components; ++d) // loop over the components
          {
            for (unsigned int i = 0; i < dofs_per_component; ++i)
              // Extract the input DoF for component d and the current SIMD lane.
              cell_src[i] = in_array[i + d * dofs_per_component][cell_in_lane];
            // We put in cell_dst (destination) the mult between cell_matrix and cell_src (source)
            cell_matrix.vmult(cell_dst, cell_src); // Apply the inverse local mass matrix to each vector component.

            for (unsigned int i = 0; i < dofs_per_component; ++i)
              // Store the computed local result back into the output SIMD array.
              out_array[i + d * dofs_per_component][cell_in_lane] = cell_dst(i);
          }
      }
    else // fully dry cells, we set the output to zero
      {
        for (unsigned int d = 0; d < n_components; ++d)
          for (unsigned int i = 0; i < dofs_per_component; ++i)
            out_array[i + d * dofs_per_component][cell_in_lane] = 0.;
      }
  }



 /**
  * The second class, instead of an exact matrix, it produces the lumped mass matrix for
  * DGQ elements. This algorithm simply inverts the diagonal. Note that it is only first
  * order accurate. However vectorization is supported and enables faster inversion with
  * respect to the first class `CellwiseInverseMassMatrix`.
  *
  */
  template <int dim,
            int fe_degree,
            int n_components             = 1,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class CellwiseInverseMassMatrixLumped
  {
  public:
    /**
     * Default constructor.
     */
    CellwiseInverseMassMatrixLumped(
      const FEEvaluationBase<dim,
                             n_components,
                             Number,
                             false,
                             VectorizedArrayType> &fe_eval);

    /**
     * Applies the inverse @ref GlossMassMatrix "mass matrix" operation on an input array and
     * assign it to an output array. It is assumed that the passed input and output pointers are
     * of correct size, namely FEEvaluation::dofs_per_cell long. Note that for the mass matrix
     * we store only the diagonal. We track the singular case
     * with a `compare_and_apply_mask<SIMDComparison::less_than>` that check if the cell has
     * been masked. The default behaviour does not check the mask.
     *
     * @param mass_array mass matrix
     * @param in_array input vector
     * @param out_array output vector
     * @param mask cell is masked?
     */
    void apply(const VectorizedArrayType *mass_array,
               const VectorizedArrayType *in_array,
               VectorizedArrayType       *out_array,
               const VectorizedArrayType  mask = 0) const;

  private:
    /**
     * A reference to the FEEvaluation object for getting info on the cell.
     */
    const FEEvaluationBase<dim,
                           n_components,
                           Number,
                           false,
                           VectorizedArrayType> &fe_eval;
  };



  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  CellwiseInverseMassMatrixLumped<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::CellwiseInverseMassMatrixLumped(
      const FEEvaluationBase<dim,
      n_components,
      Number,
      false,
      VectorizedArrayType> &fe_eval)
    : fe_eval(fe_eval)
  {}


  // CELLWISE INVERSE MASS MATRIX - LUMPED APPROXIMATION
  // Applies the inverse of a diagonal (lumped) DG mass matrix.
  // So the extra-diagonal elements are lumped onto the diagonal itself
  //
  // Unlike the full-matrix approach, only the diagonal entries
  // are stored and inverted, avoiding matrix factorization.
  //
  // This approximation is first-order accurate but supports
  // SIMD vectorization, making the inversion computationally cheaper.
  // Since we do not need to compute the whole matrix, we do not need to compute the cell_matrix for each cell
  //
  // Fully dry cells are handled by setting the output to zero.
  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  CellwiseInverseMassMatrixLumped<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::
      apply(const VectorizedArrayType *mass_array, // Local mass-matrix entries stored in SIMD format.
            const VectorizedArrayType *in_array, // Input vector to which the inverse mass matrix is applied.
            VectorizedArrayType       *out_array, // Output vector containing the result of the inversion.
                                                  // we do not have cell_in_lane because we are working only on the diagonal
            const VectorizedArrayType  mask) const // Mask used to identify active/non-dry cells.
  {
    // If fe_degree >= 0, then given_degree = fe_degree. Else we get the degree from the object fe_eval
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;
    // Compute the degree of freedom per component, assuming tensorial elements we have:
    // DoF = (1+p)^dim
    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);
    // Read the total number of quadrature points in fe_eval
    const unsigned int n_q_points =
        fe_eval.get_shape_info().n_q_points;

    for (unsigned int d = 0; d < n_components; ++d)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        // Apply the inverse diagonal mass matrix component-wise.
        // For each SIMD lane, divide by the corresponding diagonal entry
        // if the cell is unmasked; otherwise, set the result to zero.

        // Apply the inverse lumped mass matrix using SIMD operations.
        // compare_and_apply_mask<less_than>(a, b, x, y) selects x if a < b,
        // and y otherwise, independently for each SIMD lane.
        //
        // Here, mask < n_q_points selects the cells where the inversion is applied.
        // The third argument computes input / diagonal mass coefficient,
        // while the fourth argument (0.) sets the output to zero for masked cells.
        //
        // The operation is performed directly on VectorizedArrayType objects,
        // without extracting individual SIMD lanes.
        out_array[i + d * dofs_per_component] =
          compare_and_apply_mask<SIMDComparison::less_than>(
            mask, n_q_points,
            1./mass_array[i] * in_array[i + d * dofs_per_component],
            0.); // when the cell is NOT fully dry, the inversion of the diagonal is simply y = x / m_ii
  }
} // namespace MatrixFreeOperatorsOceano
#endif //OPERATORS_H
