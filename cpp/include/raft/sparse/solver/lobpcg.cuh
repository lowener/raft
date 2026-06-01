/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __LOBPCG_H
#define __LOBPCG_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/detail/lobpcg.cuh>
#include <raft/sparse/solver/lobpcg_types.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

#include <optional>

namespace RAFT_EXPORT raft {
namespace sparse::solver {

/**
 * @defgroup lobpcg LOBPCG eigensolver
 * @{
 */

/**
 * @brief Find the leading (or trailing) eigenpairs of a symmetric positive
 *        (semi-)definite matrix using the LOBPCG algorithm.
 *
 * @tparam IndexTypeT Integer type used to index the input CSR matrix.
 * @tparam ValueTypeT Floating-point type used for matrix values.
 * @tparam NNZTypeT   Integer type used to index nnz of CSR (typically same as IndexTypeT).
 *
 * @param[in]    handle      raft resources.
 * @param[in]    config      LOBPCG configuration (tolerance / max_iterations / largest).
 * @param[in]    A           Symmetric (semi-)positive-definite sparse matrix in CSR format.
 * @param[inout] X           Initial guess on input, eigenvectors on output (n x k, col-major).
 * @param[out]   W           Eigenvalues on output, length k.
 * @param[in]    B           Optional sparse SPD operator for the generalized problem
 *                           A x = lambda B x.
 * @param[in]    M           Optional sparse preconditioner.
 */
template <typename IndexTypeT, typename ValueTypeT, typename NNZTypeT = IndexTypeT>
void lobpcg_compute_eigenpairs(
  raft::resources const& handle,
  lobpcg_solver_config<ValueTypeT> const& config,
  raft::device_csr_matrix_view<const ValueTypeT, IndexTypeT, IndexTypeT, NNZTypeT> A,
  raft::device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major> X,
  raft::device_vector_view<ValueTypeT, IndexTypeT> W,
  std::optional<raft::device_csr_matrix_view<const ValueTypeT, IndexTypeT, IndexTypeT, NNZTypeT>>
    B = std::nullopt,
  std::optional<raft::device_csr_matrix_view<const ValueTypeT, IndexTypeT, IndexTypeT, NNZTypeT>>
    M = std::nullopt)
{
  using sp_mat_t = raft::spectral::matrix::sparse_matrix_t<IndexTypeT, ValueTypeT>;

  auto sA = A.structure_view();
  sp_mat_t matA(handle,
                sA.get_indptr().data(),
                sA.get_indices().data(),
                A.get_elements().data(),
                static_cast<IndexTypeT>(sA.get_n_rows()),
                static_cast<IndexTypeT>(sA.get_n_cols()),
                static_cast<std::uint64_t>(sA.get_nnz()));

  std::optional<sp_mat_t> matB;
  if (B.has_value()) {
    auto s = B->structure_view();
    matB.emplace(handle,
                 s.get_indptr().data(),
                 s.get_indices().data(),
                 B->get_elements().data(),
                 static_cast<IndexTypeT>(s.get_n_rows()),
                 static_cast<IndexTypeT>(s.get_n_cols()),
                 static_cast<std::uint64_t>(s.get_nnz()));
  }
  std::optional<sp_mat_t> matM;
  if (M.has_value()) {
    auto s = M->structure_view();
    matM.emplace(handle,
                 s.get_indptr().data(),
                 s.get_indices().data(),
                 M->get_elements().data(),
                 static_cast<IndexTypeT>(s.get_n_rows()),
                 static_cast<IndexTypeT>(s.get_n_cols()),
                 static_cast<std::uint64_t>(s.get_nnz()));
  }

  std::optional<raft::device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>> Y = std::nullopt;

  detail::lobpcg<ValueTypeT, IndexTypeT>(handle,
                                         matA,
                                         X,
                                         W,
                                         matB,
                                         matM,
                                         Y,
                                         config.tolerance,
                                         config.max_iterations,
                                         config.largest,
                                         config.verbosity_level);
}

/** @} */  // end of group lobpcg

}  // namespace sparse::solver
}  // namespace RAFT_EXPORT raft

#endif
