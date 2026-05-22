/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/sparse/solver/detail/lobpcg.cuh>

namespace raft::sparse::solver {

template <typename value_t, typename index_t>
void lobpcg(
  const raft::handle_t& handle,
  // IN
  raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,    // shape=(n,n)
  raft::device_matrix_view<value_t, index_t, raft::col_major> X,  // shape=(n,k) IN OUT Eigvectors
  raft::device_vector_view<value_t, index_t> W,                   // shape=(k) OUT Eigvals
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B =
    std::nullopt,  // shape=(n,n)
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> M =
    std::nullopt,  // shape=(n,n)
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> Y =
    std::nullopt,  // Constraint matrix shape=(n,Y)
  value_t tol           = 0,
  std::int32_t max_iter = 20,
  bool largest          = true)
{
  detail::lobpcg(handle, A, X, W, B, M, Y, tol, max_iter, largest);
}
};  // namespace raft::sparse::solver
