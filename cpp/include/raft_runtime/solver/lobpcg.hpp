/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/lobpcg_types.hpp>

#include <cstdint>

namespace raft::runtime::solver {

/**
 * @defgroup lobpcg_runtime LOBPCG Runtime API
 * @{
 */

#define FUNC_DECL(IndexType, ValueType)                                          \
  RAFT_EXPORT void lobpcg_solver(                                                \
    const raft::resources& handle,                                               \
    raft::sparse::solver::lobpcg_solver_config<ValueType> config,                \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,         \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,         \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,         \
    raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors, \
    raft::device_vector_view<ValueType, uint32_t> eigenvalues)

#define FUNC_DECL_M(IndexType, ValueType)                                        \
  RAFT_EXPORT void lobpcg_solver(                                                \
    const raft::resources& handle,                                               \
    raft::sparse::solver::lobpcg_solver_config<ValueType> config,                \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,         \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,         \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,         \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> M_rows,       \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> M_cols,       \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> M_vals,       \
    raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors, \
    raft::device_vector_view<ValueType, uint32_t> eigenvalues)

FUNC_DECL(int, float);
FUNC_DECL(int64_t, float);
FUNC_DECL(int, double);
FUNC_DECL(int64_t, double);

FUNC_DECL_M(int, float);
FUNC_DECL_M(int64_t, float);
FUNC_DECL_M(int, double);
FUNC_DECL_M(int64_t, double);

#undef FUNC_DECL
#undef FUNC_DECL_M

/** @} */  // end group lobpcg_runtime

}  // namespace raft::runtime::solver
