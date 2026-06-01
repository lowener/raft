/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_csr_matrix.hpp>
#include <raft/sparse/solver/lobpcg.cuh>

#include <raft_runtime/solver/lobpcg.hpp>

#include <optional>

// Note: the public runtime API mirrors the lanczos runtime - it uses uint32_t for the mdspan
// index type while the actual sparse CSR uses IndexType for storage indices.
#define FUNC_DEF(IndexType, ValueType)                                                            \
  void lobpcg_solver(const raft::resources& handle,                                               \
                     raft::sparse::solver::lobpcg_solver_config<ValueType> config,                \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,         \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,         \
                     raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,         \
                     raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors, \
                     raft::device_vector_view<ValueType, uint32_t> eigenvalues)                   \
  {                                                                                               \
    auto n_rows = static_cast<IndexType>(rows.extent(0)) - 1;                                     \
    auto n_cols = n_rows;                                                                         \
    auto nnz    = static_cast<IndexType>(cols.extent(0));                                         \
                                                                                                  \
    auto csr_structure =                                                                          \
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(               \
        const_cast<IndexType*>(rows.data_handle()),                                               \
        const_cast<IndexType*>(cols.data_handle()),                                               \
        n_rows,                                                                                   \
        n_cols,                                                                                   \
        nnz);                                                                                     \
    auto csr_matrix =                                                                             \
      raft::make_device_csr_matrix_view<const ValueType, IndexType, IndexType, IndexType>(        \
        const_cast<ValueType*>(vals.data_handle()), csr_structure);                               \
                                                                                                  \
    auto k             = static_cast<IndexType>(eigenvectors.extent(1));                          \
    auto eigvecs_typed = raft::make_device_matrix_view<ValueType, IndexType, raft::col_major>(    \
      eigenvectors.data_handle(), static_cast<IndexType>(n_rows), k);                             \
    auto eigvals_typed =                                                                          \
      raft::make_device_vector_view<ValueType, IndexType>(eigenvalues.data_handle(), k);          \
                                                                                                  \
    raft::sparse::solver::lobpcg_compute_eigenpairs<IndexType, ValueType, IndexType>(             \
      handle, config, csr_matrix, eigvecs_typed, eigvals_typed);                                  \
  }

#define FUNC_DEF_M(IndexType, ValueType)                                                          \
  void lobpcg_solver(const raft::resources& handle,                                               \
                     raft::sparse::solver::lobpcg_solver_config<ValueType> config,                \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,         \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,         \
                     raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,         \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> M_rows,       \
                     raft::device_vector_view<IndexType, uint32_t, raft::row_major> M_cols,       \
                     raft::device_vector_view<ValueType, uint32_t, raft::row_major> M_vals,       \
                     raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors, \
                     raft::device_vector_view<ValueType, uint32_t> eigenvalues)                   \
  {                                                                                               \
    auto n_rows = static_cast<IndexType>(rows.extent(0)) - 1;                                     \
    auto n_cols = n_rows;                                                                         \
    auto nnz    = static_cast<IndexType>(cols.extent(0));                                         \
                                                                                                  \
    auto csr_structure =                                                                          \
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(               \
        const_cast<IndexType*>(rows.data_handle()),                                               \
        const_cast<IndexType*>(cols.data_handle()),                                               \
        n_rows,                                                                                   \
        n_cols,                                                                                   \
        nnz);                                                                                     \
    auto csr_matrix =                                                                             \
      raft::make_device_csr_matrix_view<const ValueType, IndexType, IndexType, IndexType>(        \
        const_cast<ValueType*>(vals.data_handle()), csr_structure);                               \
                                                                                                  \
    auto M_n_rows = static_cast<IndexType>(M_rows.extent(0)) - 1;                                 \
    auto M_n_cols = M_n_rows;                                                                     \
    auto M_nnz    = static_cast<IndexType>(M_cols.extent(0));                                     \
                                                                                                  \
    auto M_csr_structure =                                                                        \
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(               \
        const_cast<IndexType*>(M_rows.data_handle()),                                             \
        const_cast<IndexType*>(M_cols.data_handle()),                                             \
        M_n_rows,                                                                                 \
        M_n_cols,                                                                                 \
        M_nnz);                                                                                   \
    auto M_csr_matrix =                                                                           \
      raft::make_device_csr_matrix_view<const ValueType, IndexType, IndexType, IndexType>(        \
        const_cast<ValueType*>(M_vals.data_handle()), M_csr_structure);                           \
                                                                                                  \
    auto k             = static_cast<IndexType>(eigenvectors.extent(1));                          \
    auto eigvecs_typed = raft::make_device_matrix_view<ValueType, IndexType, raft::col_major>(    \
      eigenvectors.data_handle(), static_cast<IndexType>(n_rows), k);                             \
    auto eigvals_typed =                                                                          \
      raft::make_device_vector_view<ValueType, IndexType>(eigenvalues.data_handle(), k);          \
                                                                                                  \
    std::optional<raft::device_csr_matrix_view<const ValueType, IndexType, IndexType, IndexType>> \
      M_opt = M_csr_matrix;                                                                       \
    std::optional<raft::device_csr_matrix_view<const ValueType, IndexType, IndexType, IndexType>> \
      B_opt = std::nullopt;                                                                       \
                                                                                                  \
    raft::sparse::solver::lobpcg_compute_eigenpairs<IndexType, ValueType, IndexType>(             \
      handle, config, csr_matrix, eigvecs_typed, eigvals_typed, B_opt, M_opt);                    \
  }
