/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/linalg_types.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/print.cuh>
#include <raft/matrix/reverse.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/matrix/triangular.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/spmm.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <optional>

namespace raft::sparse::solver::detail {

template <typename value_t, typename index_t>
auto make_transpose_layout_view(raft::device_matrix_view<value_t, index_t, raft::row_major> mds)
{
  return raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
    mds.data_handle(), mds.extent(1), mds.extent(0));
}
template <typename value_t, typename index_t>
auto make_transpose_layout_view(raft::device_matrix_view<value_t, index_t, raft::col_major> mds)
{
  return raft::make_device_matrix_view<value_t, index_t, raft::row_major>(
    mds.data_handle(), mds.extent(1), mds.extent(0));
}

/**
 * @brief Reduction operator to find the column-wise maximum
 */
template <typename DataT>
struct MaxOp {
  HDI DataT operator()(DataT a, DataT b) const { return a > b ? a : b; }
};

template <typename DataT>
struct isnan_test {
  HDI int operator()(const DataT a) const { return isnan(a); }
};

/**
 * Assemble a matrix from a list of blocks.  The block list is in row-major block
 * storage: `ins[j * n_blocks + i]` is the block at block-row `j`, block-col `i`.
 */
template <typename value_t, typename index_t>
void bmat(raft::resources const& handle,
          raft::device_matrix_view<value_t, index_t, col_major> out,
          const std::vector<raft::device_matrix_view<value_t, index_t, col_major>>& ins,
          index_t n_blocks)
{
  RAFT_EXPECTS(static_cast<size_t>(n_blocks * n_blocks) == ins.size(),
               "inconsistent number of blocks");
  // Row block heights are determined by block (j, 0); col block widths by block (0, i).
  std::vector<index_t> row_offsets(n_blocks + 1, 0);
  std::vector<index_t> col_offsets(n_blocks + 1, 0);
  for (index_t j = 0; j < n_blocks; j++) {
    row_offsets[j + 1] = row_offsets[j] + ins[j * n_blocks].extent(0);
  }
  for (index_t i = 0; i < n_blocks; i++) {
    col_offsets[i + 1] = col_offsets[i] + ins[i].extent(1);
  }
  for (index_t j = 0; j < n_blocks; j++) {
    for (index_t i = 0; i < n_blocks; i++) {
      auto& block = ins[j * n_blocks + i];
      raft::matrix::slice_insert(
        handle,
        block,
        out,
        raft::matrix::slice_coordinates<index_t>(row_offsets[j],
                                                 col_offsets[i],
                                                 row_offsets[j] + block.extent(0),
                                                 col_offsets[i] + block.extent(1)));
    }
  }
}

/* Modification of copyRows to reindex columns, col_major only.
 * On a 4x3 matrix, indices could be [0, 2] to select col 0 and 2.
 */
template <typename m_t, typename idx_array_t = int, typename idx_t = size_t>
void selectCols(const m_t* in,
                idx_t n_rows,
                idx_t /*n_cols*/,
                m_t* out,
                const idx_array_t* indices,
                idx_t n_cols_indices,
                cudaStream_t stream)
{
  idx_t size    = n_cols_indices * n_rows;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(idx_t idx) {
    idx_t row                   = idx % n_rows;
    idx_t new_col               = idx / n_rows;
    idx_t old_col               = indices[new_col];
    out[new_col * n_rows + row] = in[old_col * n_rows + row];
  });
}

template <typename value_t, typename index_t>
void selectColsIf(raft::resources const& handle,
                  raft::device_matrix_view<value_t, index_t, col_major> in,
                  raft::device_vector_view<index_t, index_t> mask,
                  raft::device_matrix_view<value_t, index_t, col_major> out)
{
  auto stream     = raft::resource::get_cuda_stream(handle);
  auto in_n_cols  = in.extent(1);
  auto out_n_cols = out.extent(1);
  auto rangeVec   = raft::make_device_vector<index_t, index_t>(handle, in_n_cols);
  raft::linalg::range(rangeVec.data_handle(), in_n_cols, stream);
  raft::linalg::map(
    handle,
    raft::make_const_mdspan(mask),
    raft::make_const_mdspan(rangeVec.view()),
    rangeVec.view(),
    [] __device__(index_t mask_value, index_t idx) { return mask_value == 1 ? idx : -1; });
  thrust::sort(rmm::exec_policy(stream),
               rangeVec.data_handle(),
               rangeVec.data_handle() + rangeVec.size(),
               thrust::less<index_t>());
  selectCols(in.data_handle(),
             in.extent(0),
             in.extent(1),
             out.data_handle(),
             rangeVec.data_handle() + rangeVec.size() - out_n_cols,
             out_n_cols,
             stream);
}

/**
 * Reverse (if requested) the eigenvalues/eigenvectors and truncate columns to fit eigVectorTrunc.
 */
template <typename value_t, typename index_t>
void truncEig(
  raft::resources const& handle,
  raft::device_matrix_view<value_t, index_t, raft::col_major> eigVectorin,
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> eigVectorTrunc,
  raft::device_vector_view<value_t, index_t> eigLambda,
  bool largest)
{
  // The eigenvalues coming from cuSolver eig_dc are in ascending order.
  auto nrows = eigVectorin.extent(0);
  auto ncols = eigVectorin.extent(1);
  if (largest) {
    raft::matrix::col_reverse(handle, eigVectorin);
    raft::matrix::col_reverse(handle,
                              raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
                                eigLambda.data_handle(), 1, eigLambda.extent(0)));
  }
  if (eigVectorTrunc.has_value() && ncols > eigVectorTrunc->extent(1))
    raft::matrix::detail::truncZeroOrigin(eigVectorin.data_handle(),
                                          nrows,
                                          eigVectorTrunc->data_handle(),
                                          nrows,
                                          eigVectorTrunc->extent(1),
                                          raft::resource::get_cuda_stream(handle));
}

// C = A * B for sparse A (CSR) and dense B/C (col-major)
template <typename value_t, typename index_t>
void lobpcg_spmm(raft::resources const& handle,
                 raft::spectral::matrix::sparse_matrix_t<index_t, value_t> const& A,
                 raft::device_matrix_view<value_t, index_t, raft::col_major> B,
                 raft::device_matrix_view<value_t, index_t, raft::col_major> C,
                 bool transpose_a = false,
                 bool transpose_b = false)
{
  auto stream          = raft::resource::get_cuda_stream(handle);
  auto* A_values_      = const_cast<value_t*>(A.values_);
  auto* A_row_offsets_ = const_cast<index_t*>(A.row_offsets_);
  auto* A_col_indices_ = const_cast<index_t*>(A.col_indices_);
  cusparseSpMatDescr_t sparse_A;
  cusparseDnMatDescr_t dense_B;
  cusparseDnMatDescr_t dense_C;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &sparse_A, A.nrows_, A.ncols_, A.nnz_, A_row_offsets_, A_col_indices_, A_values_));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(&dense_B,
                                                              static_cast<int64_t>(B.extent(0)),
                                                              static_cast<int64_t>(B.extent(1)),
                                                              static_cast<int64_t>(B.extent(0)),
                                                              B.data_handle(),
                                                              CUSPARSE_ORDER_COL));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(&dense_C,
                                                              static_cast<int64_t>(C.extent(0)),
                                                              static_cast<int64_t>(C.extent(1)),
                                                              static_cast<int64_t>(C.extent(0)),
                                                              C.data_handle(),
                                                              CUSPARSE_ORDER_COL));
  value_t alpha    = 1;
  value_t beta     = 0;
  size_t buff_size = 0;
  auto opA         = transpose_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB         = transpose_b ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  raft::sparse::detail::cusparsespmm_bufferSize(raft::resource::get_cusparse_handle(handle),
                                                opA,
                                                opB,
                                                &alpha,
                                                sparse_A,
                                                dense_B,
                                                &beta,
                                                dense_C,
                                                CUSPARSE_SPMM_ALG_DEFAULT,
                                                &buff_size,
                                                stream);
  rmm::device_uvector<value_t> dev_buffer(buff_size / sizeof(value_t) + 1, stream);
  raft::sparse::detail::cusparsespmm(raft::resource::get_cusparse_handle(handle),
                                     opA,
                                     opB,
                                     &alpha,
                                     sparse_A,
                                     dense_B,
                                     &beta,
                                     dense_C,
                                     CUSPARSE_SPMM_ALG_DEFAULT,
                                     dev_buffer.data(),
                                     stream);

  cusparseDestroySpMat(sparse_A);
  cusparseDestroyDnMat(dense_B);
  cusparseDestroyDnMat(dense_C);
}

/**
 * Solve the linear equation A x = b, given the Cholesky factorization of A.
 * Operation is in-place: matrix X overwrites matrix B.
 */
template <typename value_t, typename index_t>
void cho_solve(raft::resources const& handle,
               raft::device_matrix_view<const value_t, index_t, raft::col_major> A,
               raft::device_matrix_view<value_t, index_t, raft::col_major> B,
               bool lower = true)
{
  auto stream           = raft::resource::get_cuda_stream(handle);
  auto lda              = A.extent(0);
  auto dim              = A.extent(0);
  cublasFillMode_t uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  rmm::device_uvector<int> info(1, stream);
  RAFT_CUSOLVER_TRY(
    raft::linalg::detail::cusolverDnpotrs(raft::resource::get_cusolver_dn_handle(handle),
                                          uplo,
                                          dim,
                                          B.extent(1),
                                          A.data_handle(),
                                          lda,
                                          B.data_handle(),
                                          dim,
                                          info.data(),
                                          stream));
}

template <typename value_t, typename index_t>
bool cholesky(raft::resources const& handle,
              raft::device_matrix_view<value_t, index_t, raft::col_major> P,
              bool lower = true)
{
  auto thrust_exec_policy = raft::resource::get_thrust_policy(handle);
  auto stream             = raft::resource::get_cuda_stream(handle);
  int Lwork               = 0;
  auto lda                = P.extent(0);
  auto dim                = P.extent(0);
  cublasFillMode_t uplo   = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  auto P_copy =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, P.extent(0), P.extent(1));
  raft::copy(P_copy.data_handle(), P.data_handle(), P.size(), stream);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
    raft::resource::get_cusolver_dn_handle(handle), uplo, dim, P_copy.data_handle(), lda, &Lwork));

  rmm::device_uvector<value_t> workspace_decomp(Lwork, stream);
  rmm::device_uvector<int> info(1, stream);
  RAFT_CUSOLVER_TRY(
    raft::linalg::detail::cusolverDnpotrf(raft::resource::get_cusolver_dn_handle(handle),
                                          uplo,
                                          dim,
                                          P_copy.data_handle(),
                                          lda,
                                          workspace_decomp.data(),
                                          Lwork,
                                          info.data(),
                                          stream));
  int info_h = 0;
  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  if (info_h != 0) {
    // factorization failed
    return false;
  }

  int h_hasnan = thrust::transform_reduce(thrust_exec_policy,
                                          P_copy.data_handle(),
                                          P_copy.data_handle() + P_copy.size(),
                                          isnan_test<value_t>(),
                                          0,
                                          thrust::plus<int>());

  if (h_hasnan != 0) return false;

  raft::matrix::fill(handle, P, value_t(0));
  if (lower) {
    raft::matrix::lower_triangular(handle, raft::make_const_mdspan(P_copy.view()), P);
  } else {
    raft::matrix::upper_triangular(handle, raft::make_const_mdspan(P_copy.view()), P);
  }
  return true;
}

template <typename value_t, typename index_t>
void inverse(raft::resources const& handle,
             raft::device_matrix_view<value_t, index_t, raft::col_major> P,
             raft::device_matrix_view<value_t, index_t, raft::col_major> Pinv,
             bool transposeP = false)
{
  auto stream             = raft::resource::get_cuda_stream(handle);
  int Lwork               = 0;
  int lda                 = static_cast<int>(P.extent(0));
  int dim                 = static_cast<int>(P.extent(0));
  int info_h              = 0;
  cublasOperation_t trans = transposeP ? CUBLAS_OP_T : CUBLAS_OP_N;
  raft::matrix::eye(handle, Pinv);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngetrf_bufferSize(
    raft::resource::get_cusolver_dn_handle(handle), dim, dim, P.data_handle(), lda, &Lwork));

  auto P_copy =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, P.extent(0), P.extent(1));
  raft::copy(P_copy.data_handle(), P.data_handle(), P.size(), stream);
  rmm::device_uvector<value_t> workspace_decomp(Lwork, stream);
  rmm::device_uvector<int> info(1, stream);
  rmm::device_uvector<int> ipiv(dim, stream);

  RAFT_CUSOLVER_TRY(
    raft::linalg::detail::cusolverDngetrf(raft::resource::get_cusolver_dn_handle(handle),
                                          dim,
                                          dim,
                                          P_copy.data_handle(),
                                          lda,
                                          workspace_decomp.data(),
                                          ipiv.data(),
                                          info.data(),
                                          stream));

  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in getrf, info=%d | expected=0", info_h);

  RAFT_CUSOLVER_TRY(
    raft::linalg::detail::cusolverDngetrs(raft::resource::get_cusolver_dn_handle(handle),
                                          trans,
                                          dim,
                                          dim,
                                          P_copy.data_handle(),
                                          lda,
                                          ipiv.data(),
                                          Pinv.data_handle(),
                                          lda,
                                          info.data(),
                                          stream));

  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in getrs, info=%d | expected=0", info_h);
}

/**
 * Helper function for converting a generalized eigenvalue problem
 * A(X) = lambda(B(X)) to a standard eigenvalue problem via Cholesky.
 */
// Wrapper around the lower-level eigDC that avoids the signed/unsigned size comparison in the
// public raft::linalg::eig_dc which fails to compile under -Werror=sign-compare for signed
// IndexType.
template <typename value_t, typename index_t>
void eig_dc_u32(raft::resources const& handle,
                raft::device_matrix_view<const value_t, index_t, raft::col_major> in,
                raft::device_matrix_view<value_t, index_t, raft::col_major> eig_vectors,
                raft::device_vector_view<value_t, index_t> eig_vals)
{
  // Clear any stale CUDA error left over from earlier work in the process so
  // the cudaGetLastError() check inside eigDC does not pick it up.
  cudaGetLastError();
  raft::linalg::detail::eigDC(handle,
                              in.data_handle(),
                              static_cast<std::size_t>(in.extent(0)),
                              static_cast<std::size_t>(in.extent(1)),
                              eig_vectors.data_handle(),
                              eig_vals.data_handle(),
                              raft::resource::get_cuda_stream(handle));
}

// Symmetrize a square col-major matrix in place: M <- 0.5 * (M + M^T).
// This helps Cholesky on Gram matrices that lost symmetry from accumulated
// floating-point error during gemm reductions.
template <typename value_t, typename index_t>
void symmetrize(raft::resources const& handle,
                raft::device_matrix_view<value_t, index_t, raft::col_major> M)
{
  auto stream    = raft::resource::get_cuda_stream(handle);
  auto dim       = M.extent(0);
  auto policy    = raft::resource::get_thrust_policy(handle);
  value_t* data  = M.data_handle();
  index_t stride = static_cast<index_t>(dim);
  thrust::for_each(policy,
                   thrust::counting_iterator<index_t>(0),
                   thrust::counting_iterator<index_t>(dim * dim),
                   [data, stride, dim] __device__(index_t k) {
                     index_t i = k % stride;
                     index_t j = k / stride;
                     if (i < j) {
                       value_t avg = value_t(0.5) * (data[i + j * stride] + data[j + i * stride]);
                       data[i + j * stride] = avg;
                       data[j + i * stride] = avg;
                     }
                   });
}

// Returns true if every element in `v` is finite (not NaN, not +/-Inf).
template <typename value_t, typename index_t>
bool all_finite(raft::resources const& handle, raft::device_vector_view<value_t, index_t> v)
{
  auto policy = raft::resource::get_thrust_policy(handle);
  auto n      = v.extent(0);
  bool ok     = thrust::transform_reduce(
    policy,
    thrust::counting_iterator<index_t>(0),
    thrust::counting_iterator<index_t>(n),
    [data = v.data_handle()] __device__(index_t i) -> bool {
      value_t x = data[i];
      return (x == x) && (x < std::numeric_limits<value_t>::infinity()) &&
             (x > -std::numeric_limits<value_t>::infinity());
    },
    true,
    thrust::logical_and<bool>());
  return ok;
}

// Examines the diagonal of an in-place Cholesky factor (lower-triangular in
// col-major) and returns true when the smallest |L_ii| is large enough relative
// to the largest |L_ii| that subsequent inverses won't blow up.  cusolver's
// Cholesky reports success even for matrices that are numerically singular at
// double precision, so we screen for that here.
template <typename value_t, typename index_t>
bool cholesky_pivots_healthy(raft::resources const& handle,
                             raft::device_matrix_view<value_t, index_t, raft::col_major> L,
                             value_t rel_tol)
{
  auto policy    = raft::resource::get_thrust_policy(handle);
  auto dim       = L.extent(0);
  index_t stride = static_cast<index_t>(dim);
  value_t max_d  = thrust::transform_reduce(
    policy,
    thrust::counting_iterator<index_t>(0),
    thrust::counting_iterator<index_t>(dim),
    [data = L.data_handle(), stride] __device__(index_t i) -> value_t {
      value_t x = data[i + i * stride];
      return x < value_t(0) ? -x : x;
    },
    value_t(0),
    thrust::maximum<value_t>());
  if (!(max_d > value_t(0))) return false;
  value_t min_d = thrust::transform_reduce(
    policy,
    thrust::counting_iterator<index_t>(0),
    thrust::counting_iterator<index_t>(dim),
    [data = L.data_handle(), stride] __device__(index_t i) -> value_t {
      value_t x = data[i + i * stride];
      return x < value_t(0) ? -x : x;
    },
    max_d,
    thrust::minimum<value_t>());
  return min_d > rel_tol * max_d;
}

// Returns the maximum |M_ij| over all entries.
template <typename value_t, typename index_t>
value_t abs_max(raft::resources const& handle,
                raft::device_matrix_view<value_t, index_t, raft::col_major> M)
{
  auto policy = raft::resource::get_thrust_policy(handle);
  auto n      = M.size();
  return thrust::transform_reduce(
    policy,
    thrust::counting_iterator<index_t>(0),
    thrust::counting_iterator<index_t>(static_cast<index_t>(n)),
    [data = M.data_handle()] __device__(index_t i) -> value_t {
      value_t x = data[i];
      return x < value_t(0) ? -x : x;
    },
    value_t(0),
    thrust::maximum<value_t>());
}

template <typename value_t, typename index_t>
bool eigh(raft::resources const& handle,
          raft::device_matrix_view<value_t, index_t, raft::col_major> A,
          std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> B_opt,
          raft::device_matrix_view<value_t, index_t, raft::col_major> eigVecs,
          raft::device_vector_view<value_t, index_t> eigVals)
{
  auto dim    = A.extent(0);
  auto stream = raft::resource::get_cuda_stream(handle);
  // eig_dc takes a const input view; make a non-aliased copy to keep A intact.
  auto A_copy = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  raft::copy(A_copy.data_handle(), A.data_handle(), A.size(), stream);
  // Force exact symmetry; gemm-built Gram matrices often have tiny asymmetry
  // that can push Cholesky over the edge.
  symmetrize<value_t, index_t>(handle, A_copy.view());
  if (!B_opt.has_value()) {
    eig_dc_u32<value_t, index_t>(handle, raft::make_const_mdspan(A_copy.view()), eigVecs, eigVals);
    return all_finite<value_t, index_t>(handle, eigVals);
  }
  auto RTi     = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto Ri      = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto R       = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto F       = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto B_local = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  raft::copy(B_local.data_handle(), B_opt->data_handle(), B_opt->size(), stream);
  symmetrize<value_t, index_t>(handle, B_local.view());

  bool cho_success = cholesky(handle, B_local.view(), true);
  if (!cho_success) { return false; }
  // Numerical Cholesky may succeed on essentially-singular matrices, leaving
  // tiny diagonal entries that cause L^{-1} to blow up.  Reject any factor
  // whose smallest pivot is too small relative to the largest.
  value_t pivot_rel_tol = std::is_same_v<value_t, float> ? value_t(1e-5) : value_t(1e-10);
  if (!cholesky_pivots_healthy<value_t, index_t>(handle, B_local.view(), pivot_rel_tol)) {
    return false;
  }

  // After cholesky with `lower=true`, B_local holds L (lower-triangular factor).
  // R = L^T (upper-triangular).
  raft::linalg::transpose(handle, B_local.view(), R.view());

  // Reduce  A x = lambda B x  to a standard symmetric eigenproblem:
  //   F y = lambda y,  with  F = L^{-1} A L^{-T},   x = L^{-T} y.
  inverse(handle, R.view(), Ri.view(), true);    // Ri  = R^{-T} = L^{-1}
  inverse(handle, R.view(), RTi.view(), false);  // RTi = R^{-1} = L^{-T}

  auto ARTi = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  raft::linalg::gemm(handle, A_copy.view(), RTi.view(), ARTi.view());
  raft::linalg::gemm(handle, Ri.view(), ARTi.view(), F.view());
  // F should be symmetric in exact arithmetic; force it here as well.
  symmetrize<value_t, index_t>(handle, F.view());

  auto Fvecs = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  eig_dc_u32<value_t, index_t>(handle, raft::make_const_mdspan(F.view()), Fvecs.view(), eigVals);
  raft::linalg::gemm(handle, RTi.view(), Fvecs.view(), eigVecs);
  if (!all_finite<value_t, index_t>(handle, eigVals)) return false;
  // Sanity-check: in the generalized eigenproblem A x = lambda B x with B SPD,
  // every Rayleigh quotient (x^T A x)/(x^T B x) is bounded by |A|_max /
  // sigma_min(B).  We don't know sigma_min(B), but if any eigenvalue is many
  // orders of magnitude larger than |A|_max, the inverse of L has clearly
  // blown up due to a near-singular B, and the result is unusable.
  value_t maxA = abs_max<value_t, index_t>(handle, A_copy.view());
  value_t maxL = thrust::transform_reduce(
    raft::resource::get_thrust_policy(handle),
    thrust::counting_iterator<index_t>(0),
    thrust::counting_iterator<index_t>(static_cast<index_t>(dim)),
    [data = eigVals.data_handle()] __device__(index_t i) -> value_t {
      value_t x = data[i];
      return x < value_t(0) ? -x : x;
    },
    value_t(0),
    thrust::maximum<value_t>());
  value_t blowup_factor = std::is_same_v<value_t, float> ? value_t(1e6) : value_t(1e10);
  if (maxA > value_t(0) && maxL > blowup_factor * maxA) return false;
  return true;
}

/**
 * B-orthonormalize the given block vector using Cholesky.
 *
 * @param[inout] V dense matrix to normalize (col-major)
 * @param[inout] BV dense matrix.  Either filled internally (bv_is_empty=true)
 *               or used as input (bv_is_empty=false).
 * @param[in] B_opt optional sparse matrix used in the inner product.
 * @param[out] VBV_opt optional inverse of V^T B V (size kxk).
 * @param[out] V_max_opt optional per-column max of V used for scaling.
 * @return true on success.
 */
template <typename value_t, typename index_t>
bool b_orthonormalize(
  raft::resources const& handle,
  raft::device_matrix_view<value_t, index_t, raft::col_major> V,
  raft::device_matrix_view<value_t, index_t, raft::col_major> BV,
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B_opt     = std::nullopt,
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> VBV_opt = std::nullopt,
  std::optional<raft::device_vector_view<value_t, index_t>> V_max_opt                = std::nullopt,
  bool bv_is_empty                                                                   = true)
{
  auto stream        = raft::resource::get_cuda_stream(handle);
  auto V_max_buffer  = rmm::device_uvector<value_t>(0, stream);
  value_t* V_max_ptr = nullptr;
  if (!V_max_opt) {
    V_max_buffer.resize(V.extent(1), stream);
    V_max_ptr = V_max_buffer.data();
  } else {
    V_max_ptr = V_max_opt.value().data_handle();
  }
  auto V_max = raft::make_device_vector_view<value_t, index_t>(V_max_ptr, V.extent(1));

  // Coalesced reduction: V_max[j] = max_i V[i,j]  (col-major, reduce along columns).
  // Clear stale CUDA errors from prior work so the launch-time error check inside
  // coalescedReduction does not surface unrelated failures from earlier calls.
  cudaGetLastError();
  raft::linalg::reduce<false, false>(V_max.data_handle(),
                                     V.data_handle(),
                                     static_cast<index_t>(V.extent(1)),
                                     static_cast<index_t>(V.extent(0)),
                                     value_t(0),
                                     stream,
                                     false,
                                     raft::identity_op(),
                                     MaxOp<value_t>{});
  auto V_max_const =
    raft::make_device_vector_view<const value_t, index_t>(V_max.data_handle(), V_max.extent(0));
  raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(handle, V, V_max_const);

  if (!bv_is_empty) {
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(handle, BV, V_max_const);
  } else {
    if (B_opt)
      lobpcg_spmm(handle, B_opt.value(), V, BV);
    else
      raft::copy(BV.data_handle(), V.data_handle(), V.size(), stream);
  }
  auto VBV_buffer  = rmm::device_uvector<value_t>(0, stream);
  value_t* VBV_ptr = nullptr;
  if (!VBV_opt) {
    VBV_buffer.resize(V.extent(1) * V.extent(1), stream);
    VBV_ptr = VBV_buffer.data();
  } else {
    VBV_ptr = VBV_opt.value().data_handle();
  }
  auto VBV = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
    VBV_ptr, V.extent(1), V.extent(1));
  auto VBVBuffer = raft::make_device_matrix<value_t, index_t, raft::col_major>(
    handle, VBV.extent(0), VBV.extent(1));
  auto VT = make_transpose_layout_view(V);

  raft::linalg::gemm(handle, VT, BV, VBV);
  bool cholesky_success = cholesky(handle, VBV, false);
  if (!cholesky_success) { return cholesky_success; }

  inverse(handle, VBV, VBVBuffer.view());
  raft::copy(VBV.data_handle(), VBVBuffer.data_handle(), VBV.size(), stream);
  // V <- V * VBV^{-1}; out-of-place to avoid aliasing in cuBLAS.
  auto V_tmp =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, V.extent(0), V.extent(1));
  raft::linalg::gemm(handle, V, VBV, V_tmp.view());
  raft::copy(V.data_handle(), V_tmp.data_handle(), V.size(), stream);
  if (B_opt) {
    auto BV_tmp = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, BV.extent(0), BV.extent(1));
    raft::linalg::gemm(handle, BV, VBV, BV_tmp.view());
    raft::copy(BV.data_handle(), BV_tmp.data_handle(), BV.size(), stream);
  }
  return true;
}

template <typename value_t, typename index_t>
void lobpcg(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,
  raft::device_matrix_view<value_t, index_t, raft::col_major> X,
  raft::device_vector_view<value_t, index_t> W,
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B_opt   = std::nullopt,
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> M_opt   = std::nullopt,
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> Y_opt = std::nullopt,
  value_t tol                                                                      = 0,
  std::int32_t max_iter                                                            = 20,
  bool largest                                                                     = true,
  int verbosityLevel                                                               = 0)
{
  cudaStream_t stream     = raft::resource::get_cuda_stream(handle);
  auto thrust_exec_policy = raft::resource::get_thrust_policy(handle);
  auto n                  = X.extent(0);
  auto size_x             = X.extent(1);

  if (tol <= 0) { tol = raft::mySqrt(value_t(1e-15)) * n; }

  auto BX     = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  auto BXView = BX.view();
  b_orthonormalize(handle, X, BXView, B_opt);

  // Compute the initial Ritz vectors: solve the standard eigenproblem on the subspace.
  auto AX = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  lobpcg_spmm(handle, A, X, AX.view());
  auto gramXAX =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
  auto XTRowView = make_transpose_layout_view(X);
  raft::linalg::gemm(handle, XTRowView, AX.view(), gramXAX.view());
  // Conservative outer scale used to detect runaway eigenvalues later.
  // |gramXAX|_max with X B-orthonormal gives a Rayleigh-quotient sample that
  // upper-bounds the magnitude of the eigenvalues we should be returning.
  value_t outer_lambda_scale = abs_max<value_t, index_t>(handle, gramXAX.view());
  if (!(outer_lambda_scale > value_t(0))) outer_lambda_scale = value_t(1);
  auto eigVectorBuffer = rmm::device_uvector<value_t>(size_x * size_x, stream);
  auto eigVectorView   = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
    eigVectorBuffer.data(), size_x, size_x);
  auto eigLambda = raft::make_device_vector<value_t, index_t>(handle, size_x);
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> empty_matrix_opt =
    std::nullopt;
  eigh(handle, gramXAX.view(), empty_matrix_opt, eigVectorView, eigLambda.view());

  truncEig(handle, eigVectorView, empty_matrix_opt, eigLambda.view(), largest);

  // X, AX, BX <- X * eigVectors  (use scratch to avoid in-place alias).
  {
    auto tmp = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
    raft::linalg::gemm(handle, X, eigVectorView, tmp.view());
    raft::copy(X.data_handle(), tmp.data_handle(), tmp.size(), stream);
    raft::linalg::gemm(handle, AX.view(), eigVectorView, tmp.view());
    raft::copy(AX.data_handle(), tmp.data_handle(), tmp.size(), stream);
    if (B_opt) {
      raft::linalg::gemm(handle, BXView, eigVectorView, tmp.view());
      raft::copy(BXView.data_handle(), tmp.data_handle(), tmp.size(), stream);
    }
  }

  // Active mask -- 1 if residual exceeds tol, 0 otherwise.
  auto active_mask       = raft::make_device_vector<index_t, index_t>(handle, size_x);
  auto previousBlockSize = size_x;

  auto ident = rmm::device_uvector<value_t>(size_x * size_x, stream);
  auto identView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(ident.data(), size_x, size_x);
  raft::matrix::eye(handle, identView);

  auto Pbuffer  = rmm::device_uvector<value_t>(0, stream);
  auto APbuffer = rmm::device_uvector<value_t>(0, stream);
  auto BPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto PView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), 0, 0);
  auto APView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(APbuffer.data(), 0, 0);
  auto BPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(BPbuffer.data(), 0, 0);
  auto activePbuffer  = rmm::device_uvector<value_t>(0, stream);
  auto activeAPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto activeBPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto activePView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activePbuffer.data(), 0, 0);
  auto activeAPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activeAPbuffer.data(), 0, 0);
  auto activeBPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activeBPbuffer.data(), 0, 0);
  auto R = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);

  auto aux = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  auto residual_norms           = raft::make_device_vector<value_t, index_t>(handle, size_x);
  std::int32_t iteration_number = -1;
  bool restart                  = true;
  bool explicitGramFlag         = false;

  // Track the best (X, eigLambda) seen so far by residual.  LOBPCG is not
  // monotone in floating point: once the inner Gram matrix becomes
  // ill-conditioned, later iterations can return wildly incorrect Ritz values.
  // We therefore snapshot the iterate whose residual was smallest and return
  // that one if the iteration breaks down.
  auto best_X = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  auto best_eigLambda = raft::make_device_vector<value_t, index_t>(handle, size_x);
  raft::copy(best_X.data_handle(), X.data_handle(), X.size(), stream);
  raft::copy(best_eigLambda.data_handle(), eigLambda.data_handle(), size_x, stream);
  value_t best_residual = std::numeric_limits<value_t>::infinity();
  bool have_best        = false;
  while (iteration_number < max_iter + 1) {
    iteration_number += 1;
    if (B_opt) {
      raft::matrix::copy(handle, raft::make_const_mdspan(BXView), aux.view());
    } else {
      raft::matrix::copy(handle,
                         raft::make_device_matrix_view<const value_t, index_t, raft::col_major>(
                           X.data_handle(), X.extent(0), X.extent(1)),
                         aux.view());
    }
    {
      auto eig_const = raft::make_device_vector_view<const value_t, index_t>(
        eigLambda.data_handle(), eigLambda.extent(0));
      raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_ROWS>(handle, aux.view(), eig_const);
    }

    raft::linalg::subtract(
      handle, raft::make_const_mdspan(AX.view()), raft::make_const_mdspan(aux.view()), R.view());

    raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_COLUMNS>(
      handle, raft::make_const_mdspan(R.view()), residual_norms.view(), raft::sqrt_op());

    // Snapshot the current iterate when its residual is the best seen so far.
    // residual_norms at iteration K reflects how well eigLambda from
    // iteration (K-1) satisfies the eigenproblem, so we save the matching
    // (X, eigLambda) before they are overwritten further down.
    {
      value_t* rn_max_ptr =
        thrust::max_element(thrust_exec_policy,
                            residual_norms.data_handle(),
                            residual_norms.data_handle() + residual_norms.size());
      value_t rn_max = 0;
      raft::copy(&rn_max, rn_max_ptr, 1, stream);
      raft::resource::sync_stream(handle);
      if (rn_max == rn_max && rn_max < best_residual) {
        best_residual = rn_max;
        raft::copy(best_X.data_handle(), X.data_handle(), X.size(), stream);
        raft::copy(best_eigLambda.data_handle(), eigLambda.data_handle(), size_x, stream);
        have_best = true;
      }
    }

    // active_mask[i] = (residual_norms[i] > tol) ? 1 : 0
    raft::linalg::unary_op(handle,
                           raft::make_const_mdspan(residual_norms.view()),
                           active_mask.view(),
                           [tol] __device__(value_t rn) -> index_t { return rn > tol ? 1 : 0; });
    if (verbosityLevel > 2) {
      print_device_vector("active_mask", active_mask.data_handle(), active_mask.size(), std::cout);
    }
    index_t currentBlockSize = thrust::count_if(thrust::cuda::par.on(stream),
                                                active_mask.data_handle(),
                                                active_mask.data_handle() + active_mask.size(),
                                                [] __device__(index_t v) { return v > 0; });
    raft::resource::sync_stream(handle);
    if (currentBlockSize != previousBlockSize) {
      previousBlockSize = currentBlockSize;
      ident.resize(currentBlockSize * currentBlockSize, stream);
      identView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        ident.data(), currentBlockSize, currentBlockSize);
      raft::matrix::eye(handle, identView);
    }

    if (currentBlockSize == 0) break;
    if (verbosityLevel > 0) {
      printf("Iteration: %i\n", iteration_number);
      printf("current block size: %d\n", static_cast<int>(currentBlockSize));
      raft::matrix::print_separators ps{};
      printf("lambda:\n");
      raft::matrix::print(handle,
                          raft::make_device_matrix_view<const value_t, index_t, col_major>(
                            eigLambda.data_handle(), 1, eigLambda.extent(0)),
                          ps);
      printf("residual norms:\n");
      raft::matrix::print(handle,
                          raft::make_device_matrix_view<const value_t, index_t, col_major>(
                            residual_norms.data_handle(), 1, residual_norms.extent(0)),
                          ps);
    }
    auto activeR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, currentBlockSize);

    selectColsIf(handle, R.view(), active_mask.view(), activeR.view());

    if (iteration_number > 0) {
      activePbuffer.resize(n * currentBlockSize, stream);
      activeAPbuffer.resize(n * currentBlockSize, stream);
      activeBPbuffer.resize(n * currentBlockSize, stream);
      activePView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        activePbuffer.data(), n, currentBlockSize);
      activeAPView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        activeAPbuffer.data(), n, currentBlockSize);
      if (PView.extent(0) > 0 && PView.extent(1) > 0) {
        selectColsIf(handle, PView, active_mask.view(), activePView);
        selectColsIf(handle, APView, active_mask.view(), activeAPView);
      }
      if (B_opt.has_value()) {
        activeBPView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
          activeBPbuffer.data(), n, currentBlockSize);
        if (BPView.extent(0) > 0 && BPView.extent(1) > 0) {
          selectColsIf(handle, BPView, active_mask.view(), activeBPView);
        }
      }
    }
    if (M_opt.has_value()) {
      // Apply preconditioner T to the active residuals.
      auto MRtemp = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, R.extent(0), currentBlockSize);
      lobpcg_spmm(handle, M_opt.value(), activeR.view(), MRtemp.view());
      raft::copy(activeR.data_handle(), MRtemp.data_handle(), MRtemp.size(), stream);
    }
    // Constraints (Y) not supported in this first cut.

    // B-orthogonalize the preconditioned residuals to X.
    if (B_opt.has_value()) {
      auto BXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, BX.extent(1), activeR.extent(1));
      auto XBXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(0), BXTR.extent(1));

      raft::linalg::gemm(
        handle, make_transpose_layout_view(BX.view()), activeR.view(), BXTR.view());
      raft::linalg::gemm(handle, X, BXTR.view(), XBXTR.view());
      raft::linalg::subtract(handle,
                             raft::make_const_mdspan(activeR.view()),
                             raft::make_const_mdspan(XBXTR.view()),
                             activeR.view());
    } else {
      auto XTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(1), activeR.extent(1));
      auto XXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(0), XTR.extent(1));
      raft::linalg::gemm(handle, XTRowView, activeR.view(), XTR.view());
      raft::linalg::gemm(handle, X, XTR.view(), XXTR.view());
      raft::linalg::subtract(handle,
                             raft::make_const_mdspan(activeR.view()),
                             raft::make_const_mdspan(XXTR.view()),
                             activeR.view());
    }
    // B-orthonormalize the preconditioned residuals.
    auto activeBR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, activeR.extent(0), activeR.extent(1));
    auto activeBRView = activeBR.view();
    b_orthonormalize(handle, activeR.view(), activeBRView, B_opt);

    auto activeAR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, activeR.extent(1));
    lobpcg_spmm(handle, A, activeR.view(), activeAR.view());

    if (iteration_number > 0 && activePView.extent(1) > 0) {
      auto invR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, activePView.extent(1), activePView.extent(1));
      auto normal = raft::make_device_vector<value_t, index_t>(handle, activePView.extent(1));
      bool b_orth_success = true;
      if (!B_opt.has_value()) {
        auto BP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
          handle, activePView.extent(0), activePView.extent(1));
        b_orth_success = b_orthonormalize(handle,
                                          activePView,
                                          BP.view(),
                                          B_opt,
                                          std::make_optional(invR.view()),
                                          std::make_optional(normal.view()));
      } else {
        b_orth_success = b_orthonormalize(handle,
                                          activePView,
                                          activeBPView,
                                          B_opt,
                                          std::make_optional(invR.view()),
                                          std::make_optional(normal.view()),
                                          false);
      }
      if (!b_orth_success) {
        restart = true;
      } else {
        auto normal_const = raft::make_device_vector_view<const value_t, index_t>(
          normal.data_handle(), normal.extent(0));
        raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
          handle, activeAPView, normal_const);
        auto activeAPView_tmp = raft::make_device_matrix<value_t, index_t, raft::col_major>(
          handle, activeAPView.extent(0), activeAPView.extent(1));
        raft::linalg::gemm(handle, activeAPView, invR.view(), activeAPView_tmp.view());
        raft::copy(
          activeAPView.data_handle(), activeAPView_tmp.data_handle(), activeAPView.size(), stream);
        restart = false;
      }

      value_t myeps = std::is_same_v<value_t, float> ? value_t(1e-4) : value_t(1e-8);
      if (!explicitGramFlag) {
        value_t* residual_norms_max_elem =
          thrust::max_element(thrust_exec_policy,
                              residual_norms.data_handle(),
                              residual_norms.data_handle() + residual_norms.size());
        value_t residual_norms_max = 0;
        raft::copy(&residual_norms_max, residual_norms_max_elem, 1, stream);
        raft::resource::sync_stream(handle);
        explicitGramFlag = residual_norms_max > myeps;
      }

      if (!B_opt.has_value()) {
        BXView       = X;
        activeBRView = activeR.view();
        if (!restart) activeBPView = activePView;
      }
    } else {
      restart = true;
      if (!B_opt.has_value()) {
        BXView       = X;
        activeBRView = activeR.view();
      }
    }
    // Common submatrices.
    auto gramXAR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, currentBlockSize);
    auto gramRAR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBX =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
    auto gramRBR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, currentBlockSize);
    raft::linalg::gemm(handle, XTRowView, activeAR.view(), gramXAR.view());

    raft::linalg::gemm(
      handle, make_transpose_layout_view(activeR.view()), activeAR.view(), gramRAR.view());

    auto device_half = raft::make_device_scalar<value_t>(handle, value_t(0.5));
    if (explicitGramFlag) {
      // Symmetrize gramRAR: gramRAR <- 0.5 * (gramRAR + gramRAR^T).
      auto gramRAR_tmp = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, currentBlockSize, currentBlockSize);
      raft::copy(gramRAR_tmp.data_handle(), gramRAR.data_handle(), gramRAR.size(), stream);
      raft::linalg::gemm(handle,
                         make_transpose_layout_view(gramRAR.view()),
                         identView,
                         gramRAR_tmp.view(),
                         std::make_optional(device_half.view()),
                         std::make_optional(device_half.view()));
      raft::copy(gramRAR.data_handle(), gramRAR_tmp.data_handle(), gramRAR.size(), stream);

      raft::linalg::gemm(handle, XTRowView, AX.view(), gramXAX.view());
      auto gramXAX_tmp =
        raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
      raft::copy(gramXAX_tmp.data_handle(), gramXAX.data_handle(), gramXAX.size(), stream);
      raft::linalg::gemm(handle,
                         make_transpose_layout_view(gramXAX.view()),
                         raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
                           ident.data(), size_x, size_x),
                         gramXAX_tmp.view(),
                         std::make_optional(device_half.view()),
                         std::make_optional(device_half.view()));
      raft::copy(gramXAX.data_handle(), gramXAX_tmp.data_handle(), gramXAX.size(), stream);

      raft::linalg::gemm(handle, XTRowView, BXView, gramXBX.view());
      raft::linalg::gemm(
        handle, make_transpose_layout_view(activeR.view()), activeBRView, gramRBR.view());
      raft::linalg::gemm(handle, XTRowView, activeBRView, gramXBR.view());
    } else {
      raft::matrix::fill(handle, gramXAX.view(), value_t(0));
      raft::matrix::set_diagonal(handle,
                                 raft::make_device_vector_view<const value_t, index_t>(
                                   eigLambda.data_handle(), eigLambda.extent(0)),
                                 gramXAX.view());

      raft::matrix::eye(handle, gramXBX.view());
      raft::matrix::eye(handle, gramRBR.view());
      raft::matrix::fill(handle, gramXBR.view(), value_t(0));
    }
    auto gramDim = gramXAX.extent(1) + gramXAR.extent(1) + currentBlockSize;
    auto gramA =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, gramDim, gramDim);
    auto gramB =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, gramDim, gramDim);
    auto gramAView     = gramA.view();
    auto gramBView     = gramB.view();
    auto eigLambdaTemp = raft::make_device_vector<value_t, index_t>(handle, gramDim);
    auto eigVectorTemp =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, gramDim, gramDim);
    auto eigLambdaTempView = eigLambdaTemp.view();
    auto eigVectorTempView = eigVectorTemp.view();
    auto gramXAP =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, currentBlockSize);
    auto gramRAP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramPAP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBP =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, currentBlockSize);
    auto gramRBP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramPBP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXAPT = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramXAP.extent(1), gramXAP.extent(0));
    auto gramXART = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramXAR.extent(1), gramXAR.extent(0));
    auto gramRAPT = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramRAP.extent(1), gramRAP.extent(0));
    auto gramXBPT = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramXBP.extent(1), gramXBP.extent(0));
    auto gramXBRT = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramXBR.extent(1), gramXBR.extent(0));
    auto gramRBPT = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, gramRBP.extent(1), gramRBP.extent(0));
    raft::linalg::transpose(handle, gramXAR.view(), gramXART.view());
    raft::linalg::transpose(handle, gramXBR.view(), gramXBRT.view());

    if (!restart) {
      raft::linalg::gemm(handle, XTRowView, activeAPView, gramXAP.view());
      raft::linalg::gemm(
        handle, make_transpose_layout_view(activeR.view()), activeAPView, gramRAP.view());
      raft::linalg::gemm(
        handle, make_transpose_layout_view(activePView), activeAPView, gramPAP.view());
      raft::linalg::gemm(handle, XTRowView, activeBPView, gramXBP.view());
      raft::linalg::gemm(
        handle, make_transpose_layout_view(activeR.view()), activeBPView, gramRBP.view());

      if (explicitGramFlag) {
        // Symmetrize gramPAP: gramPAP <- 0.5 * (gramPAP + gramPAP^T).
        auto gramPAP_tmp = raft::make_device_matrix<value_t, index_t, raft::col_major>(
          handle, currentBlockSize, currentBlockSize);
        raft::copy(gramPAP_tmp.data_handle(), gramPAP.data_handle(), gramPAP.size(), stream);
        raft::linalg::gemm(handle,
                           make_transpose_layout_view(gramPAP.view()),
                           identView,
                           gramPAP_tmp.view(),
                           std::make_optional(device_half.view()),
                           std::make_optional(device_half.view()));
        raft::copy(gramPAP.data_handle(), gramPAP_tmp.data_handle(), gramPAP.size(), stream);

        raft::linalg::gemm(
          handle, make_transpose_layout_view(activePView), activeBPView, gramPBP.view());
      } else {
        raft::matrix::eye(handle, gramPBP.view());
      }
      raft::linalg::transpose(handle, gramXAP.view(), gramXAPT.view());
      raft::linalg::transpose(handle, gramRAP.view(), gramRAPT.view());
      raft::linalg::transpose(handle, gramXBP.view(), gramXBPT.view());
      raft::linalg::transpose(handle, gramRBP.view(), gramRBPT.view());

      std::vector<raft::device_matrix_view<value_t, index_t, raft::col_major>> A_blocks = {
        gramXAX.view(),
        gramXAR.view(),
        gramXAP.view(),
        gramXART.view(),
        gramRAR.view(),
        gramRAP.view(),
        gramXAPT.view(),
        gramRAPT.view(),
        gramPAP.view()};
      std::vector<raft::device_matrix_view<value_t, index_t, raft::col_major>> B_blocks = {
        gramXBX.view(),
        gramXBR.view(),
        gramXBP.view(),
        gramXBRT.view(),
        gramRBR.view(),
        gramRBP.view(),
        gramXBPT.view(),
        gramRBPT.view(),
        gramPBP.view()};

      bmat(handle, gramAView, A_blocks, index_t(3));
      bmat(handle, gramBView, B_blocks, index_t(3));

      bool eig_sucess = eigh(
        handle, gramAView, std::make_optional(gramBView), eigVectorTempView, eigLambdaTempView);
      if (!eig_sucess) restart = true;
    }
    if (restart) {
      gramDim = gramXAX.extent(1) + gramXAR.extent(1);
      std::vector<raft::device_matrix_view<value_t, index_t, raft::col_major>> A_blocks = {
        gramXAX.view(), gramXAR.view(), gramXART.view(), gramRAR.view()};
      std::vector<raft::device_matrix_view<value_t, index_t, raft::col_major>> B_blocks = {
        gramXBX.view(), gramXBR.view(), gramXBRT.view(), gramRBR.view()};
      gramAView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        gramA.data_handle(), gramDim, gramDim);
      gramBView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        gramB.data_handle(), gramDim, gramDim);
      eigLambdaTempView =
        raft::make_device_vector_view<value_t, index_t>(eigLambdaTempView.data_handle(), gramDim);
      eigVectorTempView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        eigVectorTempView.data_handle(), gramDim, gramDim);
      bmat(handle, gramAView, A_blocks, index_t(2));
      bmat(handle, gramBView, B_blocks, index_t(2));
      bool eig_sucess = eigh(
        handle, gramAView, std::make_optional(gramBView), eigVectorTempView, eigLambdaTempView);
      if (!eig_sucess) {
        // Even the smaller restart eigh failed; the gramB became too
        // ill-conditioned to factor.  Stop and return the best eigenpairs so
        // far instead of throwing.
        if (verbosityLevel > 0) {
          printf("lobpcg: eigh failed at iteration %d; stopping early.\n",
                 static_cast<int>(iteration_number));
        }
        break;
      }
    }
    // Reject diverging Ritz values that can appear when the inner Gram matrix
    // is badly conditioned.  The valid range of eigenvalues for the original
    // problem is bounded by |gramXAX|_max at iteration 0; if the inner eigh
    // returned values many orders of magnitude beyond that, we are in a
    // runaway and should stop with the previous good estimate.
    {
      value_t maxLambda = thrust::transform_reduce(
        thrust_exec_policy,
        thrust::counting_iterator<index_t>(0),
        thrust::counting_iterator<index_t>(static_cast<index_t>(eigLambdaTempView.extent(0))),
        [data = eigLambdaTempView.data_handle()] __device__(index_t i) -> value_t {
          value_t x = data[i];
          return x < value_t(0) ? -x : x;
        },
        value_t(0),
        thrust::maximum<value_t>());
      value_t runaway_factor = std::is_same_v<value_t, float> ? value_t(1e6) : value_t(1e10);
      if (maxLambda > runaway_factor * outer_lambda_scale) {
        if (verbosityLevel > 0) {
          printf(
            "lobpcg: detected diverging inner eigenvalues (max|lambda|=%g vs scale=%g) at "
            "iteration %d; stopping early.\n",
            static_cast<double>(maxLambda),
            static_cast<double>(outer_lambda_scale),
            static_cast<int>(iteration_number));
        }
        break;
      }
    }
    eigVectorBuffer.resize(gramDim * size_x, stream);
    eigVectorView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
      eigVectorBuffer.data(), gramDim, size_x);
    truncEig(
      handle, eigVectorTempView, std::make_optional(eigVectorView), eigLambdaTempView, largest);
    raft::copy(eigLambda.data_handle(), eigLambdaTempView.data_handle(), size_x, stream);

    auto d_one = raft::make_device_scalar<value_t>(handle, value_t(1));
    auto one   = std::make_optional(d_one.view());
    auto eigBlockVectorX =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
    auto eigBlockVectorR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, currentBlockSize, size_x);
    auto eigBlockVectorP_rows =
      (gramDim > size_x + currentBlockSize) ? gramDim - (size_x + currentBlockSize) : index_t(0);
    auto eigBlockVectorP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, eigBlockVectorP_rows, size_x);
    auto pp  = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
    auto app = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
    if (B_opt.has_value()) {
      auto bpp = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
      raft::matrix::slice(handle,
                          raft::make_const_mdspan(eigVectorView),
                          eigBlockVectorX.view(),
                          raft::matrix::slice_coordinates<index_t>(0, 0, size_x, size_x));
      if (!restart) {
        raft::matrix::slice(
          handle,
          raft::make_const_mdspan(eigVectorView),
          eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, size_x + currentBlockSize, size_x));
        raft::matrix::slice(
          handle,
          raft::make_const_mdspan(eigVectorView),
          eigBlockVectorP.view(),
          raft::matrix::slice_coordinates<index_t>(size_x + currentBlockSize, 0, gramDim, size_x));
      } else {
        raft::matrix::slice(handle,
                            raft::make_const_mdspan(eigVectorView),
                            eigBlockVectorR.view(),
                            raft::matrix::slice_coordinates<index_t>(size_x, 0, gramDim, size_x));
      }

      raft::linalg::gemm(handle, activeR.view(), eigBlockVectorR.view(), pp.view());
      raft::linalg::gemm(handle, activeAR.view(), eigBlockVectorR.view(), app.view());
      raft::linalg::gemm(handle, activeBRView, eigBlockVectorR.view(), bpp.view());
      if (!restart) {
        raft::linalg::gemm(handle, activePView, eigBlockVectorP.view(), pp.view(), one, one);
        raft::linalg::gemm(handle, activeAPView, eigBlockVectorP.view(), app.view(), one, one);
        raft::linalg::gemm(handle, activeBPView, eigBlockVectorP.view(), bpp.view(), one, one);
      }
      Pbuffer.resize(n * size_x, stream);
      APbuffer.resize(n * size_x, stream);
      BPbuffer.resize(n * size_x, stream);
      PView =
        raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), n, size_x);
      APView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        APbuffer.data(), n, size_x);
      BPView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        BPbuffer.data(), n, size_x);

      raft::copy(PView.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(APView.data_handle(), app.data_handle(), app.size(), stream);
      raft::copy(BPView.data_handle(), bpp.data_handle(), bpp.size(), stream);

      raft::linalg::gemm(handle, X, eigBlockVectorX.view(), pp.view(), one, one);
      raft::linalg::gemm(handle, AX.view(), eigBlockVectorX.view(), app.view(), one, one);
      raft::linalg::gemm(handle, BXView, eigBlockVectorX.view(), bpp.view(), one, one);

      raft::copy(X.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(AX.data_handle(), app.data_handle(), app.size(), stream);
      raft::copy(BXView.data_handle(), bpp.data_handle(), bpp.size(), stream);
    } else {
      raft::matrix::slice(handle,
                          raft::make_const_mdspan(eigVectorView),
                          eigBlockVectorX.view(),
                          raft::matrix::slice_coordinates<index_t>(0, 0, size_x, size_x));
      if (!restart) {
        raft::matrix::slice(
          handle,
          raft::make_const_mdspan(eigVectorView),
          eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, size_x + currentBlockSize, size_x));
        raft::matrix::slice(
          handle,
          raft::make_const_mdspan(eigVectorView),
          eigBlockVectorP.view(),
          raft::matrix::slice_coordinates<index_t>(size_x + currentBlockSize, 0, gramDim, size_x));
      } else {
        raft::matrix::slice(handle,
                            raft::make_const_mdspan(eigVectorView),
                            eigBlockVectorR.view(),
                            raft::matrix::slice_coordinates<index_t>(size_x, 0, gramDim, size_x));
      }

      raft::linalg::gemm(handle, activeR.view(), eigBlockVectorR.view(), pp.view());
      raft::linalg::gemm(handle, activeAR.view(), eigBlockVectorR.view(), app.view());
      if (!restart) {
        raft::linalg::gemm(handle, activePView, eigBlockVectorP.view(), pp.view(), one, one);
        raft::linalg::gemm(handle, activeAPView, eigBlockVectorP.view(), app.view(), one, one);
      }
      Pbuffer.resize(n * size_x, stream);
      APbuffer.resize(n * size_x, stream);
      PView =
        raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), n, size_x);
      APView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        APbuffer.data(), n, size_x);

      raft::copy(PView.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(APView.data_handle(), app.data_handle(), app.size(), stream);

      raft::linalg::gemm(handle, X, eigBlockVectorX.view(), pp.view(), one, one);
      raft::linalg::gemm(handle, AX.view(), eigBlockVectorX.view(), app.view(), one, one);

      raft::copy(X.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(AX.data_handle(), app.data_handle(), app.size(), stream);
    }
  }

  // Final residual + copy eigenvalues to output.  Prefer the best-residual
  // iterate over the last one, since the last iterate may have diverged.
  if (have_best) {
    raft::copy(X.data_handle(), best_X.data_handle(), X.size(), stream);
    raft::copy(W.data_handle(), best_eigLambda.data_handle(), size_x, stream);
  } else {
    raft::copy(W.data_handle(), eigLambda.data_handle(), size_x, stream);
  }
}

}  // namespace raft::sparse::solver::detail
