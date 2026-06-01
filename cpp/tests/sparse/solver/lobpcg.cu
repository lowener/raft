/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../test_utils.cuh"
#include "../../test_utils.h"

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/init.cuh>
#include <raft/sparse/solver/detail/lobpcg.cuh>
#include <raft/sparse/solver/lobpcg.cuh>
#include <raft/sparse/solver/lobpcg_types.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace raft {
namespace sparse {

template <typename math_t, typename idx_t>
struct CSRMatrixVal {
  std::vector<idx_t> row_ind_ptr;
  std::vector<idx_t> row_ind;
  std::vector<math_t> values;
};

template <typename math_t, typename idx_t>
struct LOBPCGInputs {
  CSRMatrixVal<math_t, idx_t> matrix_a;
  std::vector<math_t> init_eigvecs;
  std::vector<math_t> exp_eigvals;
  std::vector<math_t> exp_eigvecs;
  idx_t n_components;
};

template <typename math_t, typename idx_t>
class LOBPCGTest : public ::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>> {
 public:
  LOBPCGTest()
    : params(::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      ind_a(params.matrix_a.row_ind.size(), stream),
      ind_ptr_a(params.matrix_a.row_ind_ptr.size(), stream),
      values_a(params.matrix_a.values.size(), stream),
      exp_eigvals(params.exp_eigvals.size(), stream),
      exp_eigvecs(params.exp_eigvecs.size(), stream),
      act_eigvals(params.n_components, stream),
      act_eigvecs(params.exp_eigvecs.size(), stream)
  {
  }

 protected:
  void SetUp() override
  {
    n_rows_a = params.matrix_a.row_ind_ptr.size() - 1;
    nnz_a    = params.matrix_a.values.size();
  }

  // Unit test
  void test_selectcolsif()
  {
    auto a = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 5, 8);
    auto c = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 5, 4);
    auto b = raft::make_device_vector<idx_t, idx_t>(handle, 8);
    raft::linalg::range(a.data_handle(), a.size(), stream);
    std::vector<idx_t> select_h{0, 1, 1, 1, 0, 0, 0, 1};
    raft::copy(b.data_handle(), select_h.data(), 8, stream);
    raft::sparse::solver::detail::selectColsIf(handle, a.view(), b.view(), c.view());
    std::vector<math_t> res(c.size());
    std::vector<math_t> expected{5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                 15, 16, 17, 18, 19, 35, 36, 37, 38, 39};
    raft::copy(res.data(), c.data_handle(), c.size(), stream);
    resource::sync_stream(handle);

    ASSERT_TRUE(hostVecMatch(expected, res, raft::CompareApprox<math_t>(0.0001)));
  }

  // Unit test
  void test_bmat()
  {
    auto total = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 6, 6);
    auto x1    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x2    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x3    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x4    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x5    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x6    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x7    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x8    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x9    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    raft::linalg::range(x1.data_handle(), 0, 4, stream);
    raft::linalg::range(x2.data_handle(), 4, 8, stream);
    raft::linalg::range(x3.data_handle(), 8, 12, stream);
    raft::linalg::range(x4.data_handle(), 12, 16, stream);
    raft::linalg::range(x5.data_handle(), 16, 20, stream);
    raft::linalg::range(x6.data_handle(), 20, 24, stream);
    raft::linalg::range(x7.data_handle(), 24, 28, stream);
    raft::linalg::range(x8.data_handle(), 28, 32, stream);
    raft::linalg::range(x9.data_handle(), 32, 36, stream);
    std::vector<raft::device_matrix_view<math_t, idx_t, col_major>> xs = {x1.view(),
                                                                          x2.view(),
                                                                          x3.view(),
                                                                          x4.view(),
                                                                          x5.view(),
                                                                          x6.view(),
                                                                          x7.view(),
                                                                          x8.view(),
                                                                          x9.view()};
    raft::sparse::solver::detail::bmat(handle, total.view(), xs, idx_t(3));
    std::vector<math_t> res(total.size());
    std::vector<math_t> expected{0, 1, 12, 13, 24, 25, 2,  3,  14, 15, 26, 27,
                                 4, 5, 16, 17, 28, 29, 6,  7,  18, 19, 30, 31,
                                 8, 9, 20, 21, 32, 33, 10, 11, 22, 23, 34, 35};
    raft::copy(res.data(), total.data_handle(), total.size(), stream);
    resource::sync_stream(handle);
    ASSERT_TRUE(hostVecMatch(expected, res, raft::CompareApprox<math_t>(0.0001)));
  }

  void test_eigh()
  {
    // Symmetric 2x2 matrix in col-major: [[2, 1], [1, 2]] -> eigvals {1, 3}.
    std::vector<math_t> in_cpu{math_t(2), math_t(1), math_t(1), math_t(2)};
    std::vector<math_t> lambda_cpu{math_t(1), math_t(3)};
    auto in_gpu     = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto lambda_gpu = raft::make_device_vector<math_t, idx_t>(handle, 2);
    auto vector_gpu = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> empty_matrix_opt =
      std::nullopt;

    raft::copy(in_gpu.data_handle(), in_cpu.data(), 4, stream);
    raft::sparse::solver::detail::eigh(
      handle, in_gpu.view(), empty_matrix_opt, vector_gpu.view(), lambda_gpu.view());

    ASSERT_TRUE(devArrMatchHost(lambda_cpu.data(),
                                lambda_gpu.data_handle(),
                                lambda_cpu.size(),
                                raft::CompareApprox<math_t>(0.0001),
                                stream));
    ASSERT_TRUE(devArrMatchHost(vector_cpu.data(),
                                vector_gpu.data_handle(),
                                vector_cpu.size(),
                                raft::CompareApprox<math_t>(0.0001),
                                stream));
  }

  void Run()
  {
    test_eigh();
    test_bmat();
    test_selectcolsif();

    raft::update_device(
      ind_a.data(), params.matrix_a.row_ind.data(), params.matrix_a.row_ind.size(), stream);
    raft::update_device(ind_ptr_a.data(),
                        params.matrix_a.row_ind_ptr.data(),
                        params.matrix_a.row_ind_ptr.size(),
                        stream);
    raft::update_device(
      values_a.data(), params.matrix_a.values.data(), params.matrix_a.values.size(), stream);
    raft::update_device(
      act_eigvecs.data(), params.init_eigvecs.data(), params.init_eigvecs.size(), stream);
    raft::update_device(
      exp_eigvals.data(), params.exp_eigvals.data(), params.exp_eigvals.size(), stream);
    raft::update_device(
      exp_eigvecs.data(), params.exp_eigvecs.data(), params.exp_eigvecs.size(), stream);

    auto csr_structure = raft::make_device_compressed_structure_view<idx_t, idx_t, idx_t>(
      ind_ptr_a.data(), ind_a.data(), n_rows_a, n_rows_a, nnz_a);
    auto csr_view = raft::make_device_csr_matrix_view<const math_t, idx_t, idx_t, idx_t>(
      values_a.data(), csr_structure);

    raft::sparse::solver::lobpcg_solver_config<math_t> config;
    config.tolerance       = math_t(1e-6);
    config.max_iterations  = 200;
    config.largest         = true;
    config.verbosity_level = 0;

    auto X = raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      act_eigvecs.data(), n_rows_a, params.n_components);
    auto W = raft::make_device_vector_view<math_t, idx_t>(act_eigvals.data(), params.n_components);

    raft::sparse::solver::lobpcg_compute_eigenpairs<idx_t, math_t, idx_t>(
      handle, config, csr_view, X, W);

    resource::sync_stream(handle);

    // Verify residual: ||A x_i - lambda_i x_i|| <= tol * (1 + |lambda_i|)
    std::vector<math_t> W_host(params.n_components);
    std::vector<math_t> X_host(n_rows_a * params.n_components);
    raft::copy(W_host.data(), act_eigvals.data(), W_host.size(), stream);
    raft::copy(X_host.data(), act_eigvecs.data(), X_host.size(), stream);
    resource::sync_stream(handle);

    // Build dense A on host from CSR data
    std::vector<math_t> A_host(n_rows_a * n_rows_a, math_t(0));
    for (idx_t r = 0; r < n_rows_a; ++r) {
      idx_t start = params.matrix_a.row_ind_ptr[r];
      idx_t end   = params.matrix_a.row_ind_ptr[r + 1];
      for (idx_t k = start; k < end; ++k) {
        idx_t c                  = params.matrix_a.row_ind[k];
        A_host[r + c * n_rows_a] = params.matrix_a.values[k];  // col-major
      }
    }

    for (idx_t j = 0; j < params.n_components; ++j) {
      double res2 = 0;
      for (idx_t r = 0; r < n_rows_a; ++r) {
        double ax = 0;
        for (idx_t c = 0; c < n_rows_a; ++c) {
          ax += static_cast<double>(A_host[r + c * n_rows_a]) *
                static_cast<double>(X_host[c + j * n_rows_a]);
        }
        double diff =
          ax - static_cast<double>(W_host[j]) * static_cast<double>(X_host[r + j * n_rows_a]);
        res2 += diff * diff;
      }
      double res = std::sqrt(res2);
      ASSERT_LE(res, 1e-2) << "Residual too large for eigenpair " << j << " (lambda=" << W_host[j]
                           << ")";
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  LOBPCGInputs<math_t, idx_t> params;
  idx_t n_rows_a, nnz_a;
  rmm::device_uvector<idx_t> ind_a, ind_ptr_a;
  rmm::device_uvector<math_t> values_a, exp_eigvals, exp_eigvecs, act_eigvals, act_eigvecs;
};

using LOBPCGTestF = LOBPCGTest<float, int>;
TEST_P(LOBPCGTestF, Result) { Run(); }

using LOBPCGTestD = LOBPCGTest<double, int>;
TEST_P(LOBPCGTestD, Result) { Run(); }

// Diagonal 6x6 matrix with distinct positive eigenvalues {1, 2, 3, 4, 5, 6}.
const std::vector<LOBPCGInputs<float, int>> lobpcg_inputs_f = {
  {{{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}},
   {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.1f, 0.2f},
   {6.f, 5.f},
   std::vector<float>(12, 0.f),
   2}};

const std::vector<LOBPCGInputs<double, int>> lobpcg_inputs_d = {
  {{{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5}, {1., 2., 3., 4., 5., 6.}},
   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2},
   {6., 5.},
   std::vector<double>(12, 0.),
   2}};

INSTANTIATE_TEST_CASE_P(LOBPCGTest, LOBPCGTestF, ::testing::ValuesIn(lobpcg_inputs_f));
INSTANTIATE_TEST_CASE_P(LOBPCGTest, LOBPCGTestD, ::testing::ValuesIn(lobpcg_inputs_d));

}  // namespace sparse
}  // namespace raft
