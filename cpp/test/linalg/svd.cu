/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename data_t, typename idx_t>
struct SvdInputs {
  data_t tolerance;
  idx_t len;
  idx_t n_row;
  idx_t n_col;
  unsigned long long int seed;
};

template <typename data_t, typename idx_t>
::std::ostream& operator<<(::std::ostream& os, const SvdInputs<data_t, idx_t>& dims)
{
  return os;
}

template <typename data_t, typename idx_t>
class SvdTest : public ::testing::TestWithParam<SvdInputs<data_t, idx_t>> {
 public:
  SvdTest()
    : params(::testing::TestWithParam<SvdInputs<data_t, idx_t>>::GetParam()),
      stream(handle.get_stream()),
      data(params.len, stream),
      left_eig_vectors_qr(params.n_row * params.n_col, stream),
      right_eig_vectors_trans_qr(params.n_col * params.n_col, stream),
      sing_vals_qr(params.n_col, stream),
      left_eig_vectors_ref(params.n_row * params.n_col, stream),
      right_eig_vectors_ref(params.n_col * params.n_col, stream),
      sing_vals_ref(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    auto len = params.len;

    ASSERT(params.n_row == 3, "This test only supports nrows=3!");
    ASSERT(params.len == 6, "This test only supports len=6!");
    data_t data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    raft::update_device(data.data(), data_h, len, stream);

    auto left_evl  = params.n_row * params.n_col;
    auto right_evl = params.n_col * params.n_col;

    data_t left_eig_vectors_ref_h[] = {-0.308219, -0.906133, -0.289695, 0.488195, 0.110706, -0.865685};

    data_t right_eig_vectors_ref_h[] = {-0.638636, -0.769509, -0.769509, 0.638636};

    data_t sing_vals_ref_h[] = {7.065283, 1.040081};

    raft::update_device(left_eig_vectors_ref.data(), left_eig_vectors_ref_h, left_evl, stream);
    raft::update_device(right_eig_vectors_ref.data(), right_eig_vectors_ref_h, right_evl, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, params.n_col, stream);

    auto data_view = raft::make_device_matrix_view<const data_t, idx_t, raft::col_major>(
      data.data(), params.n_row, params.n_col);
    auto sing_vals_qr_view =
      raft::make_device_vector_view<data_t, idx_t>(sing_vals_qr.data(), params.n_col);
    std::optional<raft::device_matrix_view<data_t, idx_t, raft::col_major>> left_eig_vectors_qr_view =
      raft::make_device_matrix_view<data_t, idx_t, raft::col_major>(
        left_eig_vectors_qr.data(), params.n_row, params.n_col);
    std::optional<raft::device_matrix_view<data_t, idx_t, raft::col_major>>
      right_eig_vectors_trans_qr_view = raft::make_device_matrix_view<data_t, idx_t, raft::col_major>(
        right_eig_vectors_trans_qr.data(), params.n_col, params.n_col);

    svd_qr_transpose_right_vec(handle,
                               data_view,
                               sing_vals_qr_view,
                               left_eig_vectors_qr_view,
                               right_eig_vectors_trans_qr_view);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  SvdInputs<data_t, idx_t> params;
  rmm::device_uvector<data_t> data, left_eig_vectors_qr, right_eig_vectors_trans_qr, sing_vals_qr,
    left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref;
};

const std::vector<SvdInputs<float, std::int32_t>> inputsf2i32 = {{0.00001f, 3 * 2, 3, 2, 1234ULL}};
const std::vector<SvdInputs<float, std::uint64_t>> inputsf2ui64 = {{0.00001f, 3 * 2, 3, 2, 1234ULL}};
const std::vector<SvdInputs<double, std::int32_t>> inputsd2i32 = {{0.00001, 3 * 2, 3, 2, 1234ULL}};
const std::vector<SvdInputs<double, std::uint64_t>> inputsd2ui64 = {{0.00001, 3 * 2, 3, 2, 1234ULL}};

typedef SvdTest<float, std::int32_t> SvdTestValFi32;
TEST_P(SvdTestValFi32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}
typedef SvdTest<float, std::uint64_t> SvdTestValFui64;
TEST_P(SvdTestValFui64, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double, std::int32_t> SvdTestValDi32;
TEST_P(SvdTestValDi32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}
typedef SvdTest<double, std::uint64_t> SvdTestValDui64;
TEST_P(SvdTestValDui64, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef SvdTest<float, std::int32_t> SvdTestLeftVecFi32;
TEST_P(SvdTestLeftVecFi32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}
typedef SvdTest<float, std::uint64_t> SvdTestLeftVecFui64;
TEST_P(SvdTestLeftVecFui64, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double, std::int32_t> SvdTestLeftVecDi32;
TEST_P(SvdTestLeftVecDi32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}
typedef SvdTest<double, std::uint64_t> SvdTestLeftVecDui64;
TEST_P(SvdTestLeftVecDui64, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}
/*
typedef SvdTest<float> SvdTestRightVecF;
TEST_P(SvdTestRightVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestRightVecD;
TEST_P(SvdTestRightVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}
*/
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValFi32, ::testing::ValuesIn(inputsf2i32));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValFui64, ::testing::ValuesIn(inputsf2ui64));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValDi32, ::testing::ValuesIn(inputsd2i32));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValDui64, ::testing::ValuesIn(inputsd2ui64));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecFi32, ::testing::ValuesIn(inputsf2i32));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecFui64, ::testing::ValuesIn(inputsf2ui64));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecDi32, ::testing::ValuesIn(inputsd2i32));
INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecDui64, ::testing::ValuesIn(inputsd2ui64));

// INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecF,
// ::testing::ValuesIn(inputsf2));

// INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecD,
//::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
