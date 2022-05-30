/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/dense.cuh>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace sparse {

template <typename value_idx, typename value_t>
struct GatherSparseInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  uint32_t map_length;
  unsigned long long int seed;
};

template <typename MatrixT, typename MapT>
class GatherTest : public ::testing::TestWithParam<GatherInputs> {
 protected:
  GatherTest() : stream(handle.get_stream()), d_in(0, stream), d_out_exp(0, stream), d_out_act(0, stream), d_map(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<GatherInputs>::GetParam();
    raft::random::RngState r(params.seed);
    raft::random::RngState r_int(params.seed);

    uint32_t nrows      = params.indptr_h.size();
    uint32_t ncols      = params.ncols;
    uint32_t nnz        = params.data_h.size();
    uint32_t map_length = params.map_length;
    uint32_t len        = nrows * ncols;

    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    // input matrix setup
    data_dense.resize(len);
    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    raft::sparse::convert::csr_to_dense(handle,
      nrows,
      ncols,
      nnz,
      indptr.data(),
      indices.data(),
      data.data(),
      nrows,
      data_dense.data(),
      stream,
      true);

    // map setup
    d_map.resize(map_length, stream);
    h_map.resize(map_length);
    raft::random::uniformInt(handle, r_int, d_map.data(), map_length, (MapT)0, nrows);
    raft::update_host(h_map.data(), d_map.data(), map_length, stream);

    // expected and actual output matrix setup
    d_out_exp.resize(map_length * ncols, stream);
    d_out_act.resize(map_length * ncols, stream);

    // launch gather on the host and copy the results to device
    raft::matrix::gather(data_dense.data(), ncols, nrows, d_map.data(), map_length, d_out_exp.data());

    // launch device version of the kernel
    gatherLaunch(d_in.data(), ncols, nrows, d_map.data(), map_length, d_out_act.data(), stream);

    raft::interruptible::synchronize(stream);
  }
  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  GatherSparseInputs params;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;
  rmm::device_uvector<value_t> data_dense;

  //std::vector<value_t> data_dense;
  std::vector<MapT> h_map;
  rmm::device_uvector<MatrixT> d_in, d_out_exp, d_out_act;
  rmm::device_uvector<MapT> d_map;
};

const std::vector<GatherSparseInputs> inputs = {
  {9,                                                 // ncols
    {0, 2, 4, 6, 8},                                   // indptr
    {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
    {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
    2,
    1234ULL
  },
  {9,                                                 // ncols
    {0, 2, 4, 6, 8},                                   // indptr
    {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
    {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
    3,
    1234ULL
  },
  {9,                                                 // ncols
    {0, 2, 4, 6, 8},                                   // indptr
    {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
    {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
    4,
    1234ULL
  },
  {4, // ncols
    {0, 2, 4, 6, 8}, //indptr
    {0, 1, 2, 3, 0, 1, 2, 3},  // indices
    {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f}, //data
    3,
    1234ULL
  },
  {4, // ncols
    {0, 2, 4, 6, 8}, //indptr
    {0, 1, 2, 3, 0, 1, 2, 3},  // indices
    {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f}, //data
    4,
    1234ULL
  },
  {4, // ncols
    {0, 2, 4, 6, 8}, //indptr
    {0, 1, 2, 3, 0, 1, 2, 3},  // indices
    {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f}, //data
    5,
    1234ULL
  }};

typedef GatherTest<float, uint32_t> GatherTestF;
TEST_P(GatherTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    d_out_exp.data(), d_out_act.data(), params.map_length * params.ncols, raft::Compare<float>()));
}

typedef GatherTest<double, uint32_t> GatherTestD;
TEST_P(GatherTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    d_out_exp.data(), d_out_act.data(), params.map_length * params.ncols, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestF, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(GatherTests, GatherTestD, ::testing::ValuesIn(inputs));

}  // end namespace sparse
}  // end namespace raft