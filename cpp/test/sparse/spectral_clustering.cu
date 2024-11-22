/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace raft {
namespace spectral {

TEST(Raft, SpectralClustering)
{
  std::vector<int> off_h = {0,  16,  25,  35,  41,  44,  48,  52,  56,  61,  63, 66,
                            67, 69,  74,  76,  78,  80,  82,  84,  87,  89,  91, 93,
                            98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};
  std::vector<int> ind_h = {
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,  13, 17, 19,
    21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12, 13, 0,  6,  10, 0,  6,
    10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32, 33, 2,  33, 0,  4,  5,  0,  0,  3,
    0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,  0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32,
    33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26,
    32, 33, 1,  8,  32, 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31,
    33, 8,  9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  std::vector<int> cluster_id(num_verts, -1);

  rmm::device_vector<int> offsets_v(off_h);
  rmm::device_vector<int> indices_v(ind_h);
  rmm::device_vector<float> weights_v(w_h);
  rmm::device_vector<int> result_v(cluster_id);

  int n_clusters{8};
  int n_eig_vects{8};

  float evs_tolerance{.00001};
  float kmean_tolerance{.00001};
  int evs_max_iter{100};
  int kmean_max_iter{100};
  float score;

  rmm::device_vector<float> eig_vals(n_eig_vects);
  rmm::device_vector<float> eig_vects(n_eig_vects * num_verts);

  raft::handle_t handle;

  int restartIter_lanczos = 15 + n_eig_vects;

  unsigned long long seed1{1234567};
  unsigned long long seed2{12345678};
  bool reorthog{false};

  raft::spectral::matrix::sparse_matrix_t<int, float> const csr_m(handle,
                                                                  offsets_v.data().get(),
                                                                  indices_v.data().get(),
                                                                  weights_v.data().get(),
                                                                  num_verts,
                                                                  num_verts,
                                                                  num_edges);

  raft::spectral::eigen_solver_config_t<int, float> eig_cfg{
    n_eig_vects, evs_max_iter, restartIter_lanczos, evs_tolerance, reorthog, seed1};
  raft::spectral::lanczos_solver_t<int, float> eigen_solver{eig_cfg};

  raft::spectral::cluster_solver_config_t<int, float> clust_cfg{
    n_clusters, kmean_max_iter, kmean_tolerance, seed2};
  raft::spectral::kmeans_solver_t<int, float> cluster_solver{clust_cfg};

  int* clusters  = result_v.data().get();
  float* eigVals = eig_vals.data().get();
  float* eigVecs = eig_vects.data().get();

  auto stream   = raft::resource::get_cuda_stream(handle);
  auto cublas_h = raft::resource::get_cublas_handle(handle);

  std::tuple<int, float, int>
    stats;  //{iters_eig_solver,residual_cluster,iters_cluster_solver} // # iters eigen solver,
            // cluster solver residual, # iters cluster solver

  int n = csr_m.nrows_;

  raft::spectral::matrix::laplacian_matrix_t<int, float> L{handle, csr_m};

  auto eigen_config = eigen_solver.get_config();
  auto nEigVecs     = eigen_config.n_eigVecs;

  // Compute smallest eigenvalues and eigenvectors
  std::get<0>(stats) = eigen_solver.solve_smallest_eigenvectors(handle, L, eigVals, eigVecs);

  // Whiten eigenvector matrix
  raft::spectral::transform_eigen_matrix(handle, n, nEigVecs, eigVecs);

  // Find partition clustering
  auto pair_cluster = cluster_solver.solve(handle, n, nEigVecs, eigVecs, clusters);

  std::get<1>(stats) = pair_cluster.first;
  std::get<2>(stats) = pair_cluster.second;

  float edge_cut;
  float cost{0};

  raft::spectral::analyzePartition(
    handle, csr_m, n_clusters, result_v.data().get(), edge_cut, cost);

  score = edge_cut;
  if (score > 55.0) {
    std::cout << "Laplacian:" << std::endl;
    raft::print_device_vector("  offsets", L.row_offsets_, L.nrows_, std::cout);
    raft::print_device_vector("  indices", L.col_indices_, L.nnz_, std::cout);
    raft::print_device_vector("  values", L.values_, L.nnz_, std::cout);
    raft::print_device_vector("eigVals", eigVals, nEigVecs, std::cout);

    std::cout << "(" << std::get<0>(stats) << ", " << std::get<1>(stats) << ", "
              << std::get<2>(stats) << ")" << std::endl;

    raft::print_device_vector("result_v", result_v.data().get(), result_v.size(), std::cout);

    std::cout << "score = " << score << ", should be < 55.0" << std::endl;
  }
  ASSERT_LT(score, 55.0);  // should be < 55.0
}
}  // namespace spectral
}  // namespace raft