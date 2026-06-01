/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>

#include <cstdint>

namespace RAFT_EXPORT raft {
namespace sparse::solver {

/**
 * @brief Configuration parameters for the LOBPCG eigensolver.
 *
 * @tparam ValueTypeT Data type for values (float or double).
 */
template <typename ValueTypeT>
struct lobpcg_solver_config {
  /** Convergence tolerance for residual norm.  If <= 0, defaults to sqrt(1e-15) * n. */
  ValueTypeT tolerance = ValueTypeT(0);

  /** Maximum number of outer iterations. */
  std::int32_t max_iterations = 20;

  /** Compute the largest (true) or smallest (false) eigenvalues. */
  bool largest = true;

  /** Verbosity level for debug printing (0 = silent). */
  int verbosity_level = 0;
};

}  // namespace sparse::solver
}  // namespace RAFT_EXPORT raft
