# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy
import numpy
import pytest
from cupyx.scipy import sparse

from pylibraft.sparse.linalg import lobpcg


def _make_spd_matrix(n, dtype):
    """Construct a sparse SPD tridiagonal matrix with well-separated eigenvalues.

    Diagonal entries are 1..n, with -0.1 off-diagonals to break the degeneracy
    of a pure diagonal matrix while keeping the spectrum bounded and ordered.
    """
    diag = cupy.arange(1, n + 1, dtype=dtype)
    off = cupy.full((n - 1,), -0.1, dtype=dtype)
    A = sparse.diags([off, diag, off], offsets=[-1, 0, 1])
    return A.asformat("csr").astype(dtype)


def _residual(A, w, x):
    Ax = A @ x
    wx = cupy.multiply(x, w.reshape(1, -1))
    return cupy.linalg.norm(Ax - wx) / cupy.linalg.norm(w)


class TestLobpcg:
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_largest_eigenvalues(self, dtype, k):
        n = 12
        A = _make_spd_matrix(n, dtype)
        w, x = lobpcg(A, k=k, largest=True, maxiter=60)
        assert w.shape == (k,)
        assert x.shape == (n, k)
        res_tol = 1e-3 if dtype == numpy.float32 else 1e-6
        assert _residual(A, w, x) < res_tol

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_residual(self, dtype):
        """LOBPCG should produce eigenpairs with small residual."""
        n = 20
        A = _make_spd_matrix(n, dtype)
        w, x = lobpcg(A, k=2, largest=True, maxiter=50)
        res_tol = 1e-3 if dtype == numpy.float32 else 1e-6
        assert _residual(A, w, x) < res_tol

    @pytest.mark.parametrize("itype", [numpy.int32, numpy.int64])
    def test_index_types(self, itype):
        n = 12
        A = _make_spd_matrix(n, numpy.float64)
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        w, x = lobpcg(A, k=2, largest=True, maxiter=50)
        assert w.shape == (2,)
        assert _residual(A, w, x) < 1e-6

    def test_none_input_raises(self):
        with pytest.raises(Exception):
            lobpcg(None, k=1)
