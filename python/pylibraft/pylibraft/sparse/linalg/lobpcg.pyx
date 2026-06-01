#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cupy as cp
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, int64_t, uint32_t, uintptr_t

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from libcpp cimport bool


cdef extern from "cuda_runtime.h" nogil:
    int cudaGetLastError()
    int cudaDeviceSynchronize()

from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources


cdef extern from "raft/sparse/solver/lobpcg_types.hpp" \
        namespace "raft::sparse::solver" nogil:

    cdef cppclass lobpcg_solver_config[ValueTypeT]:
        ValueTypeT tolerance
        int32_t max_iterations
        bool largest
        int verbosity_level

cdef lobpcg_solver_config[float] config_float
cdef lobpcg_solver_config[double] config_double


cdef extern from "raft_runtime/solver/lobpcg.hpp" \
        namespace "raft::runtime::solver" nogil:

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[float] config,
        device_vector_view[int, uint32_t] rows,
        device_vector_view[int, uint32_t] cols,
        device_vector_view[float, uint32_t] vals,
        device_matrix_view[float, uint32_t, col_major] eigenvectors,
        device_vector_view[float, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[float] config,
        device_vector_view[int64_t, uint32_t] rows,
        device_vector_view[int64_t, uint32_t] cols,
        device_vector_view[float, uint32_t] vals,
        device_matrix_view[float, uint32_t, col_major] eigenvectors,
        device_vector_view[float, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[double] config,
        device_vector_view[int, uint32_t] rows,
        device_vector_view[int, uint32_t] cols,
        device_vector_view[double, uint32_t] vals,
        device_matrix_view[double, uint32_t, col_major] eigenvectors,
        device_vector_view[double, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[double] config,
        device_vector_view[int64_t, uint32_t] rows,
        device_vector_view[int64_t, uint32_t] cols,
        device_vector_view[double, uint32_t] vals,
        device_matrix_view[double, uint32_t, col_major] eigenvectors,
        device_vector_view[double, uint32_t] eigenvalues) except +

    # Overloads with explicit preconditioner M (also CSR).
    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[float] config,
        device_vector_view[int, uint32_t] rows,
        device_vector_view[int, uint32_t] cols,
        device_vector_view[float, uint32_t] vals,
        device_vector_view[int, uint32_t] M_rows,
        device_vector_view[int, uint32_t] M_cols,
        device_vector_view[float, uint32_t] M_vals,
        device_matrix_view[float, uint32_t, col_major] eigenvectors,
        device_vector_view[float, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[float] config,
        device_vector_view[int64_t, uint32_t] rows,
        device_vector_view[int64_t, uint32_t] cols,
        device_vector_view[float, uint32_t] vals,
        device_vector_view[int64_t, uint32_t] M_rows,
        device_vector_view[int64_t, uint32_t] M_cols,
        device_vector_view[float, uint32_t] M_vals,
        device_matrix_view[float, uint32_t, col_major] eigenvectors,
        device_vector_view[float, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[double] config,
        device_vector_view[int, uint32_t] rows,
        device_vector_view[int, uint32_t] cols,
        device_vector_view[double, uint32_t] vals,
        device_vector_view[int, uint32_t] M_rows,
        device_vector_view[int, uint32_t] M_cols,
        device_vector_view[double, uint32_t] M_vals,
        device_matrix_view[double, uint32_t, col_major] eigenvectors,
        device_vector_view[double, uint32_t] eigenvalues) except +

    cdef void lobpcg_solver(
        const device_resources &handle,
        lobpcg_solver_config[double] config,
        device_vector_view[int64_t, uint32_t] rows,
        device_vector_view[int64_t, uint32_t] cols,
        device_vector_view[double, uint32_t] vals,
        device_vector_view[int64_t, uint32_t] M_rows,
        device_vector_view[int64_t, uint32_t] M_cols,
        device_vector_view[double, uint32_t] M_vals,
        device_matrix_view[double, uint32_t, col_major] eigenvectors,
        device_vector_view[double, uint32_t] eigenvalues) except +


@auto_sync_handle
def lobpcg(A, k=1, largest=True, tol=0, maxiter=20, M=None, handle=None):
    """
    Find ``k`` largest (or smallest) eigenvalues/eigenvectors of a real
    symmetric sparse matrix ``A`` using the LOBPCG algorithm.

    Solves ``A x = lambda x`` for ``k`` eigenpairs.

    Args:
        A: A symmetric square sparse CSR matrix of shape ``(n, n)``
            (e.g. ``cupyx.scipy.sparse.csr_matrix``).
        k (int): Number of eigenpairs to compute. Must satisfy ``1 <= k < n``.
        largest (bool): If ``True``, compute the largest eigenvalues (default).
            If ``False``, compute the smallest.
        tol (float): Convergence tolerance for the residual norm. If ``0`` a
            default of ``sqrt(eps) * n`` is used.
        maxiter (int): Maximum number of outer iterations.
        M: Optional preconditioner. Must be a CSR matrix with the same
            shape, index dtype and value dtype as ``A`` (e.g.
            ``cupyx.scipy.sparse.csr_matrix``). When provided the
            algorithm applies ``M`` to the residuals at each iteration.
        handle: Optional ``pylibraft.common.Handle``.

    Returns:
        tuple: ``(eigenvalues, eigenvectors)`` as ``cupy.ndarray``.
    """
    if A is None:
        raise Exception("'A' cannot be None!")

    # Synchronize and clear any stale CUDA error state left behind by previous
    # CUDA work in the process.  This avoids RAFT's cudaPeekAtLastError checks
    # failing on errors that did not originate in this call.
    cudaDeviceSynchronize()
    cudaGetLastError()

    rows = cai_wrapper(A.indptr)
    cols = cai_wrapper(A.indices)
    vals = cai_wrapper(A.data)

    IndexType = rows.dtype
    ValueType = vals.dtype

    N = A.shape[0]
    nnz = A.nnz

    rows_ptr = <uintptr_t>rows.data
    cols_ptr = <uintptr_t>cols.data
    vals_ptr = <uintptr_t>vals.data

    cdef uintptr_t M_rows_ptr = 0
    cdef uintptr_t M_cols_ptr = 0
    cdef uintptr_t M_vals_ptr = 0
    cdef uint32_t M_nnz = 0
    if M is not None:
        if M.shape != A.shape:
            raise ValueError("M must have the same shape as A")
        M_rows_w = cai_wrapper(M.indptr)
        M_cols_w = cai_wrapper(M.indices)
        M_vals_w = cai_wrapper(M.data)
        if M_rows_w.dtype != IndexType:
            raise ValueError(
                "M.indptr dtype must match A.indptr dtype (%s vs %s)"
                % (M_rows_w.dtype, IndexType))
        if M_vals_w.dtype != ValueType:
            raise ValueError(
                "M.data dtype must match A.data dtype (%s vs %s)"
                % (M_vals_w.dtype, ValueType))
        M_rows_ptr = <uintptr_t>M_rows_w.data
        M_cols_ptr = <uintptr_t>M_cols_w.data
        M_vals_ptr = <uintptr_t>M_vals_w.data
        M_nnz = <uint32_t>M.nnz

    # LOBPCG uses the eigenvectors buffer as the INITIAL guess.  Build it from
    # numpy (host) random values so the iteration starts with a full-rank
    # block.  Using a host buffer also avoids contaminating any cupy/cuda
    # state from a host->device copy inside the binding.
    init_x = np.asfortranarray(
        np.random.default_rng(0).standard_normal((N, k)).astype(ValueType))
    eigenvectors = device_ndarray(init_x)
    eigenvalues = device_ndarray.empty((k,), dtype=ValueType, order='F')

    eigenvectors_cai = cai_wrapper(eigenvectors)
    eigenvalues_cai = cai_wrapper(eigenvalues)

    eigenvectors_ptr = <uintptr_t>eigenvectors_cai.data
    eigenvalues_ptr = <uintptr_t>eigenvalues_cai.data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    if IndexType == np.int32 and ValueType == np.float32:
        config_float.tolerance = tol
        config_float.max_iterations = maxiter
        config_float.largest = largest
        config_float.verbosity_level = 0
        if M is None:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[float]> config_float,
                make_device_vector_view(<int *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<float *>vals_ptr, <uint32_t> nnz),
                make_device_matrix_view[float, uint32_t, col_major](
                    <float *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <float *>eigenvalues_ptr, <uint32_t> k),
            )
        else:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[float]> config_float,
                make_device_vector_view(<int *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<float *>vals_ptr, <uint32_t> nnz),
                make_device_vector_view(<int *>M_rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>M_cols_ptr, M_nnz),
                make_device_vector_view(<float *>M_vals_ptr, M_nnz),
                make_device_matrix_view[float, uint32_t, col_major](
                    <float *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <float *>eigenvalues_ptr, <uint32_t> k),
            )
    elif IndexType == np.int64 and ValueType == np.float32:
        config_float.tolerance = tol
        config_float.max_iterations = maxiter
        config_float.largest = largest
        config_float.verbosity_level = 0
        if M is None:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[float]> config_float,
                make_device_vector_view(<int64_t *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<float *>vals_ptr, <uint32_t> nnz),
                make_device_matrix_view[float, uint32_t, col_major](
                    <float *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <float *>eigenvalues_ptr, <uint32_t> k),
            )
        else:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[float]> config_float,
                make_device_vector_view(<int64_t *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<float *>vals_ptr, <uint32_t> nnz),
                make_device_vector_view(
                    <int64_t *>M_rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>M_cols_ptr, M_nnz),
                make_device_vector_view(<float *>M_vals_ptr, M_nnz),
                make_device_matrix_view[float, uint32_t, col_major](
                    <float *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <float *>eigenvalues_ptr, <uint32_t> k),
            )
    elif IndexType == np.int32 and ValueType == np.float64:
        config_double.tolerance = tol
        config_double.max_iterations = maxiter
        config_double.largest = largest
        config_double.verbosity_level = 0
        if M is None:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[double]> config_double,
                make_device_vector_view(<int *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<double *>vals_ptr, <uint32_t> nnz),
                make_device_matrix_view[double, uint32_t, col_major](
                    <double *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <double *>eigenvalues_ptr, <uint32_t> k),
            )
        else:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[double]> config_double,
                make_device_vector_view(<int *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<double *>vals_ptr, <uint32_t> nnz),
                make_device_vector_view(<int *>M_rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int *>M_cols_ptr, M_nnz),
                make_device_vector_view(<double *>M_vals_ptr, M_nnz),
                make_device_matrix_view[double, uint32_t, col_major](
                    <double *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <double *>eigenvalues_ptr, <uint32_t> k),
            )
    elif IndexType == np.int64 and ValueType == np.float64:
        config_double.tolerance = tol
        config_double.max_iterations = maxiter
        config_double.largest = largest
        config_double.verbosity_level = 0
        if M is None:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[double]> config_double,
                make_device_vector_view(<int64_t *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<double *>vals_ptr, <uint32_t> nnz),
                make_device_matrix_view[double, uint32_t, col_major](
                    <double *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <double *>eigenvalues_ptr, <uint32_t> k),
            )
        else:
            lobpcg_solver(
                deref(h),
                <lobpcg_solver_config[double]> config_double,
                make_device_vector_view(<int64_t *>rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>cols_ptr, <uint32_t> nnz),
                make_device_vector_view(<double *>vals_ptr, <uint32_t> nnz),
                make_device_vector_view(
                    <int64_t *>M_rows_ptr, <uint32_t> (N + 1)),
                make_device_vector_view(<int64_t *>M_cols_ptr, M_nnz),
                make_device_vector_view(<double *>M_vals_ptr, M_nnz),
                make_device_matrix_view[double, uint32_t, col_major](
                    <double *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k),
                make_device_vector_view(
                    <double *>eigenvalues_ptr, <uint32_t> k),
            )
    else:
        raise ValueError("dtype IndexType=%s and ValueType=%s not supported" %
                         (IndexType, ValueType))

    return (cp.asarray(eigenvalues), cp.asarray(eigenvectors))
