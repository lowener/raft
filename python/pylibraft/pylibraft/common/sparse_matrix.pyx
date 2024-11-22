#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import io

import numpy as np

from cpython.buffer cimport PyBUF_FULL_RO, PyBuffer_Release, PyObject_GetBuffer
from cpython.object cimport PyObject
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint32_t, uintptr_t
from libcpp cimport bool

from pylibraft.common.cpp.sparse_matrix cimport (
    device_compressed_structure_view,
    device_csr_matrix_view,
    make_device_compressed_structure_view,
    make_device_csr_matrix_view,
)


cdef device_csr_matrix_view[float, int32_t, int32_t, int32_t] \
        get_csr_dmv_float_int32(spmatrix, structure) except *:
    if spmatrix.format != "csr":
        raise TypeError("Expected a CSR matrix, got %s" % spmatrix.format)
    if spmatrix.dtype != np.float32:
        raise TypeError("dtype %s not supported" % spmatrix.dtype)
    if spmatrix.indptr.dtype != np.int32:
        raise TypeError("indptr dtype %s, %s not supported" % spmatrix.indptr.dtype)
    if spmatrix.indices.dtype != np.int32:
        raise TypeError("indices dtype %s, %s not supported" % spmatrix.indices.dtype)
    return make_device_csr_matrix_view[float, int32_t, int32_t, int32_t](
        <float*><uintptr_t>spmatrix.data,
        <device_compressed_structure_view[int32_t, int32_t, int32_t]>structure)

cdef device_compressed_structure_view[int32_t, int32_t, int32_t] \
        get_dcsv_int32(spmatrix) except *:
    if spmatrix.format != "csr":
        raise TypeError("Expected a CSR matrix, got %s" % spmatrix.format)
    if spmatrix.indptr.dtype != np.int32_t:
        raise TypeError("indptr dtype %s, %s not supported" % spmatrix.indptr.dtype)
    if spmatrix.indices.dtype != np.int32_t:
        raise TypeError("indices dtype %s, %s not supported" % spmatrix.indices.dtype)
    return make_device_compressed_structure_view[int32_t, int32_t, int32_t](
        <int32_t*><uintptr_t>spmatrix.indptr,
        <int32_t*><uintptr_t>spmatrix.indices,
        <int32_t>spmatrix.shape[0],
        <int32_t>spmatrix.shape[1],
        <int32_t>spmatrix.nnz)
