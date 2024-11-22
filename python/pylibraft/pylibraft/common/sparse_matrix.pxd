#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t

from pylibraft.common.cpp.sparse_matrix cimport (
    device_compressed_structure_view,
    device_csr_matrix_view,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.common.optional cimport make_optional, optional

# Cython doesn't like `const float` inside template parameters
# hack around this with using typedefs
ctypedef const float const_float
ctypedef const int8_t const_int8_t
ctypedef const uint8_t const_uint8_t


cdef device_csr_matrix_view[float, int32_t, int32_t, int32_t] get_csr_dmv_float_int32(
    spmatrix, structure) except *

cdef device_compressed_structure_view[int32_t, int32_t, int32_t] get_dcsv_int32(
    spmatrix) except *
