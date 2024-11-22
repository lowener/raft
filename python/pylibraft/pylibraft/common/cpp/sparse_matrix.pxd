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

# from pylibraft.common.handle cimport device_resources
cdef extern from "raft/core/device_csr_matrix.hpp" namespace "raft" nogil:

    cdef cppclass device_compressed_structure_view[IndptrType, IndicesType, NZType]:
        pass

    cdef cppclass device_csr_matrix_view[ElementType, IndptrType, IndicesType, NZType]:
        pass

    cdef device_csr_matrix_view[ElementType, IndptrType, IndicesType, NZType] \
        make_device_csr_matrix_view[ElementType, IndptrType, IndicesType, NZType](
            ElementType* ptr,
            device_compressed_structure_view[IndptrType, IndicesType, NZType] structure) except +

    cdef device_compressed_structure_view[IndptrType, IndicesType, NZType] \
        make_device_compressed_structure_view[IndptrType, IndicesType, NZType](
            IndptrType* indptr, IndicesType* indices,
            IndptrType n_rows, IndicesType n_cols, NZType nnz) except +