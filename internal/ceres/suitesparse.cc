// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"

#ifndef CERES_NO_SUITESPARSE

#include <memory>
#include <string>
#include <vector>

#include "ceres/compressed_col_sparse_matrix_utils.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/linear_solver.h"
#include "ceres/suitesparse.h"
#include "ceres/triplet_sparse_matrix.h"
#include "cholmod.h"

namespace ceres::internal {
namespace {
int OrderingTypeToCHOLMODEnum(OrderingType ordering_type) {
  if (ordering_type == OrderingType::AMD) {
    return CHOLMOD_AMD;
  }
  if (ordering_type == OrderingType::NESDIS) {
    return CHOLMOD_NESDIS;
  }

  if (ordering_type == OrderingType::NATURAL) {
    return CHOLMOD_NATURAL;
  }
  LOG(FATAL) << "Congratulations you have discovered a bug in Ceres Solver."
             << "Please report it to the developers. " << ordering_type;
  return -1;
}

void CopyArrayToInt64(size_t size, const void* src, void* dst) {
  const int32_t* src_ptr = (const int32_t*)src;
  int64_t* dst_ptr = (int64_t*)dst;

  for (size_t i = 0; i < size; ++i) {
    dst_ptr[i] = src_ptr[i];
  }
}

void CopyArrayToInt32(size_t size, const void* src, void* dst) {
  const int64_t* src_ptr = (const int64_t*)src;
  int32_t* dst_ptr = (int32_t*)dst;

  for (size_t i = 0; i < size; ++i) {
    dst_ptr[i] = src_ptr[i];
  }
}

}  // namespace

CholmodSparseView::CholmodSparseView(CompressedRowSparseMatrix* A,
                                     cholmod_common* cc,
                                     bool use_gpu)
    : use_gpu_(use_gpu), cc_(cc) {
  if (use_gpu) {
    m_.nrow = A->num_cols();
    m_.ncol = A->num_rows();
    m_.nzmax = A->num_nonzeros();
    m_.nz = nullptr;

    m_.p = cholmod_l_malloc(m_.ncol + 1, sizeof(int64_t), cc);
    m_.i = cholmod_l_malloc(m_.nzmax, sizeof(int64_t), cc);
    CopyArrayToInt64(m_.ncol + 1, A->rows(), m_.p);
    CopyArrayToInt64(m_.nzmax, A->cols(), m_.i);

    m_.x = reinterpret_cast<void*>(A->mutable_values());
    m_.z = nullptr;

    if (A->storage_type() ==
        CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR) {
      m_.stype = 1;
    } else if (A->storage_type() ==
               CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR) {
      m_.stype = -1;
    } else {
      m_.stype = 0;
    }

    m_.itype = CHOLMOD_LONG;
    m_.xtype = CHOLMOD_REAL;
    m_.dtype = CHOLMOD_DOUBLE;
    m_.sorted = 1;
    m_.packed = 1;
  } else {
    m_.nrow = A->num_cols();
    m_.ncol = A->num_rows();
    m_.nzmax = A->num_nonzeros();
    m_.nz = nullptr;
    m_.p = reinterpret_cast<void*>(A->mutable_rows());
    m_.i = reinterpret_cast<void*>(A->mutable_cols());
    m_.x = reinterpret_cast<void*>(A->mutable_values());
    m_.z = nullptr;

    if (A->storage_type() ==
        CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR) {
      m_.stype = 1;
    } else if (A->storage_type() ==
               CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR) {
      m_.stype = -1;
    } else {
      m_.stype = 0;
    }

    m_.itype = CHOLMOD_INT;
    m_.xtype = CHOLMOD_REAL;
    m_.dtype = CHOLMOD_DOUBLE;
    m_.sorted = 1;
    m_.packed = 1;
  }
}

CholmodSparseView::~CholmodSparseView() {
  if (use_gpu_) {
    cholmod_l_free(m_.ncol + 1, sizeof(int64_t), m_.p, cc_);
    cholmod_l_free(m_.nzmax, sizeof(int64_t), m_.i, cc_);
  }
}

cholmod_sparse& CholmodSparseView::SparseRef() { return m_; }

cholmod_sparse* CholmodSparseView::SparsePtr() { return &m_; }

SuiteSparse::SuiteSparse(bool use_gpu) : use_gpu_(use_gpu) {
  if (use_gpu_) {
    cholmod_l_start(&cc_);
    cc_.useGPU = 1;
    cc_.supernodal = CHOLMOD_SUPERNODAL;
  } else {
    cholmod_start(&cc_);
  }
}

SuiteSparse::~SuiteSparse() {
  if (use_gpu_) {
    cholmod_l_finish(&cc_);
  } else {
    cholmod_finish(&cc_);
  }
}

cholmod_sparse* SuiteSparse::CreateSparseMatrix(TripletSparseMatrix* A) {
  if (use_gpu_) {
    cholmod_triplet triplet;

    triplet.nrow = A->num_rows();
    triplet.ncol = A->num_cols();
    triplet.nzmax = A->max_num_nonzeros();
    triplet.nnz = A->num_nonzeros();

    triplet.i = cholmod_l_malloc(triplet.nnz, sizeof(int64_t), &cc_);
    triplet.j = cholmod_l_malloc(triplet.nnz, sizeof(int64_t), &cc_);
    CopyArrayToInt64(triplet.nnz, A->mutable_rows(), triplet.i);
    CopyArrayToInt64(triplet.nnz, A->mutable_cols(), triplet.j);

    triplet.x = reinterpret_cast<void*>(A->mutable_values());
    triplet.stype = 0;  // Matrix is not symmetric.
    triplet.itype = CHOLMOD_LONG;
    triplet.xtype = CHOLMOD_REAL;
    triplet.dtype = CHOLMOD_DOUBLE;

    cholmod_sparse* sparse =
        cholmod_l_triplet_to_sparse(&triplet, triplet.nnz, &cc_);

    cholmod_l_free(triplet.nnz, sizeof(int64_t), triplet.i, &cc_);
    cholmod_l_free(triplet.nnz, sizeof(int64_t), triplet.j, &cc_);

    return sparse;
  } else {
    cholmod_triplet triplet;

    triplet.nrow = A->num_rows();
    triplet.ncol = A->num_cols();
    triplet.nzmax = A->max_num_nonzeros();
    triplet.nnz = A->num_nonzeros();
    triplet.i = reinterpret_cast<void*>(A->mutable_rows());
    triplet.j = reinterpret_cast<void*>(A->mutable_cols());
    triplet.x = reinterpret_cast<void*>(A->mutable_values());
    triplet.stype = 0;  // Matrix is not symmetric.
    triplet.itype = CHOLMOD_INT;
    triplet.xtype = CHOLMOD_REAL;
    triplet.dtype = CHOLMOD_DOUBLE;

    return cholmod_triplet_to_sparse(&triplet, triplet.nnz, &cc_);
  }
}

cholmod_sparse* SuiteSparse::CreateSparseMatrixTranspose(
    TripletSparseMatrix* A) {
  if (use_gpu_) {
    cholmod_triplet triplet;

    triplet.ncol = A->num_rows();  // swap row and columns
    triplet.nrow = A->num_cols();
    triplet.nzmax = A->max_num_nonzeros();
    triplet.nnz = A->num_nonzeros();

    // swap rows and columns
    triplet.j = cholmod_l_malloc(triplet.nnz, sizeof(int64_t), &cc_);
    triplet.i = cholmod_l_malloc(triplet.nnz, sizeof(int64_t), &cc_);
    CopyArrayToInt64(triplet.nnz, A->mutable_rows(), triplet.j);
    CopyArrayToInt64(triplet.nnz, A->mutable_cols(), triplet.i);

    triplet.x = reinterpret_cast<void*>(A->mutable_values());
    triplet.stype = 0;  // Matrix is not symmetric.
    triplet.itype = CHOLMOD_LONG;
    triplet.xtype = CHOLMOD_REAL;
    triplet.dtype = CHOLMOD_DOUBLE;

    cholmod_sparse* sparse =
        cholmod_l_triplet_to_sparse(&triplet, triplet.nnz, &cc_);

    cholmod_l_free(triplet.nnz, sizeof(int64_t), triplet.j, &cc_);
    cholmod_l_free(triplet.nnz, sizeof(int64_t), triplet.i, &cc_);

    return sparse;
  } else {
    cholmod_triplet triplet;

    triplet.ncol = A->num_rows();  // swap row and columns
    triplet.nrow = A->num_cols();
    triplet.nzmax = A->max_num_nonzeros();
    triplet.nnz = A->num_nonzeros();

    // swap rows and columns
    triplet.j = reinterpret_cast<void*>(A->mutable_rows());
    triplet.i = reinterpret_cast<void*>(A->mutable_cols());
    triplet.x = reinterpret_cast<void*>(A->mutable_values());
    triplet.stype = 0;  // Matrix is not symmetric.
    triplet.itype = CHOLMOD_INT;
    triplet.xtype = CHOLMOD_REAL;
    triplet.dtype = CHOLMOD_DOUBLE;

    return cholmod_triplet_to_sparse(&triplet, triplet.nnz, &cc_);
  }
}

CholmodSparseView SuiteSparse::CreateSparseMatrixTransposeView(
    CompressedRowSparseMatrix* A) {
  return CholmodSparseView(A, &cc_, use_gpu_);
}

cholmod_dense SuiteSparse::CreateDenseVectorView(const double* x, int size) {
  cholmod_dense v;
  v.nrow = size;
  v.ncol = 1;
  v.nzmax = size;
  v.d = size;
  v.x = const_cast<void*>(reinterpret_cast<const void*>(x));
  v.xtype = CHOLMOD_REAL;
  v.dtype = CHOLMOD_DOUBLE;
  return v;
}

cholmod_dense* SuiteSparse::CreateDenseVector(const double* x,
                                              int in_size,
                                              int out_size) {
  CHECK_LE(in_size, out_size);

  if (use_gpu_) {
    cholmod_dense* v = cholmod_l_zeros(out_size, 1, CHOLMOD_REAL, &cc_);
    if (x != nullptr) {
      memcpy(v->x, x, in_size * sizeof(*x));
    }
    return v;
  } else {
    cholmod_dense* v = cholmod_zeros(out_size, 1, CHOLMOD_REAL, &cc_);
    if (x != nullptr) {
      memcpy(v->x, x, in_size * sizeof(*x));
    }
    return v;
  }
}

cholmod_factor* SuiteSparse::AnalyzeCholesky(cholmod_sparse* A,
                                             OrderingType ordering_type,
                                             std::string* message) {
  cc_.nmethods = 1;
  cc_.method[0].ordering = OrderingTypeToCHOLMODEnum(ordering_type);

  // postordering with a NATURAL ordering leads to a significant regression in
  // performance. See https://github.com/ceres-solver/ceres-solver/issues/905
  if (ordering_type == OrderingType::NATURAL) {
    cc_.postorder = 0;
  }

  cholmod_factor* factor = nullptr;

  if (use_gpu_) {
    factor = cholmod_l_analyze(A, &cc_);
  } else {
    factor = cholmod_analyze(A, &cc_);
  }

  if (cc_.status != CHOLMOD_OK) {
    *message =
        StringPrintf("cholmod_analyze failed. error code: %d", cc_.status);
    return nullptr;
  }

  CHECK(factor != nullptr);
  if (VLOG_IS_ON(2)) {
    if (use_gpu_) {
      cholmod_l_print_common(const_cast<char*>("Symbolic Analysis"), &cc_);
    } else {
      cholmod_print_common(const_cast<char*>("Symbolic Analysis"), &cc_);
    }
  }

  return factor;
}

cholmod_factor* SuiteSparse::AnalyzeCholeskyWithGivenOrdering(
    cholmod_sparse* A, const std::vector<int>& ordering, std::string* message) {
  CHECK_EQ(ordering.size(), A->nrow);

  cc_.nmethods = 1;
  cc_.method[0].ordering = CHOLMOD_GIVEN;
  cholmod_factor* factor =
      cholmod_analyze_p(A, const_cast<int*>(ordering.data()), nullptr, 0, &cc_);

  if (cc_.status != CHOLMOD_OK) {
    *message =
        StringPrintf("cholmod_analyze failed. error code: %d", cc_.status);
    return nullptr;
  }

  CHECK(factor != nullptr);
  if (VLOG_IS_ON(2)) {
    cholmod_print_common(const_cast<char*>("Symbolic Analysis"), &cc_);
  }

  return factor;
}

cholmod_factor* SuiteSparse::AnalyzeCholeskyWithGivenOrdering(
    cholmod_sparse* A,
    const std::vector<int64_t>& ordering,
    std::string* message) {
  CHECK_EQ(ordering.size(), A->nrow);

  cc_.nmethods = 1;
  cc_.method[0].ordering = CHOLMOD_GIVEN;
  cholmod_factor* factor = cholmod_l_analyze_p(
      A, const_cast<int64_t*>(ordering.data()), nullptr, 0, &cc_);

  if (cc_.status != CHOLMOD_OK) {
    *message =
        StringPrintf("cholmod_analyze failed. error code: %d", cc_.status);
    return nullptr;
  }

  CHECK(factor != nullptr);
  if (VLOG_IS_ON(2)) {
    cholmod_l_print_common(const_cast<char*>("Symbolic Analysis"), &cc_);
  }

  return factor;
}

bool SuiteSparse::BlockOrdering(const cholmod_sparse* A,
                                OrderingType ordering_type,
                                const std::vector<Block>& row_blocks,
                                const std::vector<Block>& col_blocks,
                                std::vector<int>* ordering) {
  if (ordering_type == OrderingType::NATURAL) {
    ordering->resize(A->nrow);
    for (int i = 0; i < A->nrow; ++i) {
      (*ordering)[i] = i;
    }
    return true;
  }

  const int num_row_blocks = row_blocks.size();
  const int num_col_blocks = col_blocks.size();

  // Arrays storing the compressed column structure of the matrix
  // encoding the block sparsity of A.
  std::vector<int> block_cols;
  std::vector<int> block_rows;

  CompressedColumnScalarMatrixToBlockMatrix(reinterpret_cast<const int*>(A->i),
                                            reinterpret_cast<const int*>(A->p),
                                            row_blocks,
                                            col_blocks,
                                            &block_rows,
                                            &block_cols);
  cholmod_sparse_struct block_matrix;
  block_matrix.nrow = num_row_blocks;
  block_matrix.ncol = num_col_blocks;
  block_matrix.nzmax = block_rows.size();
  block_matrix.p = reinterpret_cast<void*>(block_cols.data());
  block_matrix.i = reinterpret_cast<void*>(block_rows.data());
  block_matrix.x = nullptr;
  block_matrix.stype = A->stype;
  block_matrix.itype = CHOLMOD_INT;
  block_matrix.xtype = CHOLMOD_PATTERN;
  block_matrix.dtype = CHOLMOD_DOUBLE;
  block_matrix.sorted = 1;
  block_matrix.packed = 1;

  std::vector<int> block_ordering(num_row_blocks);
  if (!Ordering(&block_matrix, ordering_type, block_ordering.data())) {
    return false;
  }

  BlockOrderingToScalarOrdering(row_blocks, block_ordering, ordering);
  return true;
}

bool SuiteSparse::BlockOrdering(const cholmod_sparse* A,
                                OrderingType ordering_type,
                                const std::vector<Block>& row_blocks,
                                const std::vector<Block>& col_blocks,
                                std::vector<int64_t>* ordering) {
  if (ordering_type == OrderingType::NATURAL) {
    ordering->resize(A->nrow);
    for (int i = 0; i < A->nrow; ++i) {
      (*ordering)[i] = i;
    }
    return true;
  }

  const int num_row_blocks = row_blocks.size();
  const int num_col_blocks = col_blocks.size();

  // Arrays storing the compressed column structure of the matrix
  // encoding the block sparsity of A.
  std::vector<int64_t> block_cols;
  std::vector<int64_t> block_rows;

  CompressedColumnScalarMatrixToBlockMatrix(
      reinterpret_cast<const int64_t*>(A->i),
      reinterpret_cast<const int64_t*>(A->p),
      row_blocks,
      col_blocks,
      &block_rows,
      &block_cols);
  cholmod_sparse_struct block_matrix;
  block_matrix.nrow = num_row_blocks;
  block_matrix.ncol = num_col_blocks;
  block_matrix.nzmax = block_rows.size();
  block_matrix.p = reinterpret_cast<void*>(block_cols.data());
  block_matrix.i = reinterpret_cast<void*>(block_rows.data());
  block_matrix.x = nullptr;
  block_matrix.stype = A->stype;
  block_matrix.itype = CHOLMOD_LONG;
  block_matrix.xtype = CHOLMOD_PATTERN;
  block_matrix.dtype = CHOLMOD_DOUBLE;
  block_matrix.sorted = 1;
  block_matrix.packed = 1;

  std::vector<int64_t> block_ordering(num_row_blocks);
  if (!Ordering(&block_matrix, ordering_type, block_ordering.data())) {
    return false;
  }

  BlockOrderingToScalarOrdering(row_blocks, block_ordering, ordering);
  return true;
}

cholmod_factor* SuiteSparse::BlockAnalyzeCholesky(
    cholmod_sparse* A,
    OrderingType ordering_type,
    const std::vector<Block>& row_blocks,
    const std::vector<Block>& col_blocks,
    std::string* message) {
  if (use_gpu_) {
    std::vector<int64_t> ordering;
    if (!BlockOrdering(A, ordering_type, row_blocks, col_blocks, &ordering)) {
      return nullptr;
    }
    return AnalyzeCholeskyWithGivenOrdering(A, ordering, message);
  } else {
    std::vector<int> ordering;
    if (!BlockOrdering(A, ordering_type, row_blocks, col_blocks, &ordering)) {
      return nullptr;
    }
    return AnalyzeCholeskyWithGivenOrdering(A, ordering, message);
  }
}

LinearSolverTerminationType SuiteSparse::Cholesky(cholmod_sparse* A,
                                                  cholmod_factor* L,
                                                  std::string* message) {
  CHECK(A != nullptr);
  CHECK(L != nullptr);

  // Save the current print level and silence CHOLMOD, otherwise
  // CHOLMOD is prone to dumping stuff to stderr, which can be
  // distracting when the error (matrix is indefinite) is not a fatal
  // failure.
  const int old_print_level = cc_.print;
  cc_.print = 0;

  cc_.quick_return_if_not_posdef = 1;
  int cholmod_status;
  if (use_gpu_) {
    cholmod_status = cholmod_l_factorize(A, L, &cc_);
  } else {
    cholmod_status = cholmod_factorize(A, L, &cc_);
  }
  cc_.print = old_print_level;

  switch (cc_.status) {
    case CHOLMOD_NOT_INSTALLED:
      *message = "CHOLMOD failure: Method not installed.";
      return LinearSolverTerminationType::FATAL_ERROR;
    case CHOLMOD_OUT_OF_MEMORY:
      *message = "CHOLMOD failure: Out of memory.";
      return LinearSolverTerminationType::FATAL_ERROR;
    case CHOLMOD_TOO_LARGE:
      *message = "CHOLMOD failure: Integer overflow occurred.";
      return LinearSolverTerminationType::FATAL_ERROR;
    case CHOLMOD_INVALID:
      *message = "CHOLMOD failure: Invalid input.";
      return LinearSolverTerminationType::FATAL_ERROR;
    case CHOLMOD_NOT_POSDEF:
      *message = "CHOLMOD warning: Matrix not positive definite.";
      return LinearSolverTerminationType::FAILURE;
    case CHOLMOD_DSMALL:
      *message =
          "CHOLMOD warning: D for LDL' or diag(L) or "
          "LL' has tiny absolute value.";
      return LinearSolverTerminationType::FAILURE;
    case CHOLMOD_OK:
      if (cholmod_status != 0) {
        return LinearSolverTerminationType::SUCCESS;
      }

      *message =
          "CHOLMOD failure: cholmod_factorize returned false "
          "but cholmod_common::status is CHOLMOD_OK."
          "Please report this to ceres-solver@googlegroups.com.";
      return LinearSolverTerminationType::FATAL_ERROR;
    default:
      *message = StringPrintf(
          "Unknown cholmod return code: %d. "
          "Please report this to ceres-solver@googlegroups.com.",
          cc_.status);
      return LinearSolverTerminationType::FATAL_ERROR;
  }

  return LinearSolverTerminationType::FATAL_ERROR;
}

cholmod_dense* SuiteSparse::Solve(cholmod_factor* L,
                                  cholmod_dense* b,
                                  std::string* message) {
  if (cc_.status != CHOLMOD_OK) {
    *message = "cholmod_solve failed. CHOLMOD status is not CHOLMOD_OK";
    return nullptr;
  }

  if (use_gpu_) {
    return cholmod_l_solve(CHOLMOD_A, L, b, &cc_);
  } else {
    return cholmod_solve(CHOLMOD_A, L, b, &cc_);
  }
}

bool SuiteSparse::Ordering(cholmod_sparse* matrix,
                           OrderingType ordering_type,
                           int* ordering) {
  CHECK_NE(ordering_type, OrderingType::NATURAL);
  if (ordering_type == OrderingType::AMD) {
    return cholmod_amd(matrix, nullptr, 0, ordering, &cc_);
  }

#ifdef CERES_NO_CHOLMOD_PARTITION
  return false;
#else
  std::vector<int> CParent(matrix->nrow, 0);
  std::vector<int> CMember(matrix->nrow, 0);
  return cholmod_nested_dissection(
      matrix, nullptr, 0, ordering, CParent.data(), CMember.data(), &cc_);
#endif
}

bool SuiteSparse::Ordering(cholmod_sparse* matrix,
                           OrderingType ordering_type,
                           int64_t* ordering) {
  CHECK_NE(ordering_type, OrderingType::NATURAL);
  if (ordering_type == OrderingType::AMD) {
    return cholmod_l_amd(matrix, nullptr, 0, ordering, &cc_);
  }

#ifdef CERES_NO_CHOLMOD_PARTITION
  return false;
#else
  std::vector<int64_t> CParent(matrix->nrow, 0);
  std::vector<int64_t> CMember(matrix->nrow, 0);
  return cholmod_l_nested_dissection(
      matrix, nullptr, 0, ordering, CParent.data(), CMember.data(), &cc_);
#endif
}

bool SuiteSparse::ConstrainedApproximateMinimumDegreeOrdering(
    cholmod_sparse* matrix, int* constraints, int* ordering) {
  return cholmod_camd(matrix, nullptr, 0, constraints, ordering, &cc_);
}

bool SuiteSparse::ConstrainedApproximateMinimumDegreeOrdering(
    cholmod_sparse* matrix, int64_t* constraints, int64_t* ordering) {
  return cholmod_l_camd(matrix, nullptr, 0, constraints, ordering, &cc_);
}

bool SuiteSparse::IsNestedDissectionAvailable() {
#ifdef CERES_NO_CHOLMOD_PARTITION
  return false;
#else
  return true;
#endif
}

std::unique_ptr<SparseCholesky> SuiteSparseCholesky::Create(
    const OrderingType ordering_type) {
  return std::unique_ptr<SparseCholesky>(
      new SuiteSparseCholesky(ordering_type));
}

SuiteSparseCholesky::SuiteSparseCholesky(const OrderingType ordering_type)
    : ordering_type_(ordering_type), ss_(true), factor_(nullptr) {}

SuiteSparseCholesky::~SuiteSparseCholesky() {
  if (factor_ != nullptr) {
    ss_.Free(factor_);
  }
}

LinearSolverTerminationType SuiteSparseCholesky::Factorize(
    CompressedRowSparseMatrix* lhs, std::string* message) {
  if (lhs == nullptr) {
    *message = "Failure: Input lhs is nullptr.";
    return LinearSolverTerminationType::FATAL_ERROR;
  }

  CholmodSparseView view = ss_.CreateSparseMatrixTransposeView(lhs);

  cholmod_sparse cholmod_lhs = view.SparseRef();

  // If a factorization does not exist, compute the symbolic
  // factorization first.
  //
  // If the ordering type is NATURAL, then there is no fill reducing
  // ordering to be computed, regardless of block structure, so we can
  // just call the scalar version of symbolic factorization. For
  // SuiteSparse this is the common case since we have already
  // pre-ordered the columns of the Jacobian.
  //
  // Similarly regardless of ordering type, if there is no block
  // structure in the matrix we call the scalar version of symbolic
  // factorization.
  if (factor_ == nullptr) {
    if (ordering_type_ == OrderingType::NATURAL ||
        (lhs->col_blocks().empty() || lhs->row_blocks().empty())) {
      factor_ = ss_.AnalyzeCholesky(&cholmod_lhs, ordering_type_, message);
    } else {
      factor_ = ss_.BlockAnalyzeCholesky(&cholmod_lhs,
                                         ordering_type_,
                                         lhs->col_blocks(),
                                         lhs->row_blocks(),
                                         message);
    }
  }

  if (factor_ == nullptr) {
    return LinearSolverTerminationType::FATAL_ERROR;
  }

  // Compute and return the numeric factorization.
  return ss_.Cholesky(&cholmod_lhs, factor_, message);
}

CompressedRowSparseMatrix::StorageType SuiteSparseCholesky::StorageType()
    const {
  return ((ordering_type_ == OrderingType::NATURAL)
              ? CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR
              : CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR);
}

LinearSolverTerminationType SuiteSparseCholesky::Solve(const double* rhs,
                                                       double* solution,
                                                       std::string* message) {
  // Error checking
  if (factor_ == nullptr) {
    *message = "Solve called without a call to Factorize first.";
    return LinearSolverTerminationType::FATAL_ERROR;
  }

  const int num_cols = factor_->n;
  cholmod_dense cholmod_rhs = ss_.CreateDenseVectorView(rhs, num_cols);
  cholmod_dense* cholmod_dense_solution =
      ss_.Solve(factor_, &cholmod_rhs, message);

  if (cholmod_dense_solution == nullptr) {
    return LinearSolverTerminationType::FAILURE;
  }

  memcpy(solution, cholmod_dense_solution->x, num_cols * sizeof(*solution));
  ss_.Free(cholmod_dense_solution);
  return LinearSolverTerminationType::SUCCESS;
}

}  // namespace ceres::internal

#endif  // CERES_NO_SUITESPARSE
