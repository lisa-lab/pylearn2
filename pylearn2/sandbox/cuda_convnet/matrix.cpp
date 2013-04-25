/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <matrix.h>
#include <matrix_funcs.h>

#if defined(_WIN64) || defined(_WIN32)
double sqrt(int _X) {return sqrt((double) _X);}
double log(int _X) {return log((double) _X);}
#endif

using namespace std;

void Matrix::_init(MTYPE* data, long int numRows, long int numCols, bool transpose, bool ownsData) {
    _updateDims(numRows, numCols);
    _ownsData = ownsData;
    _trans = transpose ? CblasTrans : CblasNoTrans;
    _data = data;
}

Matrix::Matrix() {
    _init(NULL, 0, 0, false, true);
}

Matrix::Matrix(long int numRows, long int numCols) {
    _init(NULL, numRows, numCols, false, true);
    this->_data = numRows * numCols > 0 ? new MTYPE[this->_numElements] : NULL;
}

Matrix::Matrix(const Matrix &like) {
    _init(NULL, like.getNumRows(), like.getNumCols(), false, true);
    this->_data = new MTYPE[this->_numElements];
}

/* construct a matrix with another matrix's data. the resultant
 * matrix does NOT own its data */
Matrix::Matrix(MTYPE* data, long int numRows, long int numCols) {
    _init(data, numRows, numCols, false, false);
}

/* construct a matrix with another matrix's data (and optionally transpose it). the resultant
 * matrix does NOT own its data -- it is a VIEW */
Matrix::Matrix(MTYPE* data, long int numRows, long int numCols, bool transpose) {
    _init(data, numRows, numCols, transpose, false);
}

#ifdef NUMPY_INTERFACE
Matrix::Matrix(const PyArrayObject *src) {
    this->_data = NULL;
    this->_trans = CblasNoTrans;
    if (src != NULL) {
        this->_updateDims(PyArray_DIM(src,0), PyArray_DIM(src,1));
        if (src->flags & NPY_CONTIGUOUS || src->flags & NPY_FORTRAN) {
            this->_data = (MTYPE*) src->data;
            this->_ownsData = false;
            this->_trans = src->flags & NPY_CONTIGUOUS ? CblasNoTrans : CblasTrans;
        } else {
            this->_data = new MTYPE[PyArray_DIM(src,0) * PyArray_DIM(src,1)];
            for (long int i = 0; i < PyArray_DIM(src,0); i++) {
                for (long int j = 0; j < PyArray_DIM(src,1); j++) {
                    (*this)(i,j) = *reinterpret_cast<MTYPE*>(PyArray_GETPTR2(src,i,j));
                }
            }
            this->_ownsData = true;
        }
    }
}
#endif
Matrix::~Matrix() {
    if(this->_data != NULL && this->_ownsData) {
        delete[] this->_data;
    }
}


void Matrix::_updateDims(long int numRows, long int numCols) {
    this->_numRows = numRows;
    this->_numCols = numCols;
    this->_numElements = numRows * numCols;
}

void Matrix::_checkBounds(long int startRow, long int endRow, long int startCol, long int endCol) const {
    assert(startRow >= 0 && startRow <= _numRows);
    assert(endRow >= 0 && endRow <= _numRows);
    assert(startCol >= 0 && startCol <= _numCols);
    assert(endCol >= 0 && endCol <= _numCols);
}

/* will return a view if possible */
Matrix& Matrix::slice(long int startRow, long int endRow, long int startCol, long int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    if (!isTrans() && ((startCol == 0 && endCol == this->_numCols) || (startRow == endRow - 1))) {
        return *new Matrix(this->_data + startRow * this->_numCols + startCol, endRow - startRow, endCol - startCol);
    } else if (isTrans() && ((startRow == 0 && endRow == this->_numRows) || (startCol == endCol - 1))) {
        return *new Matrix(this->_data + startCol * this->_numRows + startRow, endRow - startRow, endCol - startCol, true);
    }
    Matrix& newSlice = *new Matrix(endRow - startRow, endCol - startCol);
    this->copy(newSlice, startRow, endRow, startCol, endCol, 0, 0);
    return newSlice;
}

/* this will NEVER return a view, unlike Matrix_slice */
void Matrix::slice(long int startRow, long int endRow, long int startCol, long int endCol, Matrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    target.resize(endRow - startRow, endCol - startCol);
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

Matrix& Matrix::sliceRows(long int startRow, long int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void Matrix::sliceRows(long int startRow, long int endRow, Matrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

Matrix& Matrix::sliceCols(long int startCol, long int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void Matrix::sliceCols(long int startCol, long int endCol, Matrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

void Matrix::subtractFromScalar(MTYPE scalar) {
    subtractFromScalar(scalar, *this);
}

void Matrix::subtractFromScalar(MTYPE scalar, Matrix& target) const {
    if(&target != this) {
        copy(target);
    }
    target.scale(-1);
    target.addScalar(scalar);
}

void Matrix::biggerThanScalar(MTYPE scalar) {
    biggerThanScalar(scalar, *this);
}

void Matrix::smallerThanScalar(MTYPE scalar) {
    smallerThanScalar(scalar, *this);
}

void Matrix::equalsScalar(MTYPE scalar) {
    equalsScalar(scalar, *this);
}

void Matrix::biggerThanScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_bigger, target);
}

void Matrix::smallerThanScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_smaller, target);
}

void Matrix::equalsScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_equal, target);
}

void Matrix::add(const Matrix &m) {
    add(m, 1, *this);
}

void Matrix::add(const Matrix &m, Matrix& target) {
    add(m, 1, target);
}

void Matrix::add(const Matrix &m, MTYPE scale) {
    add(m, scale, *this);
}

void Matrix::subtract(const Matrix &m) {
    add(m, -1, *this);
}

void Matrix::subtract(const Matrix &m, Matrix& target) {
    add(m, -1, target);
}

void Matrix::subtract(const Matrix &m, MTYPE scale) {
    add(m, -scale, *this);
}

void Matrix::subtract(const Matrix &m, MTYPE scale, Matrix& target) {
    add(m, -scale, target);
}

void Matrix::add(const Matrix &m, MTYPE scale, Matrix &target) {
    assert(this->isSameDims(m));
    if (isTrans() != m.isTrans() || isTrans() != target.isTrans()) {
        if (&target != this) {
            target.resize(*this);
        }
        if(scale == 1) {
            this->_applyLoop2(m, &_add, target);
        } else {
            this->_applyLoop2(m, &_addWithScale, scale, target);
        }
    } else {
        if (&target != this) {
            copy(target);
        }
        CBLAS_AXPY(getNumElements(), scale, m._data, 1, target._data, 1);
    }
}

void Matrix::addScalar(MTYPE scalar) {
    addScalar(scalar, *this);
}

void Matrix::addScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_add, target);
}

void Matrix::maxWithScalar(MTYPE scalar) {
    maxWithScalar(scalar, *this);
}

void Matrix::maxWithScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_max, target);
}

void Matrix::minWithScalar(MTYPE scalar) {
    minWithScalar(scalar, *this);
}

void Matrix::minWithScalar(MTYPE scalar, Matrix& target) const {
    target.resize(*this);
    _applyLoopScalar(scalar, &_min, target);
}

void Matrix::biggerThan(Matrix& a) {
    biggerThan(a, *this);
}

void Matrix::biggerThan(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_bigger, target);
}

void Matrix::smallerThan(Matrix& a) {
    smallerThan(a, *this);
}

void Matrix::smallerThan(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_smaller, target);
}

void Matrix::equals(Matrix& a) {
    equals(a, *this);
}

void Matrix::equals(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_equal, target);
}

void Matrix::notEquals(Matrix& a) {
    notEquals(a, *this);
}

void Matrix::notEquals(Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_notEqual, target);
}

void Matrix::minWith(Matrix &a) {
    minWith(a, *this);
}

void Matrix::minWith(Matrix &a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_min, target);
}

void Matrix::maxWith(Matrix &a) {
    maxWith(a, *this);
}

void Matrix::maxWith(Matrix &a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    _applyLoop2(a, &_max, target);
}

/* this := this + scale*tile(vec) */
void Matrix::addVector(const Matrix& vec, MTYPE scale, Matrix& target) {
    if(&target != this) {
        copy(target);
    }
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    assert((rowVector && vec.getNumCols() == target.getNumCols()) || (!rowVector && vec.getNumRows() == target.getNumRows()));
    const long int loopTil = rowVector ? target.getNumRows() : target.getNumCols();
    const long int dataInc = ((rowVector && target.isTrans()) || (!rowVector && !target.isTrans())) ? 1 : (rowVector ? target.getNumCols() : target.getNumRows());
    const long int myStride = ((target.isTrans() && rowVector) || (!target.isTrans() && !rowVector)) ? loopTil : 1;
    for (long int i = 0; i < loopTil; i++) {
        CBLAS_AXPY(vec.getNumElements(), scale, vec._data, 1, target._data + dataInc * i, myStride);
    }
}

/* this := this + scale*tile(vec) */
void Matrix::addVector(const Matrix& vec, MTYPE scale) {
    addVector(vec, scale, *this);
}

void Matrix::addVector(const Matrix& vec) {
    addVector(vec, 1, *this);
}

void Matrix::addVector(const Matrix& vec, Matrix& target) {
    addVector(vec, 1, target);
}

void Matrix::eltWiseMultByVector(const Matrix& vec) {
    eltWiseMultByVector(vec, *this);
}

/* omg test these */
void Matrix::eltWiseMultByVector(const Matrix& vec, Matrix& target) {
    if(&target != this) {
        copy(target);
    }
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    assert((rowVector && vec.getNumCols() == target.getNumCols()) || (!rowVector && vec.getNumRows() == target.getNumRows()));
    const long int dataInc = ((rowVector && !target.isTrans()) || (!rowVector && target.isTrans())) ? 1 : (rowVector ? target.getNumRows() : target.getNumCols());
    const long int myStride = ((!target.isTrans() && !rowVector) || (target.isTrans() && rowVector)) ? 1 : vec.getNumElements();
    const long int numScaling = rowVector ? target.getNumRows() : target.getNumCols();
    for (long int i = 0; i < vec.getNumElements(); i++) {
        CBLAS_SCAL(numScaling, vec._data[i], target._data + dataInc * i, myStride);
    }
}

/* return := scale * this * b */
void Matrix::rightMult(const Matrix& b, MTYPE scale) {
    rightMult(b, scale, *this);
}

/* return := this * b */
void Matrix::rightMult(const Matrix& b) {
    rightMult(b, 1);
}

/* target := this * b
 * also resizes target if necessary.*/
void Matrix::rightMult(const Matrix &b, Matrix &target) const {
    rightMult(b, 1, target);
}

/* target := scaleAB * this * b
 * also resizes target if necessary.*/
void Matrix::rightMult(const Matrix &b, MTYPE scaleAB, Matrix &target) const {
    if(&target != this) {
        target.resize(this->_numRows, b._numCols);
    }
    target.addProduct(*this, b, scaleAB, 0);
}

/* this := scaleAB * a*b + scaleC * this
 * ALL SIZES MUST BE CORRECT. */
void Matrix::addProduct(const Matrix& a, const Matrix& b, MTYPE scaleAB, MTYPE scaleThis) {
    assert(a.getNumCols() == b.getNumRows());
    assert(this->getNumRows() == a.getNumRows() && this->getNumCols() == b.getNumCols());
    assert(!isTrans());
    CBLAS_GEMM(CblasRowMajor, a._trans, b._trans, a._numRows, b._numCols, a._numCols, scaleAB, a._data,
            a._getNumColsBackEnd(), b._data, b._getNumColsBackEnd(), scaleThis, this->_data, this->_numCols);
}

void Matrix::addProduct(const Matrix& a, const Matrix& b) {
    addProduct(a, b, 1, 1);
}

Matrix& Matrix::transpose() const {
    return *new Matrix(this->_data, this->_numCols, this->_numRows, !isTrans());
}

Matrix& Matrix::transpose(bool hard) const {
    if (!hard || isTrans()) {
        return transpose();
    }
    Matrix &meTrans = *new Matrix(_numCols, _numRows);
    for (long int i = 0; i < _numRows; i++) {
        for (long int j = 0; j < _numCols; j++) {
            meTrans(j, i) = (*this)(i, j);
        }
    }
    return meTrans;
}

Matrix& Matrix::tile(long int timesY, long int timesX) const {
    Matrix& tiled = *new Matrix(this->_numRows * timesY, this->_numCols * timesX);
    _tileTo2(tiled);
    return tiled;
}

/* resizes target if necessary */
void Matrix::tile(long int timesY, long int timesX, Matrix& target) const {
    target.resize(this->_numRows * timesY, this->_numCols * timesX);
    _tileTo2(target);
}

/* a variant ... seems to be no faster than original. */
void Matrix::_tileTo2(Matrix& target) const {
    for(long int y = 0; y < target._numRows; y += this->_numRows) {
        for(long int x = 0; x < target._numCols; x += this->_numCols) {
            this->copy(target, 0, -1, 0, -1, y, x);
        }
    }
}

/* guarantees that result will be non-transposed */
void Matrix::resize(long int newNumRows, long int newNumCols) {
    if(this->_numRows != newNumRows || this->_numCols != newNumCols) {
        assert(!isView());
        if (this->getNumElements() != newNumRows * newNumCols) {
            delete[] this->_data; //deleting NULL is ok, sez c++
            this->_data = new MTYPE[newNumRows * newNumCols];
        }
        this->_updateDims(newNumRows, newNumCols);
        this->_trans = CblasNoTrans;
    }
}

void Matrix::resize(const Matrix& like) {
    resize(like.getNumRows(), like.getNumCols());
}

void Matrix::scale(MTYPE alpha) {
    scale(alpha, *this);
}

void Matrix::scale(MTYPE alpha, Matrix& target) {
    if (&target != this) {
        target.resize(*this);
        copy(target);
    }
    CBLAS_SCAL(getNumElements(), alpha, target._data, 1);
}

/* performs no resizing.
 * Warnings:
 * 1. ALL DIMENSIONS MUST BE CORRECT
 * 2. The source and destination memories better not overlap! */
void Matrix::copy(Matrix& dest, long int srcStartRow, long int srcEndRow, long int srcStartCol, long int srcEndCol, long int destStartRow, long int destStartCol) const {
    srcEndRow = srcEndRow < 0 ? this->_numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? this->_numCols : srcEndCol;
    assert(destStartRow >= 0 && destStartCol >= 0); //some range-checking
    assert(srcEndRow <= _numRows && srcEndCol <= _numCols);
    assert(destStartRow + srcEndRow - srcStartRow <= dest.getNumRows());
    assert(destStartCol + srcEndCol - srcStartCol <= dest.getNumCols());
    // I found no evidence that memcpy is actually faster than just
    // copying element-by-element.
    if (!isTrans() && !dest.isTrans()) {
        long int src_start_idx = this->_numCols * srcStartRow + srcStartCol;
        long int dest_start_idx = dest._numCols * destStartRow + destStartCol;
        long int copy_row_width = srcEndCol - srcStartCol;

        for (long int i = srcStartRow; i < srcEndRow; i++) {
            memcpy(dest._data + dest_start_idx + dest._numCols * (i - srcStartRow),
                    this->_data + src_start_idx + this->_numCols * (i - srcStartRow), sizeof(MTYPE) * copy_row_width);
        }
    } else {
        for (long int i = srcStartRow; i < srcEndRow; i++) {
            for (long int j = srcStartCol; j < srcEndCol; j++) {
                dest(i - srcStartRow + destStartRow, j - srcStartCol + destStartCol) = (*this)(i, j);
            }
        }
    }
}

/* preserves everything excluding transposedness.
 * new matrix owns its data */
Matrix& Matrix::copy() const {
    Matrix& copy = *new Matrix(*this);
    this->copy(copy);
    return copy;
}

/* resizes target if necessary */
void Matrix::copy(Matrix& target) const {
    target.resize(this->_numRows, this->_numCols); //target is now non-transposed
    if(this->isTrans() == target.isTrans()) {
        this->_copyAllTo(target);
    } else { //if I'm transposed, make sure that target is non-transposed copy
        this->copy(target, 0, -1, 0, -1, 0, 0);
    }
}

void Matrix::_copyAllTo(Matrix& target) const {
    assert(target.isTrans() == isTrans());
    memcpy((void*) target._data, (void*) this->_data, this->getNumDataBytes());
    target._trans = this->_trans;
}

MTYPE Matrix::min() const {
    return _aggregate(&_min, MTYPE_MAX);
}

Matrix& Matrix::min(long int axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->min(axis, target);
    return target;
}

void Matrix::min(long int axis, Matrix& target) const {
    _aggregate(axis, target, &_min, MTYPE_MAX);
}

MTYPE Matrix::max() const {
    return _aggregate(&_max, -MTYPE_MAX);
}

Matrix& Matrix::max(long int axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->max(axis, target);
    return target;
}

void Matrix::max(long int axis, Matrix& target) const {
    _aggregate(axis, target, &_max, -MTYPE_MAX);
}

MTYPE Matrix::sum() const {
    return _aggregate(&_add, 0);
}

MTYPE Matrix::norm() const {
    return sqrt(norm2());
}

MTYPE Matrix::norm2() const {
    return _aggregate(&_addSquare, 0);
}

Matrix& Matrix::sum(long int axis) const {
    Matrix& target = axis == 0 ? *new Matrix(1, this->_numCols) : *new Matrix(this->_numRows, 1);
    this->sum(axis, target);
    return target;
}

void Matrix::sum(long int axis, Matrix& target) const {
    _aggregate(axis, target, &_add, 0);
}

void Matrix::_aggregate(long int axis, Matrix& target, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    if (axis == 0) {
        target.resize(1, this->_numCols);
        for (long int j = 0; j < this->_numCols; j++) {
            target(0, j) = _aggregateCol(j, agg_func, initialValue);
        }
    } else {
        target.resize(this->_numRows, 1);
        for (long int i = 0; i < this->_numRows; i++) {
            target(i, 0) = _aggregateRow(i, agg_func, initialValue);
        }
    }
}

MTYPE Matrix::_aggregateRow(long int row, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    for (long int j = 0; j < this->_numCols; j++) {
        v = agg_func((*this)(row, j), v);
    }
    return v;
}

MTYPE Matrix::_aggregateCol(long int col, MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    for (long int i = 0; i < this->_numRows; i++) {
        v = agg_func((*this)(i, col), v);
    }
    return v;
}

MTYPE Matrix::_aggregate(MTYPE (*agg_func)(MTYPE, MTYPE), MTYPE initialValue) const {
    MTYPE v = initialValue;
    MTYPE* ptr = _data;
    for (long int i = 0; i < getNumElements(); i++, ptr++) {
        v = agg_func(*ptr, v);
    }
    return v;
}

void Matrix::printShape(const char* name) const {
    printf("%%s: %%ldx%%ld\n", name, getNumRows(), getNumCols());
}

void Matrix::print() const {
    print(0,getNumRows(),0, getNumCols());
}

void Matrix::print(long int rows, long int cols) const {
    print(0,rows,0, cols);
}

void Matrix::print(long int startRow, long int rows, long int startCol, long int cols) const {
    for (long int i = startRow; i < std::min(startRow+rows, this->_numRows); i++) {
        for (long int j = startCol; j < std::min(startCol+cols, this->_numCols); j++) {
            printf("%%.15f ", (*this)(i, j));
        }
        printf("\n");
    }
}

void Matrix::apply(Matrix::FUNCTION f) {
    apply(f, *this);
}

#ifdef USE_MKL

void Matrix::apply(Matrix::FUNCTION f, Matrix& target) {
    target.resize(*this);
    if (f == EXP) {
        MKL_EXP(this->getNumElements(), this->_data, target._data);
    } else if (f == TANH) {
        MKL_TANH(this->getNumElements(), this->_data, target._data);
    } else if (f == RECIPROCAL) {
        MKL_RECIP(this->getNumElements(), this->_data, target._data);
    } else if (f == SQUARE) {
        MKL_SQUARE(this->getNumElements(), this->_data, target._data);
    } else if(f == LOG) {
        MKL_LOG(this->getNumElements(), this->_data, target._data);
    } else if (f == ZERO) {
        _applyLoop(&_zero, target);
    } else if (f == ONE) {
        _applyLoop(&_one, target);
    } else if(f == ABS) {
        _applyLoop(&_abs, target);
    } else if(f == SIGN) {
        _applyLoop(&_sign, target);
    } else if (f == LOGISTIC1) {
        if(&target != this) {
            copy(target);
        }
        target.scale(0.5);
        target.apply(Matrix::TANH);
        target.addScalar(1);
        target.scale(0.5);
    } else if (f == LOGISTIC2) {
        if(&target != this) {
            copy(target);
        }
        target.scale(-1);
        target.apply(Matrix::EXP);
        target.addScalar(1);
        target.apply(Matrix::RECIPROCAL);
    }
}

void Matrix::eltWiseMult(const Matrix& a) {
    eltWiseMult(a, *this);
}

void Matrix::eltWiseMult(const Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    if (a.isTrans() == this->isTrans()) {
        MKL_VECMUL(getNumElements(), this->_data, a._data, target._data);
    } else {
        this->_applyLoop2(a, &_mult, target);
    }
}

void Matrix::eltWiseDivide(const Matrix& a) {
    eltWiseDivide(a, *this);
}

void Matrix::eltWiseDivide(const Matrix& a, Matrix &target) const {
    assert(isSameDims(a));
    target.resize(*this);
    if (a.isTrans() == this->isTrans() && a.isTrans() == target.isTrans()) {
        MKL_VECDIV(getNumElements(), this->_data, a._data, target._data);
    } else {
        this->_applyLoop2(a, &_divide, target);
    }
}

void Matrix::eltWiseDivideByVector(const Matrix& vec) {
    eltWiseDivideByVector(vec, *this);
}

/* This function may allocate a chunk of memory at most as big as the input vector */
void Matrix::eltWiseDivideByVector(const Matrix& vec, Matrix& target) {
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    if(&target != this) {
        target.resize(*this);
    }
    const bool rowVector = vec.getNumRows() == 1;
    assert(rowVector && vec.getNumCols() == getNumCols() || !rowVector && vec.getNumRows() == getNumRows());
    if (isTrans() == target.isTrans() && (!isTrans() && rowVector || isTrans() && !rowVector)) {
        const long int loopTil = rowVector ? getNumRows() : getNumCols();
        const long int dataInc = rowVector ? getNumCols() : getNumRows();
        for (long int i = 0; i < loopTil; i++) {
            MKL_VECDIV(vec.getNumElements(), this->_data + i * dataInc, vec._data, target._data + i * dataInc);
        }
    } else {
        _divideByVector(vec, target);
    }
}

void Matrix::randomizeUniform(VSLStreamStatePtr stream) {
    assert(stream != NULL);
    MKL_UNIFORM(MKL_UNIFORM_RND_METHOD, stream, getNumElements(), this->_data, 0, 1);
}

void Matrix::randomizeNormal(VSLStreamStatePtr stream, MTYPE mean, MTYPE stdev) {
    assert(stream != NULL);
    MKL_NORMAL(MKL_GAUSSIAN_RND_METHOD, stream, getNumElements(), this->_data, mean, stdev);
}

void Matrix::randomizeNormal(VSLStreamStatePtr stream) {
    randomizeNormal(stream, 0, 1);
}

#else

void Matrix::apply(Matrix::FUNCTION f, Matrix& target) {
    MTYPE (*func)(MTYPE);
    if(f == EXP) {
        func = &_exp;
    } else if(f == TANH) {
        func = &_tanh;
    } else if(f == RECIPROCAL) {
        func = &_recip;
    } else if (f == SQUARE) {
        func = &_square;
    } else if(f == LOG) {
        func = &_log;
    } else if(f == ZERO) {
        func = &_zero;
    } else if (f == ONE) {
        func = &_one;
    } else if(f == LOGISTIC1) {
        func = &_sigma1;
    } else if(f == LOGISTIC2) {
        func = &_sigma2;
    } else if (f == ABS) {
        func = &_abs;
    } else if (f == SIGN) {
        func = &_sign;
    } else {
        throw "Matrix::apply: Unknown function type";
    }
    this->_applyLoop(func, target);
}

void Matrix::eltWiseMult(const Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    this->_applyLoop2(a, &_mult, target);
}

void Matrix::eltWiseDivide(const Matrix& a, Matrix& target) const {
    assert(isSameDims(a));
    target.resize(*this);
    this->_applyLoop2(a, &_divide, target);
}

void Matrix::eltWiseMult(const Matrix& a) {
    eltWiseMult(a, *this);
}

void Matrix::eltWiseDivide(const Matrix& a) {
    eltWiseDivide(a, *this);
}

void Matrix::randomizeUniform() {
    this->_applyLoop(&_rand);
}

void Matrix::randomizeNormal() {
    throw "randomizeNormal only implemented on MKL!";
}

void Matrix::randomizeNormal(MTYPE mean, MTYPE stdev) {
    throw "randomizeNormal only implemented on MKL!";
}

void Matrix::eltWiseDivideByVector(const Matrix& vec) {
    eltWiseDivideByVector(vec, *this);
}

/* This function allocates a chunk of memory at most as big as the input vector */
void Matrix::eltWiseDivideByVector(const Matrix& vec, Matrix& target) {
    assert(std::min(vec.getNumCols(), vec.getNumRows()) == 1);
    const bool rowVector = vec.getNumRows() == 1;
    assert((rowVector && vec.getNumCols() == getNumCols()) || (!rowVector && vec.getNumRows() == getNumRows()));
    if(&target != this) {
        target.resize(*this);
    }
    _divideByVector(vec, target);
}

#endif /* USE_MKL */

void Matrix::_divideByVector(const Matrix& vec, Matrix& target) {
    Matrix& vecInverse = vec.copy();
    vecInverse.apply(RECIPROCAL);
    eltWiseMultByVector(vecInverse,target);
    delete &vecInverse;
}

void Matrix::reshape(long int numRows, long int numCols) {
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
}

Matrix& Matrix::reshaped(long int numRows, long int numCols) {
    assert(_numElements == numRows*numCols);
    return *new Matrix(_data, numRows, numCols, isTrans());
}

void Matrix::_applyLoop(MTYPE (*func)(MTYPE), Matrix& target) {
    MTYPE *ptr = this->_data, *tgtPtr = target._data;
    for (long int i = 0; i < getNumElements(); i++, ptr++, tgtPtr++) {
        *tgtPtr = (*func)(*ptr);
    }
}

void Matrix::_applyLoop(MTYPE (*func)(MTYPE)) {
    _applyLoop(func, *this);
}

void Matrix::_applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE), Matrix& target) const {
    for (long int i = 0; i < getNumRows(); i++) {
        for (long int j = 0; j < getNumCols(); j++) {
            target(i, j) = (*func)((*this)(i, j), a(i, j));
        }
    }
}

void Matrix::_applyLoop2(const Matrix& a, MTYPE (*func)(MTYPE,MTYPE, MTYPE), MTYPE scalar, Matrix& target) const {
    for (long int i = 0; i < getNumRows(); i++) {
        for (long int j = 0; j < getNumCols(); j++) {
            target(i, j) = (*func)((*this)(i, j), a(i, j), scalar);
        }
    }
}

void Matrix::_applyLoopScalar(const MTYPE scalar, MTYPE(*func)(MTYPE, MTYPE), Matrix& target) const {
    MTYPE *myPtr = _data;
    MTYPE *targetPtr = target._data;
    for (long int i = 0; i < getNumElements(); i++, myPtr++, targetPtr++) {
        *targetPtr = (*func)(*myPtr, scalar);
    }
}

bool Matrix::hasNan() const {
    for (long int r = 0; r < _numRows; r++) {
        for (long int c = 0; c < _numCols; c++) {
            if (isnan((*this)(r,c))) {
                return true;
            }
        }
    }
    return false;
}

bool Matrix::hasInf() const {
    for (long int r = 0; r < _numRows; r++) {
        for (long int c = 0; c < _numCols; c++) {
            if (isinf((*this)(r,c))) {
                return true;
            }
        }
    }
    return false;
}


