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

#include <set>
#include <vector>
#include <assert.h>
#include <cublas.h>
//#include <cutil_inline.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include <nvmatrix.cuh>
#include <nvmatrix_operators.cuh>
#include <map>

using namespace std;

/*
 * Device random number generator pointers.
 */
//map<int,curandGenerator_t> NVMatrix::rndGen;
map<int,curandState*> NVMatrix::rndDevStates;
pthread_mutex_t* NVMatrix::_rndMutex = makeMutex();

pthread_mutex_t* NVMatrix::makeMutex() {
    pthread_mutex_t* m = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(m, NULL);
    return m;
}

NVMatrix::NVMatrix(const CudaNdarray * view,
		int numRows, int numCols)
{
    //Check that the array is contiguous
    const int * dims = CudaNdarray_HOST_DIMS(view);
    const int * strides = CudaNdarray_HOST_STRIDES(view);
    int total = 1;
    for (int i = 0; i < view->nd; i++)
    {
	if (i + 1 == view->nd)
	{
	    assert(strides[i] == 4);
        total *= dims[i];
    }


    //Check that there is the right total amount of elements
    assert(total == numRows * numCols);

    //Make the view
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = false;
    _isTrans = false;
    _devData = view->devdata;
    _stride = getLeadingDim();
}

void NVMatrix::_init(int numRows, int numCols, int stride, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = true;

    _isTrans = isTrans;
    _devData = NULL;
    if (_numElements > 0) {
        cublasAlloc(_numElements, sizeof(float), (void**) &_devData);
        checkCublasError("!!!! device memory allocation error\n");
    }
    _stride = stride < 0 ? getLeadingDim() : stride;
}

NVMatrix::NVMatrix() {
    _init(0, 0, -1, false);
}

NVMatrix::NVMatrix(bool isTrans) {
    _init(0, 0, -1, isTrans);
}

NVMatrix::NVMatrix(int numRows, int numCols, bool isTrans) {
    _init(numRows, numCols, -1, isTrans);
}

NVMatrix::NVMatrix(const Matrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols(), -1, like.isTrans());
    if (copy) {
        copyFromHost(like);
    }
}

NVMatrix::NVMatrix(const NVMatrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols(), -1, like.isTrans());
    if (copy) {
        like.copy(*this);
    }
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const NVMatrix& like) {
    _init(like.getNumRows(), like.getNumCols(), -1, like.isTrans());
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const Matrix& like) {
    _init(like.getNumRows(), like.getNumCols(), -1, false);
}

NVMatrix::NVMatrix(float* devData, int numRows, int numCols, int stride, bool isTrans) :
    _numRows(numRows),
    _numCols(numCols),
    _numElements(numRows*numCols),
    _ownsData(false),
    _devData(devData),
    _isTrans(isTrans) {
    _stride = stride < 0 ? getLeadingDim() : stride;
}

NVMatrix::~NVMatrix() {
    if(_ownsData && _numElements > 0) {
	// This line was modified by Ian Goodfellow to use device_free
	// so that theano may keep track of device memory usage
        cublasStatus status = device_free(_devData);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! memory free error\n");
            exit(EXIT_FAILURE);
        }
    }
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix) {
    if (resizeDeviceMatrix) {
        resize(hostMatrix);
    }
    copyFromHost(hostMatrix);
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix) {
//    assert(getStride() == getLeadingDim());
    assert(isSameDims(hostMatrix));
    setTrans(hostMatrix.isTrans());

    if (getNumElements() > 0) {
        cublasStatus status = cublasSetMatrix(hostMatrix.getLeadingDim(), hostMatrix.getFollowingDim(), sizeof(float),
                                              hostMatrix.getData(), hostMatrix.getLeadingDim(), _devData, _stride);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! device access error (write)\n");
            exit( EXIT_FAILURE);
        }
    }
}

void NVMatrix::copyToHost(Matrix& hostMatrix) const {
//    assert(getStride() == getLeadingDim());
    assert(isSameDims(hostMatrix));
    hostMatrix.setTrans(_isTrans);
    if (getNumElements() > 0) {
    //    printf("rows: %d, cols: %d, stride: %d\n", getNumRows(), getNumCols(), getStride());
        cublasStatus status = cublasGetMatrix(getLeadingDim(),getFollowingDim(), sizeof(float),
                                             _devData, getStride(), hostMatrix.getData(), hostMatrix.getLeadingDim());
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! device access error (read)\n");
            exit( EXIT_FAILURE);
        }
    }
}

void NVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    if (resizeTarget) {
        hostMatrix.resize(_numRows, _numCols);
    }
    copyToHost(hostMatrix);
}

void NVMatrix::copy(NVMatrix& dest) const {
    dest.resize(*this);
    copy(dest, 0, -1, 0, -1, 0, 0);
}

NVMatrix& NVMatrix::copy() const {
    NVMatrix* c = new NVMatrix();
    copy(*c);
    return *c;
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const {
    assert(isContiguous() && b.isContiguous() && target.isContiguous());
//    assert(&target != &b);
    assert(_numCols == b.getNumRows());
    if(&target != this) {
        target.resize(_numRows, b.getNumCols());
        target.setTrans(true);
    }
    assert(target.getNumRows() == _numRows);
    assert(target.getNumCols() == b.getNumCols());
    if(_numRows % 64 != 0 || _numCols % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(getTransChar(), b.getTransChar(), _numRows, b.getNumCols(), _numCols,
                scaleAB, _devData, getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                0, target.getDevData(), getNumRows());
    checkCublasError("cublasSgemm failed");
//    cudaThreadSynchronize();
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB) {
    rightMult(b, scaleAB, *this);
}

void NVMatrix::rightMult(const NVMatrix &b, NVMatrix& target) const {
    rightMult(b, 1, target);
}

/*
 * This will only work if this matrix is in column-major order! In other words,
 * if isTrans() returns true.
 */
void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB) {
    if (scaleThis == 0) {
        a.rightMult(b, scaleAB, *this);
        return;
    }
    assert(isContiguous());
    assert(a.getNumCols() == b.getNumRows());
    assert(this->getNumRows() == a.getNumRows());
    assert(this->getNumCols() == b.getNumCols());
    assert(_isTrans);
    if(a.getNumRows() % 64 != 0 || a.getNumCols() % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(a.getTransChar(), b.getTransChar(), a.getNumRows(), b.getNumCols(), a.getNumCols(),
                scaleAB, a.getDevData(), a.getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                scaleThis, _devData, getLeadingDim());
    checkCublasError("cublasSgemm failed");
//    cudaThreadSynchronize();
}

void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b) {
    addProduct(a, b, 1, 1);
}

template <class Randomizer>
void NVMatrix::_unaryRandomize(NVMatrix& target, Randomizer rnd) {
    assert(isRndInitialized());
    assert(isContiguous() && target.isContiguous());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    kUnaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    cutilCheckMsg("kUnaryRandomize: Kernel execution failed");
}

template <class Randomizer>
void NVMatrix::_binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd) {
    assert(isRndInitialized());
    assert(isContiguous() && data2.isContiguous() && target.isContiguous());
    assert(isSameDims(data2));
    assert(isTrans() == data2.isTrans());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert(isTrans() == target.isTrans());
    kBinaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(getDevData(), data2.getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    cutilCheckMsg("kBinaryRandomize: Kernel execution failed");
}

/* Function removed by Ian Goodfellow.
We do not need this function in theano / pylearn2 and it uses cudaMalloc directly.
If you need to enable it, modify it to use device_malloc instead.
Otherwise, theano will not be able to keep track of how much memory is used on
the device.
void NVMatrix::initRandom(unsigned long long seed) {
    assert(!isRndInitialized());
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    rndDevStates[d] = NULL;
    CUDA_CALL(cudaMalloc((void **)&rndDevStates[d], NUM_RND_STREAMS * sizeof(curandState)));
    pthread_mutex_unlock(_rndMutex);
    kSetupCurand<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(getCurandState(), 1 + seed*2); // so there's no chance it'll be correlated with the other one
    cutilCheckMsg("initRandom: Kernel execution failed");
}
*/

void NVMatrix::initRandom() {
    NVMatrix::initRandom(time(0));
}

curandState* NVMatrix::getCurandState() {
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    assert(rndDevStates.count(d) != 0);
    curandState* r = rndDevStates[d];
    pthread_mutex_unlock(_rndMutex);
    return r;
}

int NVMatrix::getDeviceID() {
    int d;
    cudaGetDevice(&d);
    return d;
}

bool NVMatrix::isRndInitialized() {
    pthread_mutex_lock(_rndMutex);
    bool b = rndDevStates.count(getDeviceID()) != 0;
    pthread_mutex_unlock(_rndMutex);
    return b;
}

/* Function removed by Ian Goodfellow due to not needing
   it and it using cudaFree instead of device_free 
void NVMatrix::destroyRandom() {
    assert(isRndInitialized());
    int d = getDeviceID();
    
    pthread_mutex_lock(_rndMutex);
    CUDA_CALL(cudaFree(rndDevStates[d]));
    rndDevStates.erase(d);
    pthread_mutex_unlock(_rndMutex);
} */

void NVMatrix::binarizeProbs() {
    binarizeProbs(*this);
}

void NVMatrix::binarizeProbs(NVMatrix& target) {
    _unaryRandomize(target, BinarizeUnaryRandomizer());
}

void NVMatrix::randomizeUniform() {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateUniform(rndGen, _devData, getNumElements()));
    _unaryRandomize(*this, UniformUnaryRandomizer());
}

void NVMatrix::randomizeGaussian() {
    randomizeGaussian(1);
}

void NVMatrix::randomizeGaussian(float stdev) {
    randomizeGaussian(0, stdev);
}

void NVMatrix::randomizeGaussian(float mean, float stdev) {
    assert(isContiguous());
    assert(isRndInitialized());
//    CURAND_CALL(curandGenerateNormal(rndGen, _devData, getNumElements(), mean, stdev));
    _unaryRandomize(*this, GaussianUnaryRandomizer(mean, stdev));
}

/*
 * Kind of a hack since we don't actually need the contents of this matrix for it,
 * so we don't really need a binary randomizer.
 */
void NVMatrix::randomizeGaussian(NVMatrix& stdevs) {
    _binaryRandomize(stdevs, *this, GaussianBinaryRandomizer());
}

void NVMatrix::addGaussianNoise() {
    addGaussianNoise(1);
}

void NVMatrix::addGaussianNoise(float stdev) {
    addGaussianNoise(stdev, *this);
}

void NVMatrix::addGaussianNoise(float stdev, NVMatrix& target) {
    _unaryRandomize(target, AddGaussianUnaryRandomizer(stdev));
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var) {
    addGaussianNoise(stdevs, var, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs) {
    addGaussianNoise(stdevs, false, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target) {
    if (var) {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<true>());
    } else {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<false>());
    }
}

void NVMatrix::biggerThan(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::BiggerThan(), b, target);
}

void NVMatrix::biggerThan(NVMatrix& b) {
    biggerThan(b, *this);
}

void NVMatrix::equals(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Equals(), b, target);
}

void NVMatrix::equals(NVMatrix& m) {
    equals(m, *this);
}

void NVMatrix::biggerThanVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::BiggerThan(), vec, target);
}

void NVMatrix::biggerThanVector(NVMatrix& vec) {
    biggerThanVector(vec, *this);
}

void NVMatrix::_checkBounds(int startRow, int endRow, int startCol, int endCol) const {
    assert(startRow >= 0 && startRow < _numRows);
    assert(endRow > startRow && endRow <= _numRows);
    assert(startCol >= 0 && startCol < _numCols);
    assert(endCol > startCol && endCol <= _numCols);
}

/*
 * The only place where stride is supported for now!
 * Will ALWAYS return a view of the original data, sometimes non-contiguous.
 */
NVMatrix& NVMatrix::slice(int startRow, int endRow, int startCol, int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    if (!isTrans()) {
        return *new NVMatrix(this->_devData + startRow * _stride + startCol, endRow - startRow, endCol - startCol, _stride, false);
    } 
    return *new NVMatrix(this->_devData + startCol * _stride + startRow, endRow - startRow, endCol - startCol, _stride, true);
}

/* this will NEVER return a view */
void NVMatrix::slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    int sliceRows = endRow - startRow, sliceCols = endCol - startCol;
    if (target.getNumRows() != sliceRows || target.getNumCols() != sliceCols) {
        target.resize(sliceRows, sliceCols);
    }
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

NVMatrix& NVMatrix::sliceRows(int startRow, int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void NVMatrix::sliceRows(int startRow, int endRow, NVMatrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

NVMatrix& NVMatrix::sliceCols(int startCol, int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void NVMatrix::sliceCols(int startCol, int endCol, NVMatrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.

Function removed by Ian Goodfellow due to not needing it and it using
cudaFree instead of device_free

bool NVMatrix::resize(int numRows, int numCols) {
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert(_ownsData);
        if (_numElements != numRows * numCols) {
            if (_numElements > 0) { // free old memory
                cublasStatus status = cublasFree(_devData);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "!!!! memory free error: %X\n", status);
                    exit(EXIT_FAILURE);
                }
            }
            if (numRows * numCols > 0) { // allocate new memory
                cublasStatus status = cublasAlloc(numCols * numRows, sizeof(float), (void**) &_devData);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "!!!! device memory allocation error\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                _devData = NULL;
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
        _stride = getLeadingDim();
    }
    return reallocated;
}
*/

bool NVMatrix::resize(const NVMatrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

bool NVMatrix::resize(const Matrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

void NVMatrix::reshape(int numRows, int numCols) {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
    _stride = getLeadingDim();
}

NVMatrix& NVMatrix::reshaped(int numRows, int numCols) {
    assert(isContiguous());
    assert(_numElements == numRows*numCols);
    return *new NVMatrix(_devData, numRows, numCols, -1, _isTrans);
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol) const {
    srcEndRow = srcEndRow < 0 ? _numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? _numCols : srcEndCol;
    NVMatrix* srcSlice = &slice(srcStartRow, srcEndRow, srcStartCol, srcEndCol);
    NVMatrix* destSlice = &dest.slice(destStartRow, destStartRow + srcEndRow - srcStartRow, destStartCol, destStartCol + srcEndCol - srcStartCol);
    srcSlice->apply(NVMatrixOps::Identity(), *destSlice);
    delete srcSlice;
    delete destSlice;
}


NVMatrix& NVMatrix::getTranspose() {
    return *new NVMatrix(_devData, _numCols, _numRows, _stride, !_isTrans);;
}

void NVMatrix::transpose(NVMatrix& target) {
    flipTrans(target);
    target.setTrans(!target.isTrans());
    target.reshape(target.getNumCols(), target.getNumRows());
}

void NVMatrix::transpose() {
    int tmp = _numCols;
    _numCols = _numRows;
    _numRows = tmp;
    _isTrans = !_isTrans;
}

bool NVMatrix::transpose(bool trans) {
    bool oldTrans = _isTrans;
    if (oldTrans != trans) {
        transpose();
    }
    return oldTrans;
}

/*
 * Flips the ordering of the matrix from row-major to column-major and vice versa.
 * This creates temporary storage -- not a cheap operation.
 *
 * This is not equivalent to a "hard transpose". The resultant matrix still has
 * the same dimensions, its layout in memory just changes.
 */
NVMatrix& NVMatrix::flipTrans() {
    NVMatrix* meTrans = new NVMatrix(*this);
    flipTrans(*meTrans);
    return *meTrans;
}

void NVMatrix::flipTrans(NVMatrix& target) {
    assert(&target != this);
    target.resize(_numRows, _numCols);
    target.setTrans(!isTrans());
    apply(NVMatrixOps::Identity(), target);
}

void NVMatrix::squaredDiff(NVMatrix& b) {
    squaredDiff(b, *this);
}

void NVMatrix::squaredDiff(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::SquaredDiff(), b, target);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target) {
    if (scaleA == 0) {
        b.scale(scaleB, target);
        return;
    }
    if (scaleA == 1 && scaleB == 1) { // slight optimization
        applyBinary(NVMatrixBinaryOps::Add(), b, target);
    } else {
        applyBinary(NVMatrixBinaryOps::WeightedAdd(scaleA, scaleB), b, target);
    }
}

void NVMatrix::add(NVMatrix& b, float scaleB, NVMatrix& target) {
    add(b, 1, scaleB, target);
}

void NVMatrix::add(NVMatrix& b, NVMatrix& target) {
    add(b, 1, target);
}

void NVMatrix::add(NVMatrix& b, float scaleB) {
    add(b, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB) {
    add(b, scaleA, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b) {
    add(b, 1, *this);
}

void NVMatrix::subtract(NVMatrix& b, NVMatrix& target) {
    add(b, -1, target);
}

void NVMatrix::subtract(NVMatrix& b) {
    add(b, -1);
}

void NVMatrix::eltwiseMult(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Multiply(), b, target);
}

void NVMatrix::eltwiseMult(NVMatrix& b) {
    eltwiseMult(b, *this);
}

void NVMatrix::eltwiseDivide(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Divide(), b, target);
}

void NVMatrix::eltwiseDivide(NVMatrix& b) {
    eltwiseDivide(b, *this);
}

void NVMatrix::tile(int timesY, int timesX, NVMatrix& target) {
    assert(isContiguous() && target.isContiguous());
    assert(timesX > 0 && timesY > 0);
    target.resize(_numRows*timesY, _numCols*timesX);
    target.setTrans(_isTrans);
    if(!isTrans()) {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK>>>(_devData, target._devData, _numCols, _numRows, target._numCols, target._numRows);
    } else {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK>>>(_devData, target._devData, _numRows, _numCols, target._numRows, target._numCols);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::WeightedAdd(1, scaleVec), vec, target);
}

void NVMatrix::addVector(NVMatrix& vec) {
    addVector(vec, 1, *this);
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec) {
    addVector(vec, scaleVec, *this);
}

void NVMatrix::addVector(NVMatrix& vec, NVMatrix& target) {
    addVector(vec, 1, target);
}

void NVMatrix::equalsVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Equals(), vec, target);
}

void NVMatrix::equalsVector(NVMatrix& vec) {
    equalsVector(vec, *this);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Multiply(), vec, target);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec) {
    eltwiseMultByVector(vec, *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec) {
    eltwiseDivideByVector(vec,  *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Divide(), vec, target);
}

/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 *
 * TODO: this is a mess, fix it. it works pretty fast but it's too ugly.
 * TODO: this function is _really_ bad for very long aggregations of few columns.
 */
template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp op) {
    assert(axis == 0 || axis == 1);
    assert(isContiguous()  && target.isContiguous());
    assert(&target != this);
    int width = _isTrans ? _numRows : _numCols;
    int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
        assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
        assert(numBlocks < NUM_BLOCKS_MAX);
        kDumbAggCols<Agg, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK>>>(_devData, target._devData, width, height, agg, op);
        cutilCheckMsg("kDumbAggCols: Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY,2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        kAggShortRows<Agg, BinaryOp, 1, 4><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else if(width <= 8) {
                        kAggShortRows<Agg, BinaryOp, 1, 8><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else if(width <= 12) {
                        kAggShortRows<Agg, BinaryOp, 1, 12><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else {
                        kAggShortRows<Agg, BinaryOp, 1, 16><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    }
                } else if(width <= 32) {
                    kAggShortRows<Agg, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else if(width <= 48){
                    kAggShortRows<Agg, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else if(width <= 64){
                    kAggShortRows<Agg, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else {
                    kAggShortRows2<Agg, BinaryOp><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                }
            } else {
                if (width >= 512) {
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, std::min(1024, height));
                    kAggRows_wholerow_nosync<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
//                    dim3 threads(AWR_NUM_THREADS);
//                    dim3 blocks(1, std::min(1024, height));
//                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
                    
                } else {
//                    dim3 threads(AWR_NUM_THREADS);
//                    dim3 blocks(1, std::min(1024, height));
//                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
                    NVMatrix *prevSum = this;
                    while (prevSum->getLeadingDim() > 1) {
                        int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                        int numThreadsY = 1;
                        int numBlocksX = DIVUP(width, 2*numThreadsX);
                        int numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                        NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);

                        dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                        assert(numBlocksX <= NUM_BLOCKS_MAX);
                        assert(numBlocksY <= NUM_BLOCKS_MAX);

                        if(width <= 64) {
                            kAggRows<Agg, BinaryOp, 32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 128) {
                            kAggRows<Agg, BinaryOp, 64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 256) {
                            kAggRows<Agg, BinaryOp, 128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 512) {
                            kAggRows<Agg, BinaryOp, 256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else {
                            kAggRows<Agg, BinaryOp, 512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        }
                        cutilCheckMsg("agg rows: Kernel execution failed");
                        cudaThreadSynchronize();
                        width = numBlocksX; // only true in reduction agg, but for linear agg this doesn't matter anyway

                        if (prevSum != this) {
                            delete prevSum;
                        }
                        prevSum = nvSumAccum;
                    }
                }
            }
        } else {
            copy(target);
        }
    }
}

void NVMatrix::inRangeInc(float lower, float upper) {
    inRangeInc(lower, upper, *this);
}
void NVMatrix::inRangeInc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<false>(lower, upper), target);
}

void NVMatrix::inRangeExc(float lower, float upper) {
    inRangeExc(lower, upper, *this);
}

void NVMatrix::inRangeExc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<true>(lower, upper), target);
}

void NVMatrix::biggerThanScalar(float scalar) {
    biggerThanScalar(scalar, *this);
}

void NVMatrix::biggerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::BiggerThanScalar(scalar), target);
}

void NVMatrix::smallerThanScalar(float scalar) {
    smallerThanScalar(scalar, *this);
}

void NVMatrix::smallerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::SmallerThanScalar(scalar), target);
}

void NVMatrix::addScalar(float scaleThis, float scalar, NVMatrix& target) {
    apply(NVMatrixOps::WeightedAddScalar(scaleThis, scalar), target);
}

void NVMatrix::addScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::AddScalar(scalar), target);
}

void NVMatrix::addScalar(float scalar) {
    addScalar(scalar, *this);
}

void NVMatrix::minWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MinWithScalar(scalar), target);
}

void NVMatrix::minWithScalar(float scalar) {
    minWithScalar(scalar, *this);
}

void NVMatrix::maxWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MaxWithScalar(scalar), target);
}

void NVMatrix::maxWithScalar(float scalar) {
    maxWithScalar(scalar, *this);
}

void NVMatrix::pow(float p, NVMatrix& target) {
    apply(NVMatrixOps::Pow(p), target);
}

void NVMatrix::pow(float p) {
    pow(p, *this);
}

void NVMatrix::scale(float _scale) {
    scale(_scale, *this);
}

void NVMatrix::scale(float _scale, NVMatrix& target) {
    if (_scale != 1 || &target != this) { // optimize away scale by 1
        apply(NVMatrixOps::MultByScalar(_scale), target);
    }
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp op) {
    NVMatrix *sumVec = new NVMatrix();
    _aggregate<Agg, BinaryOp>(axis, *sumVec, agg, op);
    return *sumVec;
}


void NVMatrix::max(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

void NVMatrix::addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleSum));
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::SecondScaled(scaleSum));
    }
}

void NVMatrix::sum(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

void NVMatrix::min(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::max(int axis) {
    return _aggregate(axis, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::sum(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::min(int axis) {
    return _aggregate(axis, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

void NVMatrix::_sum_setParams(int n, dim3* blocks, dim3* threads, int* numCols) {
    int logn = int(ceil(log(double(n)) / log(2)));
    *numCols = DIVUP(n, logn);
    int numThreads = *numCols;
    *blocks = dim3(DIVUP(numThreads, DP_BLOCKSIZE));
    *threads = dim3(DP_BLOCKSIZE);
}

float NVMatrix::mean() {
    return sum() / getNumElements();
}

float NVMatrix::sum() {
    return _totalAgg(NVMatrixAggs::Sum());
}

float NVMatrix::max() {
    return _totalAgg(NVMatrixAggs::Max());
}

float NVMatrix::min() {
    return _totalAgg(NVMatrixAggs::Min());
}

template<class Agg>
float NVMatrix::_totalAgg(Agg agg) {
    assert(isContiguous());
    dim3 blocks, threads;
    int numCols;
    // Sum most of it on GPU
    NVMatrix* src = this;
    for (NVMatrix* target = NULL; src->getNumElements() > CPUSUM_MAX; src = target) {
        _sum_setParams(src->getNumElements(), &blocks, &threads, &numCols);
        target = new NVMatrix(1, blocks.x);
        kTotalAgg<<<blocks, threads>>>(src->getDevData(), target->getDevData(), numCols, src->getNumElements(), agg);
        cutilCheckMsg("kTotalAgg: Kernel execution failed");
        cudaThreadSynchronize(); // not really necessary?
        delete (src == this ? NULL : src);
    }

    Matrix srcCPU(src->getNumRows(), src->getNumCols());
    src->copyToHost(srcCPU);
    if (src->getNumElements() > 1) { // Sum remainder on CPU
        delete (src == this ? NULL : src);
        if (typeid(Agg) == typeid(NVMatrixAggs::Sum)) {
            return srcCPU.sum();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Max)) {
            return srcCPU.max();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Min)) {
            return srcCPU.min();
        } else {
            assert(false);
        }
    }
    return srcCPU(0,0);
}

/*
 * Fast dot product only for matrices with same transposedness.
 */
float NVMatrix::dotProduct(NVMatrix& b) {
    assert(isContiguous() && b.isContiguous());
    assert(isSameDims(b));
    assert(isTrans() == b.isTrans()); // see?
    dim3 blocks, threads;
    int numCols;
    _sum_setParams(getNumElements(), &blocks, &threads, &numCols);
    NVMatrix target(1, blocks.x);
    kDotProduct_r<<<blocks, threads>>>(getDevData(), b.getDevData(), target.getDevData(), numCols, getNumElements());
    cutilCheckMsg("kDotProduct: Kernel execution failed");
    cudaThreadSynchronize();
    return target.sum();
}

float NVMatrix::norm2() {
    return dotProduct(*this);
}

float NVMatrix::norm() {
    return sqrt(norm2());
}

void NVMatrix::print(int startRow, int rows, int startCol, int cols) const {
    cudaThreadSynchronize();
    Matrix hm = Matrix(_numRows, _numCols);
    copyToHost(hm);
    hm.print(startRow, rows, startCol, cols);
}

void NVMatrix::print(int rows, int cols) const {
    print(0, rows, 0, cols);
}

void NVMatrix::printShape(const char* name) const {
    printf("%s: %dx%d\n", name, _numRows, _numCols);
}
