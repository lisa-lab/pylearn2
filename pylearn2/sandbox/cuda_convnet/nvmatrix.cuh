/* 
 * This file was copied from cuda-convnet by Alex Krizhevsky
 * It has been modified slightly by Ian Goodfellow for use with
 * theano / pylearn2
 *
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
#ifndef NVMATRIX_H_
#define NVMATRIX_H_

//#define RND_MULTIPLIERS_FILE ("rnd_multipliers_32bit.txt")

#ifndef RND_MULTIPLIERS_FILE
#define RND_MULTIPLIERS_FILE ("rnd_multipliers_32bit.txt")
#endif

#include <pthread.h>
#include <map>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cutil_inline.h>
#include <time.h>
#include <curand_kernel.h>

#include <Python.h>
#include <cuda_ndarray.cuh>

//Commented by Ian Goodfellow-- we don't actually need this dependency, it just increase theano compile times
//#include <matrix.h>
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

#ifdef WARNINGS
#define WARN(msg) printf("WARN: File %s, line %d: %s\n", __FILE__, __LINE__, msg);
#else
#define WARN(msg) ;
#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

#ifdef _WIN32
#ifdef _NVMATRIX_EXPORT
#define DllExport   __declspec( dllexport )
#else
#define DllExport   __declspec( dllimport )
#endif
#else //else _WIN32
#define DllExport
#endif

class DllExport NVMatrix {
private:
    int _numCols, _numRows;
    int _numElements;
    int _stride;
    float* _devData;
    bool _isTrans;
    bool _ownsData;

//    static std::map<int,curandGenerator_t> rndGen;
    static std::map<int,curandState*> rndDevStates;
    static pthread_mutex_t *_rndMutex;

    static void checkCublasError(cublasStatus_t status, const char* msg) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, msg, NULL);
            exit(EXIT_FAILURE);
        }
    }

    char getTransChar() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed krizhevsky matrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? 'n' : 't';
    }
    cublasOperation_t getTransOp() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed krizhevsky matrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? CUBLAS_OP_N : CUBLAS_OP_T;
    }

    void _init(int numRows, int numCols);
    void _init(int numRows, int numCols, int stride, bool isTrans);
    void _sum_setParams(int n, dim3* blocks, dim3* threads, int* numCols);
    template<class Agg> float _totalAgg(Agg agg);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp op);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp op);
    template <class Randomizer> void _unaryRandomize(NVMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd);   
public:
    NVMatrix();
    NVMatrix(bool isTrans);
    NVMatrix(int numRows, int numCols, bool isTrans=false);
    // Commented by IG. Depends on Matrix
    //NVMatrix(const Matrix& like, bool copy);
    NVMatrix(const NVMatrix& like, bool copy);

    //Constructor added by Ian Goodfellow. Make a view of a CudaNdarray.
    NVMatrix(const CudaNdarray * view, int numRows, int numCols, const char * msg);

    NVMatrix(const NVMatrix& like);
    // Commented by IG. Depends on Matrix
    // NVMatrix(const Matrix& like);
    NVMatrix(float* devData, int numRows, int numCols, int stride, bool isTrans);
    ~NVMatrix();

    // static void initRandom(unsigned long long seed);
    // static void initRandom();
    static int getDeviceID();
    static bool isRndInitialized();
    static curandState* getCurandState();
    static void destroyRandom();
    static pthread_mutex_t* makeMutex();

    /*
     * DO NOT DEREFERENCE IN HOST CODE! This is a device memory pointer.
     */
    float* getCellPtr(int i, int j) const {
        if (_isTrans) {
            return &_devData[j * _numRows + i];
        }
        return &_devData[i * _numCols + j];
    }

    // Commented by IG. Depends on Matrix
    // bool isSameDims(const Matrix& m) const {
    //    return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    //}

    bool isSameDims(const NVMatrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    int getNumRows() const {
        return _numRows;
    }

    int getNumCols() const {
        return _numCols;
    }

    int getStride() const {
        return _stride;
    }

    int getLeadingDim() const {
        return _isTrans ? _numRows : _numCols;
    }

    int getFollowingDim() const {
        return !_isTrans ? _numRows : _numCols;
    }

    /*
     * FALSE:    Row-major order.
     * TRUE:     Column-major order.
     */
    bool isTrans() const {
        return _isTrans;
    }

    bool isView() const {
        return !_ownsData;
    }

    float* getDevData() const {
        return _devData;
    }

    unsigned int getNumElements() const {
        return _numElements;
    }

    /*
     * Only use if you know what you're doing!
     * Does not actually transpose matrix.
     */
    void setTrans(bool trans) {
        if (trans != _isTrans) {
            assert(isContiguous());
            _isTrans = trans;
            _stride = getLeadingDim();
        }
    }
    
    /*
     * Only use if you know what you're doing!
     * This toggles whether this object will free its GPU memory when it's destroyed.
     */
    void setView(bool isView) {
        _ownsData = !isView;
    }

    bool isContiguous() const {
        return _stride == getLeadingDim() || getFollowingDim() == 1;
    }
    
    void truncate() {
        resize(0,0);
    }
   
    // Commented by IG. Depends on Matrix
    // void copyFromHost(const Matrix& hostMatrix);
    // void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);
    // void copyToHost(Matrix& hostMatrix) const;
    // void copyToHost(Matrix& hostMatrix, bool resizeTarget) const;
    void copy(NVMatrix& dest) const;
    NVMatrix& copy() const;
    void addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB);
    void addProduct(const NVMatrix& a, const NVMatrix &b);
    void rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, float scaleAB);
    void randomizeUniform();
    void addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target);
    void addGaussianNoise(float stdev, NVMatrix& target);
    void addGaussianNoise(NVMatrix& stdevs, bool var);
    void addGaussianNoise(NVMatrix& stdevs);
    void addGaussianNoise(float stdev);
    void addGaussianNoise();
    void randomizeGaussian();
    void randomizeGaussian(float stdev);
    void randomizeGaussian(float mean, float stdev);
    void randomizeGaussian(NVMatrix& stdevs);
    void randomizeGaussian(NVMatrix& stdevs, NVMatrix& target);
    void binarizeProbs();
    void binarizeProbs(NVMatrix& target);

    void biggerThan(NVMatrix& m, NVMatrix& target);
    void biggerThan(NVMatrix& m);
    void biggerThanVector(NVMatrix& vec, NVMatrix& target);
    void biggerThanVector(NVMatrix& vec);
    void equals(NVMatrix& m, NVMatrix& target);
    void equals(NVMatrix& m);

    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    NVMatrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const;
    NVMatrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, NVMatrix& target) const;
    NVMatrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, NVMatrix& target) const;

    template <class Op> void apply(Op op, NVMatrix& target) {
        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }
        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        if (target.isTrans() == isTrans()) {
            kEltwiseUnaryOp<Op><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            cutilCheckMsg("kEltwiseUnaryOp: Kernel execution failed");
        } else {
            bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
            if (checkBounds) {
                kEltwiseUnaryOpTrans<Op, true><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            } else {
                kEltwiseUnaryOpTrans<Op, false><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            }
            cutilCheckMsg("kEltwiseUnaryOpTrans: Kernel execution failed");
        }
    }
    
    template <class Op> void apply(Op op) {
        apply(op, *this);
    }
    
    template <class Op> void applyBinary(Op op, NVMatrix& b) {
        applyBinary(op, b, *this);
    }

    template <class Op> void applyBinary(Op op, NVMatrix& b, NVMatrix& target) {
        assert(this->isSameDims(b));

        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }

        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                    std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        if (target.isTrans() == isTrans() && target.isTrans() == b.isTrans()) {
            kEltwiseBinaryOp<Op><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width, getStride(),
                                                      b.getStride(), target.getStride(), op);
            cutilCheckMsg("kEltwiseBinaryOp: Kernel execution failed");
        } else {
            //  both x here since y divides x
            bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
            if (target.isTrans() == isTrans() && target.isTrans() != b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,false,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,false,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                }
            } else if (target.isTrans() != isTrans() && target.isTrans() != b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,true,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,true,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                }
            } else if (target.isTrans() != isTrans() && target.isTrans() == b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,false,true><<<blocks, threads>>>(b._devData, _devData, target._devData, height, width,b.getStride(),
                                                               getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,false,true><<<blocks, threads>>>(b._devData, _devData, target._devData, height, width, b.getStride(),
                                                               getStride(), target.getStride(), op);
                }
            }
            cutilCheckMsg("kEltwiseBinaryOpTrans: Kernel execution failed");
        }
    }
    
    template <class Op> void applyTernary(Op op, NVMatrix& b, NVMatrix& c, NVMatrix& target) {
        assert(this->isSameDims(b));
        assert(this->isSameDims(c));
        // For now ternary ops are only supported for matrices of same transposedness
        assert(isTrans() == b.isTrans());
        assert(isTrans() == c.isTrans());
        if (!target.isSameDims(*this) || target.isTrans() != isTrans()) {
            target.resize(*this);
        }

        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                    std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        kEltwiseTernaryOp<Op><<<blocks, threads>>>(_devData, b._devData, c._devData, target._devData, height, width,
                                                   getStride(), b.getStride(), c.getStride(), target.getStride(), op);
        cutilCheckMsg("kEltwiseTernaryOp: Kernel execution failed");
    }

    bool resize(int numRows, int numCols);
    bool resize(const NVMatrix &like);
    // Commented by IG. Depends on Matrix
    // bool resize(const Matrix &like);
    void reshape(int numRows, int numCols);
    NVMatrix& reshaped(int numRows, int numCols);
    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol) const;
    void add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, NVMatrix& target);
    void add(NVMatrix& b, float scaleB);
    void add(NVMatrix& b, float scaleA, float scaleB);
    void add(NVMatrix& b);
    void eltwiseMult(NVMatrix& b);
    void eltwiseMult(NVMatrix& b, NVMatrix& target);
    void eltwiseDivide(NVMatrix& b);
    void eltwiseDivide(NVMatrix& b, NVMatrix& target);
    void squaredDiff(NVMatrix& b);
    void squaredDiff(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b);
    void addVector(NVMatrix& vec, float scaleVec, NVMatrix& target);
    void addVector(NVMatrix& vec);
    void addVector(NVMatrix& vec, float scaleVec);
    void addVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec);
    void eltwiseMultByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseMultByVector(NVMatrix& vec);
    void eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseDivideByVector(NVMatrix& vec);
    void tile(int timesY, int timesX, NVMatrix& target);

    void sum(int axis, NVMatrix& target);
    void addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum);
    // Commented by IG NVMatrix& max(int axis);
    void max(int axis, NVMatrix& target);
    /* Commented by IG. Depends on _aggregate
    NVMatrix& sum(int axis);
    void min(int axis, NVMatrix& target);
    NVMatrix& min(int axis);
    */
    // Commented by IG. Depends on sum
    // float mean();
    // Commented by IG float max();
    /* Commented by IG. Depends on _totalAgg
    float sum();
    float min();
    Depend on dotProduct:
    float norm2();
    float norm();
    */
    
    void inRangeInc(float lower, float upper);
    void inRangeInc(float lower, float upper, NVMatrix& target);
    void inRangeExc(float lower, float upper);
    void inRangeExc(float lower, float upper, NVMatrix& target);
    void biggerThanScalar(float scalar);
    void biggerThanScalar(float scalar, NVMatrix& target);
    void smallerThanScalar(float scalar);
    void smallerThanScalar(float scalar, NVMatrix& target);
    void addScalar(float scaleThis, float scalar, NVMatrix& target);
    void addScalar(float scalar, NVMatrix& target);
    void addScalar(float scalar);
    void minWithScalar(float scalar, NVMatrix& target);
    void minWithScalar(float scalar);
    void maxWithScalar(float scalar, NVMatrix& target);
    void maxWithScalar(float scalar);
    void pow(float p, NVMatrix& target);
    void pow(float p);
    void scale(float _scale);
    void scale(float _scale, NVMatrix& target);

    // Commented by IG. Depends on sum
    // float dotProduct(NVMatrix& b);

    /*
     * Does SOFT transpose and returns result, leaving this matrix unchanged
     */
    NVMatrix& getTranspose();

    /*
     * Does HARD transpose and puts result in target
     */
    void transpose(NVMatrix& target);

    /*
     * Does SOFT transpose
     */
    void transpose();
    bool transpose(bool trans);

    void flipTrans(NVMatrix& target);
    NVMatrix& flipTrans();

    /*
    Commented out by Ian Goodfellow. These methods bring in more dependencies /
    increase the theano compile time, and we don't really need them.
    void print(int startRow, int rows, int startCol, int cols) const;
    void print(int rows, int cols) const;
    */
    void printShape(const char* name) const;

    template <class Op> void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target) {
        assert(&target != &vec); // for now
        assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
        assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
        assert(vec.isContiguous());

        target.resize(*this); // target must be same orientation as me for now

        int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
        int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
        dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);
        dim3 blocks(MIN(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), MIN(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
        if (vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
            kColVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
        } else {
            kRowVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
        }
        cutilCheckMsg("Kernel execution failed");
    //    cudaThreadSynchronize();
    }

    /* Commented by Ian Goodfellow because it depends on _totalAgg
    template<class UnaryOperator> float argMax(UnaryOperator u) {
       return _totalAgg(NVMatrixAggs::ArgMax<UnaryOperator>(u));
    }
    */
};

#endif /* NVMATRIX_H_ */
