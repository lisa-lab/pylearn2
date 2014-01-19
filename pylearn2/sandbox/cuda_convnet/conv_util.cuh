/*
 * There has been some new functions added to the original code.
 *
 * Modifications copyright:
 *  Copyright 2010-2013, Universite de Montreal
 *  Credits: Mehdi Mirza
 *  License: 3-clause BSD
 *  mainainer: Mehdi Mirza
 *  email: mirzamom@iro
 *
 * Orginal copyright:
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

#ifndef CONV_UTIL_CUH
#define	CONV_UTIL_CUH

#include <nvmatrix.cuh>

#ifdef _WIN32
#ifdef _CONV_UTIL_EXPORT
#define DllExport   __declspec( dllexport )
#else
#define DllExport   __declspec( dllimport )
#endif
#else //else _WIN32
#define DllExport
#endif

DllExport void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
DllExport void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

DllExport void localProbMaxUndo(NVMatrix& maxout_h, NVMatrix& maxout_p, NVMatrix& hGrads, NVMatrix& pGrads, NVMatrix& target_z,
                        NVMatrix& target_t, int subsX, int startX, int strideX, int outputsX, int imgSize,
                        float * gp_iszero, float * gh_iszero);

DllExport void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput);
DllExport void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

DllExport void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
DllExport void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);
DllExport void convContrastNorm(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
DllExport void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& meanDiffs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

DllExport void convGaussianBlur(NVMatrix& images, NVMatrix& filter, NVMatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs);
DllExport void convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput);
DllExport void convBedOfNailsUndo(NVMatrix& actsGrad, NVMatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput);

DllExport void convResizeBilinear(NVMatrix& images, NVMatrix& target, int imgSize, int tgtSize, float scale);
DllExport void convRGBToYUV(NVMatrix& images, NVMatrix& target);
DllExport void convRGBToLAB(NVMatrix& images, NVMatrix& target, bool center);
DllExport void convCrop(NVMatrix& imgs, NVMatrix& target, int imgSize, int tgtSize, int startY, int startX);
DllExport void normalizeLocalWeights(NVMatrix& weights, int numModules, float norm);
DllExport void convTICAGrad(NVMatrix& images, NVMatrix& ticas, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
DllExport void convTICA(NVMatrix& images, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
DllExport void convContrastNormCrossMap(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target,
                             int numFilters, int sizeF, float addScale, float powScale, bool blocked);
DllExport void convResponseNormCrossMapUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeF, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput);
DllExport void convResponseNormCrossMap(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, float addScale,
                              float powScale, bool blocked);

class AvgPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a / regionSize;
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fmaxf(a, b);
    }
    __device__ inline float getBaseValue() const {
        return -2e38; 
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

class MaxAbsPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return 0.0f;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool2(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX,
                           const int outputsX, Agg agg) {
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
    const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
//    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int myX = threadIdx.y % 4;
    const int myY = threadIdx.y / 4;
    
    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
    
    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;
    
    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSize + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], shImgs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
                }
            }
        }
    }
}


/*
Probabilistic max pooling
*/

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kProbPool(float* imgs, float* top_down, float* ptargets, float* htargets, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX, const int outputsX) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    htargets += myFilterIdx * imgPixels * numImages + imgIdx;
    top_down += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    ptargets += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    MaxPooler max_pooler;
    max_pooler = MaxPooler();
    AvgPooler  avg_pooler;
    avg_pooler = AvgPooler();

    float prod[filtersPerThread][imgsPerThread];
    float denom[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = max_pooler.getBaseValue();
            denom[f][i] = avg_pooler.getBaseValue();
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);


    // get the max
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {

                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = max_pooler(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }

    // top down value for max
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = max_pooler(prod[f][i], -top_down[f * numOutputs * numImages + i * B_X]);
            }
        }
    }

    // get the denom
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        denom[f][i] = avg_pooler(denom[f][i],
                                        __expf(imgs[(f * imgPixels + imgPx) * numImages + i * B_X] - prod[f][i]));
                    }
                }
            }
        }
    }

    //add top down value to denom
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                denom[f][i] = avg_pooler(denom[f][i], __expf(-top_down[f * numOutputs * numImages + i * B_X] - prod[f][i]));
            }
        }
    }

    // get P
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                float off_pt = __expf(-top_down[f * numOutputs * numImages + i * B_X] - prod[f][i]);
                ptargets[f * numOutputs * numImages + i * B_X]  = 1. - off_pt / denom[f][i];
            }
        }
    }

    // get h
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        htargets[(f * imgPixels + imgPx) * numImages + i * B_X] = __expf(imgs[(f * imgPixels + imgPx) * numImages + i * B_X] - prod[f][i]) / denom[f][i];
                    }
                }
            }
        }
    }
}


/*
 * Stochastic Max Pool
 */

__device__ inline void normalize(float* arr, const int arrSize) {
    float max = 0;
    float sum = 0;

    for (int i = 0; i < arrSize; i++) {
        sum += arr[i];
    }

    sum = 1 / sum;
    for (int i = 0; i < arrSize; i++) {
        arr[i] = arr[i] * sum;
    }
}

__device__ inline void multinomial(float* arr, const int arrSize, float rnd) {

    float sum = 0, prevSum = 0;
    for (int i = 0; i < arrSize; i++) {
        sum += arr[i];
        arr[i] = rnd >= prevSum && rnd < sum;
        prevSum = sum;
    }
}

__device__ inline float MaskMax(float* arr, float* mask, const int arrSize) {
    //float max = 0;
    for (int i = 0; i < arrSize; i++) {
        if (mask[i] != 0) {
            return arr[i];
        }
    }
    //TODO rasie error here
    return 0;
}

__device__ inline float WeightedSum(float* arr, float* weight, const int arrSize) {
    float sum = 0;
    for (int i = 0; i < arrSize; i++) {
        sum += arr[i] * weight[i];
    }
    return sum;
}

__global__ void setup_kernel(curandState *state, float *seed)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pitch = blockDim.x * gridDim.x;
    int idx = x + y * pitch;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(*seed, idx, 0, &state[idx]);
}

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalSotchasticMaxPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg, curandState *state) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
   
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pitch = blockDim.x * gridDim.x;
    int idx = x + y * pitch;
    curandState localState = state[idx];
 
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    const int regionX = (loopEndX - loopStartX);
    const int regionY = (loopEndY - loopStartY);
    // TODO: Maybe can allocate this in the shared memory?
    float* window = (float *)malloc(regionSize * sizeof(float));
    float* pxl_value = (float *)malloc(regionSize * sizeof(float));
    float rnd;

    
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            for (int f = 0; f < filtersPerThread; f++) {
                for (int y = loopStartY; y < loopEndY; y++) {
                    for (int x = loopStartX; x < loopEndX; x++) {
                        const int imgPx = y * imgSize + x;
                        window[(y - loopStartY) * regionX + (x - loopStartX)] = imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
                        pxl_value[(y - loopStartY) * regionX + (x - loopStartX)] = imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
                    }
                }
                normalize(window, regionSize);
                rnd = curand_uniform(&localState);
                multinomial(window, regionSize, rnd);
                target[f * numOutputs * numImages + i * B_X] = MaskMax(pxl_value, window, regionSize);
            }
        }
    }
    state[idx] = localState;
    free(window);
    free(pxl_value);
}

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalWeightedPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
   
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    const int regionX = (loopEndX - loopStartX);
    const int regionY = (loopEndY - loopStartY);
    // TODO: Maybe can allocate this in the shared memory?
    float* window = (float *)malloc(regionSize * sizeof(float));
    float* pxl_value = (float *)malloc(regionSize * sizeof(float));

    
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            for (int f = 0; f < filtersPerThread; f++) {
                for (int y = loopStartY; y < loopEndY; y++) {
                    for (int x = loopStartX; x < loopEndX; x++) {
                        const int imgPx = y * imgSize + x;
                        window[(y - loopStartY) * regionX + (x - loopStartX)] = imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
                        pxl_value[(y - loopStartY) * regionX + (x - loopStartX)] = imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
                    }
                }
                normalize(window, regionSize);
                target[f * numOutputs * numImages + i * B_X] = WeightedSum(pxl_value, window, regionSize);
            }
        } 
    }
    free(window);
    free(pxl_value);
}




/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void convLocalPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt((double)imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);

    if (strideX == 1 && subsX >= 6) {
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        assert(numFilters % filtersPerThread == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsX, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        }
    } else {
        
        int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        }

    }

    cutilCheckMsg("convLocalPool: kernel execution failed");
}

/*
Probabilistic Max Pool
*/
template<class Pooler>
void probabilisticPool(NVMatrix& images, NVMatrix& top_down, NVMatrix& ptargets, NVMatrix& htargets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt((double)imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(!top_down.isTrans());
    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    
    int outputs = outputsX * outputsX;
    ptargets.resize(numFilters*outputs, numImages);
    htargets.resize(numFilters*imgSize*imgSize, numImages);

    int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);



    cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 1, 2, true>, cudaFuncCachePreferShared);
    kProbPool<Pooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);


    if (imgsPerThread == 4) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 4, 2, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 4, 2, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        }
    } else if (imgsPerThread == 2) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 2, 2, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        }
    } else {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);

            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            } else {
                cudaFuncSetCacheConfig(kProbPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferShared);
                kProbPool<Pooler, 4, 32, 1, 2, false><<<blocks, threads>>>(images.getDevData(), top_down.getDevData(),
                                                    ptargets.getDevData(), htargets.getDevData(), imgSize,
                                                    numFilters, numImages, subsX, startX, strideX, outputsX);
            }
        }
    }


    cutilCheckMsg("kProbPool: kernel execution failed");
}

/*
Stochastic Max Pool
*/

template<class Pooler>
void convLocalStochasticMaxPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler, float * seed) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt((double)imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);

    int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
    //seed = time(NULL);
    //float* h_C = (float*)malloc(sizeof(float));
    //cudaMemcpy(seed, h_C, sizeof(float), cudaMemcpyDeviceToHost);
    //printf("\nSEED %d\n", *h_C);
    curandState *devStates;
    cudaMalloc((void **)&devStates, blocks.x * blocks.y * threads.x * threads.y * sizeof(curandState));
    // Turn this on for seeding
    setup_kernel<<<blocks, threads>>>(devStates, seed);

    if (imgsPerThread == 4) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        }
    } else if (imgsPerThread == 2) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        }
    } else {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler, devStates);
            }
        }
    }
    cudaFree(devStates);
    cutilCheckMsg("convLocalPool: kernel execution failed");
}

/*
Weighted Pool
*/

template<class Pooler>
void convLocalWeightedPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt((double)imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());

    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);

    int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);

    if (imgsPerThread == 4) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        }
    } else if (imgsPerThread == 2) {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 2, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 2, 2, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        }
    } else {
        if (filtersPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 1, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            } else {
                cudaFuncSetCacheConfig(kLocalSotchasticMaxPool<Pooler, 4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                kLocalWeightedPool<Pooler, 4, 32, 1, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
                                                                  imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
            }
        }
    }
    cutilCheckMsg("convLocalPool: kernel execution failed");
}


#endif	/* CONV_UTIL_CUH */

