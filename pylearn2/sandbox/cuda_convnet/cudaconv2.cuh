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

#ifndef COMMON_CUH
#define	COMMON_CUH

//#include <cutil_inline.h>
#include <nvmatrix.cuh>
#include "conv_util.cuh"

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int partialSum);
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int partialSum,
                    float scaleTargets, float scaleOutput);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void convFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups);
void convFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups,
                          float scaleTargets, float scaleOutput);

void localFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups,
                          float scaleTargets, float scaleOutput);
void localFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups);

void convWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                         int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                         int numImgColors, int numFilterColors, int numGroups);
void convWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numFilterColors,
                        int numGroups, int partialSum, float scaleTargets, float scaleOutput);

void localWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                         int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                         int numImgColors, int numFilterColors, int numGroups);
void localWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numFilterColors,
                        int numGroups, float scaleTargets, float scaleOutput);

void convImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups);
void convImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups,
                       float scaleTargets, float scaleOutput);

void localImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups);
void localImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups,
                       float scaleTargets, float scaleOutput);


#endif	/* COMMON_CUH */

