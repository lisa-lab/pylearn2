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

#ifndef NVMATRIX_OPERATORS_CUH
#define	NVMATRIX_OPERATORS_CUH

#include <cutil_inline.h>

class NVMatrixOps {
public:
    class Exp {
    public:
        __device__ inline float operator()(const float a) const {
            return __expf(a);
        }
    };

    class Logistic {
    public:
        __device__ inline float operator()(const float a) const {
            return __fdividef(1.0f, 1.0f + __expf(-a));
        }
    };

    class Log {
    public:
        __device__ inline float operator()(const float a) const {
            return __logf(a);
        }
    };

    class Square {
    public:
        __device__ inline float operator()(const float a) const {
            return a * a;
        }
    };

    class Sqrt {
    public:
        __device__ inline float operator()(const float a) const {
            return sqrtf(a);
        }
    };

    class Reciprocal {
    public:
        __device__ inline float operator()(const float a) const {
            return 1.0f / a;
        }
    };

    class Abs {
    public:
        __device__ inline float operator()(const float a) const {
            return a > 0 ? a : -a;
        }
    };

    class Sign {
    public:
        __device__ inline float operator()(const float a) const {
            return (a > 0) - (a < 0);
        }
    };
    
    class Identity {
    public:
        __device__ inline float operator()(const float a) const {
            return a;
        }
    };

    class Zero {
    public:
        __device__ inline float operator()(const float a) const {
            return 0;
        }
    };

    class One {
    public:
        __device__ inline float operator()(const float a) const {
            return 1;
        }
    };
    
    class SmallerThanScalar {
    private:
        const float scalar;
    public:
        SmallerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a < scalar;
        }
    };

    class BiggerThanScalar {
    private:
        const float scalar;
    public:
        BiggerThanScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar;
        }
    };

    class AddScalar {
    private:
        const float scalar;
    public:
        AddScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a + scalar;
        }
    };

    class WeightedAddScalar {
    private:
        const float weight, scalar;
    public:
        WeightedAddScalar(const float _weight, const float _scalar) : weight(_weight), scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return weight * a + scalar;
        }
    };

    class MultByScalar {
    private:
        const float scalar;
    public:
        MultByScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a * scalar;
        }
    };

    class Pow {
    private:
        const float p;
    public:
        Pow(const float _p) : p(_p) {
        }
        __device__ inline float operator()(const float a) const {
            return __powf(a, p);
        }
    };

    template <bool exclusive>
    class InRange {
    private:
        const float lower, upper;
    public:
        InRange(const float _lower, const float _upper) : lower(_lower), upper(_upper) {
        }
        __device__ inline float operator()(const float a) const {
            return exclusive ? a > lower && a < upper : a >= lower && a <= upper;
        }
    };

    class MinWithScalar {
    private:
        const float scalar;
    public:
        MinWithScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar ? scalar : a;
        }
    };

    class MaxWithScalar {
    private:
        const float scalar;
    public:
        MaxWithScalar(const float _scalar) : scalar(_scalar) {
        }
        __device__ inline float operator()(const float a) const {
            return a > scalar ? a : scalar;
        }
    };
};

class NVMatrixBinaryOps {
public:
    class Equals {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a == b;
        }
    };

    class BiggerThan {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b;
        }
    };

    class Divide {
    public:
        __device__ inline float operator()(const float a, const float b) const  {
            return __fdividef(a, b);
        }
    };

    class Multiply {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a * b;
        }
    };

    class SquaredDiff {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return (a - b) * (a - b);
        }
    };

    class WeightedAdd {
    private:
        const float scaleA, scaleB;
    public:
        WeightedAdd(const float _scaleA, const float _scaleB) : scaleA(_scaleA), scaleB(_scaleB) {
        }
        __device__ inline float operator()(const float a, const float b) const {
            return a * scaleA + b * scaleB;
        }
    };

    class Add {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + b;
        }
    };
    
    class First {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a;
        }
    };
    
    class Second {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return b;
        }
    };
    
    class SecondScaled {
    private:
        const float scale;
    public:
        SecondScaled(const float _scale) : scale(_scale) {
        }
        __device__ inline float operator()(const float a, const float b) const {
            return scale * b;
        }
    };
};

class NVMatrixAggs {
public:
    class Sum {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a + b;
        }
        __device__ inline float getBaseValue() {
            return 0;
        }
    };

    class Max {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b ? a : b;
        }
        __device__ inline float getBaseValue() {
            return -2e38;
        }
    };

    class Min {
    public:
        __device__ inline float operator()(const float a, const float b) const {
            return a > b ? b : a;
        }
        __device__ inline float getBaseValue() {
            return 2e38;
        }
    };

    template<class UnaryOperator>
    class ArgMax {
    private:
       UnaryOperator u;
    public:
       ArgMax(UnaryOperator _u) : u(_u) {
       }
       __device__ inline float operator()(const float a, const float b) const {
           return u(a) > u(b) ? a : b;
       }
       __device__ inline float getBaseValue() {
           return u.getArgMin();
       }
    };
};

class NVMatrixTernaryOps {
public:
    class Add {
    public:
        __device__ inline float operator()(const float a, const float b, const float c) const {
            return a + b + c;
        }
    };
};

#endif	/* NVMATRIX_OPERATORS_CUH */

