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

#ifndef MATRIX_FUNCS_H_
#define MATRIX_FUNCS_H_

#include <stdlib.h>
#include <math.h>
#include <algorithm>

#ifdef DOUBLE_PRECISION
#define MTYPE double
#else
#define MTYPE float
#endif

#define MYRAND ((double)rand() / ((double)RAND_MAX + 1))

inline MTYPE _zero(MTYPE x) {
    return 0;
}

inline MTYPE _one(MTYPE x) {
    return 1;
}

inline MTYPE _abs(MTYPE x) {
    return x > 0 ? x : -x;
}

inline MTYPE _square(MTYPE x) {
    return x * x;
}

inline MTYPE _sigma1(MTYPE x) {
    return (tanh(x / 2) + 1) / 2;
}

inline MTYPE _sigma2(MTYPE x) {
    return 1 / (1 + exp(-x));
}

inline MTYPE _recip(MTYPE x) {
    return 1 / x;
}

inline MTYPE _exp(MTYPE x) {
    return exp(x);
}

inline MTYPE _log(MTYPE x) {
    return log(x);
}

inline MTYPE _tanh(MTYPE x) {
    return tanh(x);
}

inline MTYPE _sign(MTYPE x) {
    return x > 0 ? 1 : -1;
}

inline MTYPE _rand(MTYPE x) {
    return MYRAND;
}

inline MTYPE _divide(MTYPE x, MTYPE y) {
    return x / y;
}

inline MTYPE _mult(MTYPE x, MTYPE y) {
    return x * y;
}

inline MTYPE _add(MTYPE x, MTYPE y) {
    return x + y;
}

inline MTYPE _addSquare(MTYPE x, MTYPE y) {
    return x*x + y;
}

inline MTYPE _addWithScale(MTYPE x, MTYPE y, MTYPE scale) {
    return x + scale*y;
}

inline MTYPE _max(MTYPE x, MTYPE y) {
    return std::max(x, y);
}

inline MTYPE _min(MTYPE x, MTYPE y) {
    return std::min(x, y);
}

inline MTYPE _bigger(MTYPE x, MTYPE y) {
    return x > y;
}

inline MTYPE _smaller(MTYPE x, MTYPE y) {
    return x < y;
}

inline MTYPE _equal(MTYPE x, MTYPE y) {
    return x == y;
}

inline MTYPE _notEqual(MTYPE x, MTYPE y) {
    return x != y;
}

#endif /* MATRIX_FUNCS_H_ */
