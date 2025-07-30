#ifndef NUMERICAL_TOKENIZER_H
#define NUMERICAL_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>

typedef struct __attribute__((aligned(8))) {
    int num_bits;
    double min_val;
    double max_val;
    bool fitted;
    int offset;
} NumericalTokenizer;

// Initialize tokenizer
void numerical_init(NumericalTokenizer* t, int num_bits, int offset);

// Fit to data (calculate min/max)
void numerical_fit(NumericalTokenizer* t, const double* values, size_t n);

// Encode value into tokens
void numerical_encode(const NumericalTokenizer* t, double value, int* indices, int* count);

// Decode tokens into value
double numerical_decode(const NumericalTokenizer* t, const int* indices, int count);

#endif