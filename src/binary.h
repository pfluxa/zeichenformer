#ifndef BINARY_TOKENIZER_H
#define BINARY_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>

typedef struct __attribute__((aligned(8))) {
    int num_bits;
    double min_val;
    double max_val;
    bool fitted;
    int offset;
} BinaryTokenizer;

// Initialize tokenizer
void binary_init(BinaryTokenizer* t, int num_bits, int offset);

// Fit to data (calculate min/max)
void binary_fit(BinaryTokenizer* t, const double* values, size_t n);

// Encode value into tokens
void binary_encode(const BinaryTokenizer* t, double value, int* indices, int* count);

// Decode tokens into value
double binary_decode(const BinaryTokenizer* t, const int* indices, int count);

#endif