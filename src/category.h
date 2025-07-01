#ifndef CATEGORY_TOKENIZER_H
#define CATEGORY_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>

typedef struct __attribute__((aligned(8))) {
    char** categories;
    size_t num_categories;
    bool fitted;
    int offset;
} CategoryTokenizer;

// Initialize tokenizer
void category_init(CategoryTokenizer* t, int offset);

// Fit to data (extract unique categories)
void category_fit(CategoryTokenizer* t, const char** values, size_t n);

// Encode value into tokens
int category_encode(const CategoryTokenizer* t, const char* value);

// Decode token into value
const char* category_decode(const CategoryTokenizer* t, int token);

// Free resources
void category_free(CategoryTokenizer* t);

#endif