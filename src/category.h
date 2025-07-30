#ifndef CATEGORY_H
#define CATEGORY_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char** categories;       // Sorted list of unique category strings
    size_t num_categories;   // The number of learned string categories
    int num_tokens;          // Total codes (num_categories + 2 for missing/unknown)
    int offset;
    bool fitted;
} CategoryTokenizer;

/**
 * @brief Initializes a CategoryTokenizer.
 */
void category_init(CategoryTokenizer* t, int offset);

/**
 * @brief Free allocated memory by a CategoryTokenizer.
 */
void category_free(CategoryTokenizer* t);

/**
 * @brief Fits the tokenizer to a dataset of strings.
 */
void category_fit(CategoryTokenizer* t, const char** values, size_t n);

/**
 * @brief Encodes a single string value into its integer token.
 * @return The integer token, or -1 if the tokenizer is not fitted.
 */
int category_encode(const CategoryTokenizer* t, const char* value);

const char* category_decode(const CategoryTokenizer* t, int token);

#endif // CATEGORY_H