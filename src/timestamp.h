#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h> // Required for struct tm

typedef struct {
    int min_year;
    int max_year;
    int offset;
    bool fitted;
} TimestampTokenizer;

/**
 * @brief Initializes a TimestampTokenizer.
 */
void timestamp_init(TimestampTokenizer* t, int min_year, int max_year, int offset);

/**
 * @brief Encodes an ISO timestamp string into an array of 6 integer tokens.
 */
void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count);

/**
 * @brief Decodes an array of 6 integer tokens back into an ISO timestamp string.
 */
void timestamp_decode(const TimestampTokenizer* t, const int* tokens, int count, char* output, size_t output_size);

#endif // TIMESTAMP_H