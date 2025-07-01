#ifndef TIMESTAMP_TOKENIZER_H
#define TIMESTAMP_TOKENIZER_H

#include <stdbool.h>
#include <time.h>

typedef struct __attribute__((aligned(8))) {
    int min_year;
    int max_year;
    bool fitted;
    int offset;
} TimestampTokenizer;

// Initialize tokenizer
void timestamp_init(TimestampTokenizer* t, int min_year, int max_year, int offset);

// Encode timestamp into tokens
void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count);

// Decode tokens into ISO 8601 string
void timestamp_decode(const TimestampTokenizer* t, const int* tokens, int count, char* output);

#endif