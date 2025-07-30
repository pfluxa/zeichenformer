#include "timestamp.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Forward declaration for internal helper function
static bool timestamp_parse(const char* iso, struct tm* tm);

void timestamp_init(TimestampTokenizer* t, int min_year, int max_year, int offset) {
    t->min_year = min_year;
    t->max_year = max_year;
    t->offset = offset;
    // This tokenizer is configured, not data-fitted, so it's always "fitted".
    t->fitted = true;
}

// Internal helper to parse an ISO 8601 timestamp.
static bool timestamp_parse(const char* iso, struct tm* tm) {
    if (iso == NULL || tm == NULL) {
        return false;
    }
    memset(tm, 0, sizeof(struct tm));

    const char* separator = strchr(iso, 'T');
    if (separator == NULL) separator = strchr(iso, ' ');
    if (separator == NULL) return false;

    // YYYY-MM-DD must be 10 chars
    if ((size_t)(separator - iso) != 10) return false;
    
    // Parse date and time parts
    float seconds;
    int matched = sscanf(iso, "%4d-%2d-%2d", &tm->tm_year, &tm->tm_mon, &tm->tm_mday);
    if (matched != 3) return false;

    matched = sscanf(separator + 1, "%2d:%2d:%f", &tm->tm_hour, &tm->tm_min, &seconds);
    if (matched != 3) return false;
    tm->tm_sec = (int)seconds; // Truncate fractional part

    // --- CRITICAL FIX: Adjust struct tm fields to standard C conventions ---
    tm->tm_year -= 1900; // tm_year is years since 1900
    tm->tm_mon -= 1;     // tm_mon is 0-11

    // Basic range validation
    if ((tm->tm_year + 1900) < 1 ||
        tm->tm_mon < 0 || tm->tm_mon > 11 ||
        tm->tm_mday < 1 || tm->tm_mday > 31 ||
        tm->tm_hour < 0 || tm->tm_hour > 23 ||
        tm->tm_min < 0 || tm->tm_min > 59 ||
        tm->tm_sec < 0 || tm->tm_sec > 60) { // 60 for leap seconds
        return false;
    }
    return true;
}

void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count) {
    struct tm tm;

    // Use a clearer error signal if parsing fails or tokenizer isn't fitted
    if (!t->fitted || !timestamp_parse(iso, &tm)) {
        *count = -1; // Disambiguated error signal
        return;
    }
    *count = 0;
    
    // Check if parsed year is within the tokenizer's configured range
    if ((tm.tm_year + 1900) < t->min_year || (tm.tm_year + 1900) > t->max_year) {
         *count = -1; // Year out of bounds
        return;
    }

    // Correctly encode using the adjusted tm values
    tokens[(*count)++] = (tm.tm_year + 1900 - t->min_year) + t->offset;
    tokens[(*count)++] = (tm.tm_mon + 1) + t->offset; // Month (1-12)
    tokens[(*count)++] = tm.tm_mday + t->offset;       // Day (1-31)
    tokens[(*count)++] = tm.tm_hour + t->offset;       // Hour (0-23)
    tokens[(*count)++] = tm.tm_min + t->offset;        // Minute (0-59)
    tokens[(*count)++] = tm.tm_sec + t->offset;        // Second (0-60)
}

void timestamp_decode(const TimestampTokenizer* t, const int* tokens, int count, char* output, size_t output_size) {
    // We expect exactly 6 tokens
    if (count != 6 || !t->fitted) {
        snprintf(output, output_size, "0000-00-00T00:00:00");
        return;
    }

    // Decode tokens back to raw values by subtracting the offset
    int year_val = (tokens[0] - t->offset) + t->min_year;
    int month_val = tokens[1] - t->offset;
    int day_val = tokens[2] - t->offset;
    int hour_val = tokens[3] - t->offset;
    int min_val = tokens[4] - t->offset;
    int sec_val = tokens[5] - t->offset;

    // Validate the decoded ranges
    if (year_val < t->min_year || year_val > t->max_year ||
        month_val < 1 || month_val > 12 ||
        day_val < 1 || day_val > 31 ||
        hour_val < 0 || hour_val > 23 ||
        min_val < 0 || min_val > 59 ||
        sec_val < 0 || sec_val > 60) { // FIX: Allow up to 60 for leap seconds
        snprintf(output, output_size, "0000-00-00T00:00:00");
        return;
    }

    // Safely format the output string
    snprintf(output, output_size, "%04d-%02d-%02dT%02d:%02d:%02d",
             year_val, month_val, day_val,
             hour_val, min_val, sec_val);
}