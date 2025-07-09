#include "timestamp.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Parse ISO 8601 timestamp
bool timestamp_parse(const TimestampTokenizer* t, const char* iso, struct tm* tm);

void timestamp_init(TimestampTokenizer* t, int min_year, int max_year, int offset) {
    t->min_year = min_year;
    t->max_year = max_year;
    t->fitted = true;
    t->offset = offset;
    // initialize fixed offsets
    // -> year (0 is invalid)
    t->bucket_offsets[0] = 1 + offset;
    // -> month (+ year range)
    t->bucket_offsets[1] = t->bucket_offsets[0] + (max_year - min_year);
    // -> day (+12 months)
    t->bucket_offsets[2] = t->bucket_offsets[1] + 12;
    // -> hour (+31 days)
    t->bucket_offsets[3] = t->bucket_offsets[2] + 31;
    // -> minute (+24 hours)
    t->bucket_offsets[4] = t->bucket_offsets[3] + 24;
    // -> second (+60 minutes)
    t->bucket_offsets[5] = t->bucket_offsets[4] + 60;
    
    // -> total number of tokens (+60 seconds)
    t->num_tokens = t->bucket_offsets[5] + 60 - offset;
}

bool timestamp_parse(const TimestampTokenizer* t, const char* iso, struct tm* tm) {
    if (iso == NULL || tm == NULL) {
        return false;
    }

    memset(tm, 0, sizeof(struct tm));
    
    // Check for either 'T' or space as separator
    const char* separator = strchr(iso, 'T');
    if (separator == NULL) {
        separator = strchr(iso, ' ');
        if (separator == NULL) {
            return false;
        }
    }
    
    // Split into date and time parts
    size_t date_len = separator - iso;
    if (date_len != 10) {  // YYYY-MM-DD should be 10 chars
        return false;
    }
    
    // Parse date part (YYYY-MM-DD)
    if (sscanf(iso, "%4d-%2d-%2d",
               &tm->tm_year, &tm->tm_mon, &tm->tm_mday) != 3) {
        return false;
    }
    
    // Parse time part (HH:MM:SS or HH:MM:SS.sss)
    const char* time_part = separator + 1;
    int matched = sscanf(time_part, "%2d:%2d:%2d",
                        &tm->tm_hour, &tm->tm_min, &tm->tm_sec);
    
    // If we didn't match full time, try with fractional seconds
    if (matched != 3) {
        float seconds;
        matched = sscanf(time_part, "%2d:%2d:%f",
                        &tm->tm_hour, &tm->tm_min, &seconds);
        if (matched == 3) {
            tm->tm_sec = (int)seconds;  // truncate fractional part
        } else {
            return false;
        }
    }
    
    // Validate ranges
    if (tm->tm_year < t->min_year || tm->tm_year > t->max_year ||
        tm->tm_mon < 1 || tm->tm_mon > 12 ||
        tm->tm_mday < 1 || tm->tm_mday > 31 ||
        tm->tm_hour < 0 || tm->tm_hour > 23 ||
        tm->tm_min < 0 || tm->tm_min > 59 ||
        tm->tm_sec < 0 || tm->tm_sec > 60) {  // 60 accounts for leap seconds
        return false;
    }
    
    // Adjust struct tm fields (year is years since 1900, month is 0-11)
    // tm->tm_year -= 1900;
    // tm->tm_mon -= 1;
    
    return true;
}


void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count) {
    *count = 0;
    struct tm tm;
    int last = 0;

    if (!timestamp_parse(t, iso, &tm)) {
        printf("%s\n", iso);
        // Invalid format - mark all components as invalid
        *count = 6;
        tokens[0] = t->bucket_offsets[0];
        tokens[1] = t->bucket_offsets[1];
        tokens[2] = t->bucket_offsets[2];
        tokens[3] = t->bucket_offsets[3];
        tokens[4] = t->bucket_offsets[4];
        tokens[5] = t->bucket_offsets[5];
        return;
    }

    // Year (offset 0)
    tokens[(*count)++] = (tm.tm_year - t->min_year) + t->bucket_offsets[0];
    // Month (offset 1)
    tokens[(*count)++] = tm.tm_mon + t->bucket_offsets[1];
    // Day (offset 2)
    tokens[(*count)++] = tm.tm_mday + t->bucket_offsets[2];
    // Hour (offset 3)
    tokens[(*count)++] = tm.tm_hour + t->bucket_offsets[3];
    // Minute (offset 4)
    tokens[(*count)++] = tm.tm_min + t->bucket_offsets[4];
    // Second (offset 5)
    tokens[(*count)++] = tm.tm_sec + t->bucket_offsets[5];
}

void timestamp_decode(const TimestampTokenizer* t, const int* tokens, int count, char* output) {
    bool valid = true;
    // We expect exactly 6 tokens (year, month, day, hour, minute, second)
    if (count != 6) {
        strcpy(output, "__invalid__");
        return;
    }
    int tt[6];
    tt[0] = (tokens[0] + t->min_year) - t->bucket_offsets[0];
    tt[1] = tokens[1] - t->bucket_offsets[1];
    tt[2] = tokens[2] - t->bucket_offsets[2];
    tt[3] = tokens[3] - t->bucket_offsets[3];
    tt[4] = tokens[4] - t->bucket_offsets[4];
    tt[5] = tokens[5] - t->bucket_offsets[5];
    // year = 0 -> invalid date
    if(tt[0] == 0) {
        strcpy(output, "__invalid__");
        return;
    }
    // Adjust month (0-11 -> 1-12) for output
    snprintf(output, 64, "%04d-%02d-%02dT%02d:%02d:%02d",
            tt[0], tt[1], tt[2],
            tt[3], tt[4], tt[5]
    );
}