#include "timestamp.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void timestamp_init(TimestampTokenizer* t, int min_year, int max_year) {
    t->min_year = min_year;
    t->max_year = max_year;
    t->fitted = true;
}

bool timestamp_parse(const char* iso, struct tm* tm) {
    memset(tm, 0, sizeof(struct tm));
    return sscanf(iso, "%d-%d-%dT%d:%d:%d",
                 &tm->tm_year, &tm->tm_mon, &tm->tm_mday,
                 &tm->tm_hour, &tm->tm_min, &tm->tm_sec) == 6;
}

void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count) {
    *count = 0;
    struct tm tm;
    
    if (!timestamp_parse(iso, &tm)) {
        // Invalid format - mark all components as invalid
        *count = 6;
        tokens[0] = 0;
        tokens[1] = tokens[0] + 15;
        tokens[2] = tokens[1] + 31;
        tokens[3] = tokens[2] + 24;
        tokens[4] = tokens[3] + 60;
        tokens[5] = tokens[5] + 60;
        return;
    }

    // Year (offset 0)
    if (tm.tm_year < t->min_year) {
        tokens[(*count)++] = 0;  // Below min
    } else if (tm.tm_year > t->max_year) {
        tokens[(*count)++] = 1;  // Above max
    } else {
        tokens[(*count)++] = 2 + (tm.tm_year - t->min_year);
    }

    // Month (offset 1)
    if (tm.tm_mon < 1 || tm.tm_mon > 12) {
        tokens[(*count)++] = 3 + 12 + 0;  // Invalid
    } else {
        tokens[(*count)++] = 3 + (tm.tm_mon - 1);
    }

    // Day (offset 2)
    if (tm.tm_mday < 1 || tm.tm_mday > 31) {
        tokens[(*count)++] = 3 + 12 + 31 + 0;  // Invalid
    } else {
        tokens[(*count)++] = 3 + 12 + (tm.tm_mday - 1);
    }

    // Hour (offset 3)
    if (tm.tm_hour < 0 || tm.tm_hour >= 24) {
        tokens[(*count)++] = 3 + 12 + 31 + 24 + 0;  // Invalid
    } else {
        tokens[(*count)++] = 3 + 12 + 31 + tm.tm_hour;
    }

    // Minute (offset 4)
    if (tm.tm_min < 0 || tm.tm_min >= 60) {
        tokens[(*count)++] = 3 + 12 + 31 + 24 + 60 + 0;  // Invalid
    } else {
        tokens[(*count)++] = 3 + 12 + 31 + 24 + tm.tm_min;
    }

    // Second (offset 5)
    if (tm.tm_sec < 0 || tm.tm_sec >= 60) {
        tokens[(*count)++] = 3 + 12 + 31 + 24 + 60 + 60 + 0;  // Invalid
    } else {
        tokens[(*count)++] = 3 + 12 + 31 + 24 + 60 + tm.tm_sec;
    }
}

void timestamp_decode(const TimestampTokenizer* t, const int* tokens, int count, char* output) {
    struct tm tm = {0};
    bool valid = true;

    // We expect exactly 6 tokens (year, month, day, hour, minute, second)
    if (count != 6) {
        valid = false;
    } else {
        // Year (position 0)
        if (tokens[0] == 0) {
            tm.tm_year = t->min_year - 1;  // Below min
            valid = false;
        } else if (tokens[0] == 1) {
            tm.tm_year = t->max_year + 1;  // Above max
            valid = false;
        } else {
            tm.tm_year = t->min_year + (tokens[0] - 2);
        }

        // Month (position 1)
        if (tokens[1] == 15) {  // 3+12 = 15 (invalid marker)
            valid = false;
        } else {
            tm.tm_mon = (tokens[1] - 3);  // 0-11 range
        }

        // Day (position 2)
        if (tokens[2] == 46) {  // 15+31 = 46 (invalid marker)
            valid = false;
        } else {
            tm.tm_mday = (tokens[2] - 15) + 1;  // 1-31 range
        }

        // Hour (position 3)
        if (tokens[3] == 70) {  // 46+24 = 70 (invalid marker)
            valid = false;
        } else {
            tm.tm_hour = (tokens[3] - 46);
        }

        // Minute (position 4)
        if (tokens[4] == 130) {  // 70+60 = 130 (invalid marker)
            valid = false;
        } else {
            tm.tm_min = (tokens[4] - 70);
        }

        // Second (position 5)
        if (tokens[5] == 190) {  // 130+60 = 190 (invalid marker)
            valid = false;
        } else {
            tm.tm_sec = (tokens[5] - 130);
        }
    }

    // Format output
    if (valid) {
        // Adjust month (0-11 -> 1-12) for output
        snprintf(output, 64, "%04d-%02d-%02dT%02d:%02d:%02d",
                tm.tm_year, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
    } else {
        strcpy(output, "__invalid__");
    }
}