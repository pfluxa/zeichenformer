#include "timestamp.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void timestamp_init(TimestampTokenizer* t, int min_year, int max_year) {
    t->min_year = min_year;
    t->max_year = max_year;
    t->fitted = true;
}

bool timestamp_parse_simple(const char* iso, struct tm* tm) {
    memset(tm, 0, sizeof(struct tm));
    if(strchr(iso, 'T') != NULL)
    {
        return sscanf(iso, "%d-%d-%d %d:%d:%d",
                    &tm->tm_year, &tm->tm_mon, &tm->tm_mday,
                    &tm->tm_hour, &tm->tm_min, &tm->tm_sec) == 6;
    }
}

bool timestamp_parse(const char* iso, struct tm* tm) {
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
    if (tm->tm_mon < 1 || tm->tm_mon > 12 ||
        tm->tm_mday < 1 || tm->tm_mday > 31 ||
        tm->tm_hour < 0 || tm->tm_hour > 23 ||
        tm->tm_min < 0 || tm->tm_min > 59 ||
        tm->tm_sec < 0 || tm->tm_sec > 60) {  // 60 accounts for leap seconds
        return false;
    }
    
    // Adjust struct tm fields (year is years since 1900, month is 0-11)
    tm->tm_year -= 1900;
    tm->tm_mon -= 1;
    
    return true;
}


void timestamp_encode(const TimestampTokenizer* t, const char* iso, int* tokens, int* count) {
    *count = 0;
    struct tm tm;
    int last = 0;

    if (!timestamp_parse(iso, &tm)) {
        // Invalid format - mark all components as invalid
        *count = 6;
        tokens[0] = 0;
        tokens[1] = 0;
        tokens[2] = 0;
        tokens[3] = 0;
        tokens[4] = 0;
        tokens[5] = 0;
        return;
    }

    // Year (offset 0)
    tokens[(*count)++] = (tm.tm_year - t->min_year);

    // Month (offset 1)
    tokens[(*count)++] = tokens[0] + (tm.tm_mon - 1);

    // Day (offset 2)
    tokens[(*count)++] = tokens[1] + (tm.tm_mday - 1);

    // Hour (offset 3)
    tokens[(*count)++] = tokens[2] + tm.tm_hour;

    // Minute (offset 4)
    tokens[(*count)++] = tokens[3] + tm.tm_min;

    // Second (offset 5)
    tokens[(*count)++] = tokens[4] + tm.tm_sec;
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