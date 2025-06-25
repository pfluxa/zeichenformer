#include "binary.h"
#include <math.h>
#include <float.h>

void binary_init(BinaryTokenizer* t, int num_bits) {
    t->num_bits = num_bits;
    t->min_val = NAN;
    t->max_val = NAN;
    t->fitted = false;
}

void binary_fit(BinaryTokenizer* t, const double* values, size_t n) {
    if (n == 0) {
        t->fitted = false;
        return;
    }

    double min = DBL_MAX;
    double max = -DBL_MAX;

    for (size_t i = 0; i < n; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    t->min_val = min;
    t->max_val = max;
    t->fitted = true;
}

void binary_encode(const BinaryTokenizer* t, double value, int* indices, int* count) {
    *count = 0;
    if (!t->fitted || isnan(value)) return;
    if (!((value >= t->min_val)&&(value <= t->max_val))) return;
    
    double center = (t->min_val + t->max_val) / 2.0;
    double width = (t->max_val - t->min_val) / 2.0;

    // sentinel value for below minimum
    for (int b = 0; b < t->num_bits; b++) {
        if (value > center) {
            indices[((*count)++)] = b;
            center += width / 2.0;
        } else {
            center -= width / 2.0;
        }
        width /= 2.0;
    }
}

double binary_decode(const BinaryTokenizer* t, const int* indices, int count) {
    if (!t->fitted || count == 0) return NAN;

    double center = (t->min_val + t->max_val) / 2.0;
    double width = (t->max_val - t->min_val) / 2.0;
    double value = center;

    for (int b = 0; b < t->num_bits; b++) {
        int active = 0;
        for (int i = 0; i < count; i++) {
            if (indices[i] == b) active = 1;
        }
        value += active ? (width / 2.0) : (-width / 2.0);
        width /= 2.0;
    }

    return value;
}