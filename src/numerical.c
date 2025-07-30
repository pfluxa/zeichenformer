#include "numerical.h"
#include <math.h>
#include <string.h> // Required for memset
#include <float.h>

void numerical_init(NumericalTokenizer* t, int num_bits, int offset) {
    t->num_bits = num_bits + 2;
    t->min_val = NAN;
    t->max_val = NAN;
    t->fitted = false;
    t->offset = offset;
}

void numerical_fit(NumericalTokenizer* t, const double* values, size_t n) {
    if (n == 0) {
        t->fitted = false;
        return;
    }

    double min = DBL_MAX;
    double max = -DBL_MAX;

    for (size_t i = 0; i < n; i++) {
        // This correctly ignores NaNs
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    // Add this check: if min was never updated, no valid numbers were found.
    if (min > max) {
        t->fitted = false;
        return;
    }
    
    t->min_val = min;
    t->max_val = max;
    t->fitted = true;
}

void numerical_encode(const NumericalTokenizer* t, double value, int* indices, int* count) {
   
    int bits[t->num_bits];
    double center;
    double width;
    bool invalid = false;

    // not fitted? return
    if (!t->fitted) {
        *count = -1;
        return;
    }
    
    bits[0] = 0;
    bits[1] = 0;
    *count  = 0;
    // values below minimum are encoded as 10 + <0 x num_bits>
    if (value <= t->min_val)
    {
        bits[0] = 1;
        bits[1] = 0;
        for(int b=2; b < t->num_bits; b++) bits[b] = 0;
        invalid = true;
    } 
    // values above maximum are encoded as 01 + <0 x num_bits>
    if (value > t->max_val)
    {
        bits[0] = 0;
        bits[1] = 1;
        for(int b=2; b < t->num_bits; b++) bits[b] = 0;
        invalid = true;
    }
    // nan values are encoded as 11 + <0 x num_bits>
    if(isnan(value)) {
        bits[0] = 1;
        bits[1] = 1;
        for(int b=2; b < t->num_bits; b++) bits[b] = 0;
        invalid = true;
    }

    // the actual encoding magic
    if(!invalid)
    {
        center = (t->min_val + t->max_val) / 2.0;
        width = (t->max_val - t->min_val) / 2.0;
        for(int b = 2; b < t->num_bits; b++)
        {
            if (value > center) 
            {
                bits[b] = 1;
                center += width / 2.0;
            } 
            else 
            {
                bits[b] = 0;
                center -= width / 2.0;
            }
            width /= 2.0;
        }
    }
    
    // transform bits to tokens
    for(int b = 0; b < t->num_bits; b++)
    {
        if(bits[b] > 0)
        {
            indices[((*count)++)] = (b + 2) + t->offset;
        }
    }
}

double numerical_decode(const NumericalTokenizer* t, const int* indices, int count) {
   
    // not fitted or invalid count? return NaN
    if (!t->fitted || count < 0) {
        return NAN;
    }

    int bits[t->num_bits];
    double center;
    double width;

    // 1. Reconstruct the dense bit array from the sparse indices
    memset(bits, 0, sizeof(int)); // Initialize all bits to 0
    for (int i = 0; i < count; i++) {
        int b = indices[i] - 2 - t->offset;
        // Safety check: ensure the calculated bit index is within bounds
        if (b >= 0 && b < t->num_bits) {
            bits[b] = 1;
        } else {
            // An invalid index was provided
            return NAN; 
        }
    }

    // 2. Decode the value from the bit array
    
    // Handle special case prefixes
    if (bits[0] == 1 && bits[1] == 1) {
        // [1, 1] prefix for NaN
        return NAN;
    } else if (bits[0] == 1 && bits[1] == 0) {
        // [1, 0] prefix for value <= min_val
        return t->min_val;
    } else if (bits[0] == 0 && bits[1] == 1) {
        // [0, 1] prefix for value > max_val
        return t->max_val;
    } else {
        // [0, 0] prefix for an in-range value
        // Reconstruct the value by walking down the quantization tree
        center = (t->min_val + t->max_val) / 2.0;
        width = (t->max_val - t->min_val) / 2.0;
        
        for (int b = 2; b < t->num_bits; b++) {
            if (bits[b] == 1) {
                // Bit 1 means it was in the upper half of the interval
                center += width / 2.0;
            } else {
                // Bit 0 means it was in the lower half
                center -= width / 2.0;
            }
            width /= 2.0;
        }
        return center;
    }
}

// double numerical_decode(const NumericalTokenizer* t, const int* indices, int count) 
// {
//     double center;
//     double width;
//     double value;    
    
//     if (!t->fitted || count == -1) return NAN;

//     center = (t->min_val + t->max_val) / 2.0;
//     width = (t->max_val - t->min_val) / 2.0;
//     value = center;

//     for (int b = 0; b < t->num_bits; b++) {
//         int active = 0;
//         for (int i = 0; i < count; i++) {
//             if (indices[i] - (1 + t->offset) == b) active = 1;
//         }
//         value += active ? (width / 2.0) : (-width / 2.0);
//         width /= 2.0;
//     }

//     return value;
// }