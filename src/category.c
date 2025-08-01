#include "category.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Helper function for qsort
static int compare_strings(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

void category_init(CategoryTokenizer* t, int offset) {
    t->categories = NULL;
    t->num_categories = 0;
    t->num_tokens = 2; // Reserve codes 0 (missing) and 1 (unknown)
    t->fitted = false;
    t->offset = offset;
}

void category_free(CategoryTokenizer* t) {
    if(t->fitted)
    {
        free(t->categories);
    }
}

void category_fit(CategoryTokenizer* t, const char** values, size_t n) {
    if (n == 0) {
        t->fitted = false;
        return;
    }

    // --- High-performance unique string finder (O(N log N)) ---

    // 1. Create a mutable copy of the input array to sort it.
    char** sorted_values = malloc(n * sizeof(char*));
    if (!sorted_values) return; // Allocation failed
    memcpy(sorted_values, values, n * sizeof(char*));

    // 2. Sort the entire array. All duplicates are now adjacent.
    qsort(sorted_values, n, sizeof(char*), compare_strings);

    // 3. Iterate once to find unique strings and copy them.
    // We can over-allocate then reallocate, but counting first is clean.
    size_t unique_count = 0;
    if (n > 0) {
        unique_count = 1;
        for (size_t i = 1; i < n; i++) {
            if (strcmp(sorted_values[i], sorted_values[i - 1]) != 0) {
                unique_count++;
            }
        }
    }
    
    char** unique_cats = malloc(unique_count * sizeof(char*));
    if (!unique_cats) { // Allocation failed
        free(sorted_values);
        return;
    }
    
    // Populate the final unique categories array
    if (n > 0) {
        unique_cats[0] = strdup(sorted_values[0]);
        size_t current_unique_idx = 1;
        for (size_t i = 1; i < n; i++) {
            if (strcmp(sorted_values[i], sorted_values[i - 1]) != 0) {
                unique_cats[current_unique_idx] = strdup(sorted_values[i]);
                current_unique_idx++;
            }
        }
    }
    
    free(sorted_values); // Free the temporary sorted copy

    // --- Update tokenizer state ---

    // Free old categories if refitting
    if (t->categories) {
        for (size_t i = 0; i < t->num_categories; i++) {
            free(t->categories[i]);
        }
        free(t->categories);
    }

    t->categories = unique_cats;
    t->num_categories = unique_count; // Set the correct category count
    t->num_tokens = unique_count + 2; // Update total token count
    t->fitted = true;
}

int category_encode(const CategoryTokenizer* t, const char* value) {
    if (!t->fitted) {
        return -1; // Not fitted
    }
    
    // Handle missing value (NULL or empty string)
    if (!value || value[0] == '\0') {
        return 0 + t->offset; // Missing value token
    }

    // Binary search for the category in the sorted list
    int low = 0;
    int high = t->num_categories - 1; // Use the correct num_categories field
    while (low <= high) {
        int mid = low + (high - low) / 2;
        int cmp = strcmp(value, t->categories[mid]);

        if (cmp == 0) {
            // Found: offset by 2 (for missing/unknown)
            return (mid + 2) + t->offset;
        } else if (cmp < 0) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    // Not found in the list of known categories
    return 1 + t->offset; // Unknown category token
}

const char* category_decode(const CategoryTokenizer* t, int token) {
    if (!t || !t->fitted) return "Error: Unfitted";

    int code = token - t->offset;

    if (code == 0) return "Missing"; // Or whatever you want for missing
    if (code == 1) return "Unknown"; // Or whatever you want for unknown
    
    // Adjust for the 2 special tokens
    int category_index = code - 2;

    if (category_index >= 0 && (size_t)category_index < t->num_categories) {
        return t->categories[category_index];
    }

    return "Error: Invalid Token";
}

const char** category_get_categories(const CategoryTokenizer* t, size_t* count) {
    if (!t->fitted) {
        *count = 0;
        return NULL;
    }
    *count = t->num_categories;
    return (const char**)t->categories;
}