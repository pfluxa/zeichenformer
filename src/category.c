#include "category.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static int compare_strings(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

void category_init(CategoryTokenizer* t) {
    t->categories = NULL;
    t->num_categories = 0;
    t->fitted = false;
}

void category_fit(CategoryTokenizer* t, const char** values, size_t n) {
    if (n == 0) {
        t->fitted = false;
        return;
    }

    // Temporary storage for unique categories
    char** unique = malloc(n * sizeof(char*));
    size_t unique_count = 0;

    for (size_t i = 0; i < n; i++) {
        bool found = false;
        for (size_t j = 0; j < unique_count; j++) {
            if (strcmp(values[i], unique[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            unique[unique_count] = strdup(values[i]);
            unique_count++;
        }
    }

    // Sort categories alphabetically
    qsort(unique, unique_count, sizeof(char*), compare_strings);

    // Free old categories if they exist
    if (t->categories) {
        for (size_t i = 0; i < t->num_categories; i++) {
            free(t->categories[i]);
        }
        free(t->categories);
    }

    t->categories = unique;
    t->num_categories = unique_count;
    t->fitted = true;
}

int category_encode(const CategoryTokenizer* t, const char* value) {
    if (!t->fitted) return -2;  // Not fitted
    
    // Check for NULL/empty string
    if (!value || strlen(value) == 0) return -1;  // Missing value

    // Binary search for the category
    int low = 0, high = t->num_categories - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        int cmp = strcmp(value, t->categories[mid]);
        if (cmp == 0) return mid + 2;  // Offset by 2 (sentinels at 0 and 1)
        else if (cmp < 0) high = mid - 1;
        else low = mid + 1;
    }

    return 1;  // Unknown category
}

const char* category_decode(const CategoryTokenizer* t, int token) {
    if (!t->fitted) return "__not_fitted__";
    if (token == 0) return "__missing__";
    if (token == 1) return "__unknown__";
    if (token - 2 < 0 || token - 2 >= t->num_categories) return "__invalid__";
    return t->categories[token - 2];
}

void category_free(CategoryTokenizer* t) {
    if (t->categories) {
        for (size_t i = 0; i < t->num_categories; i++) {
            free(t->categories[i]);
        }
        free(t->categories);
    }
    t->num_categories = 0;
    t->fitted = false;
}