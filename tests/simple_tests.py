import numpy as np
import time

# Assuming the compiled module is in a package named 'zeichenformer'
from zeichenformer import (
    NumericalTokenizer,
    CategoryTokenizer,
    TimestampTokenizer
)

def run_numerical_tests():
    """Minimal debug script for NumericalTokenizer."""
    print("--- Running NumericalTokenizer Debug Script ---")
    try:
        tokenizer = NumericalTokenizer(num_bits=16, offset=204)
        data = np.random.uniform(-1.0, 1.0, 1000)
        
        print("Fitting numerical tokenizer...")
        tokenizer.fit(data)
        assert tokenizer.num_tokens == 16 + 2 # User bits + 2 sentinel bits

        print("Testing encode/decode roundtrip...")
        values = np.array([-10.0, -1.0, -0.245, 0.0, 0.173, 1.0, 13.0, np.nan])
        
        tokens = tokenizer.encode(values)
        print("Encode successful. Now attempting to decode...")

        # The crash happens here, according to the pytest traceback.
        # This call will likely trigger the segmentation fault.
        decoded = tokenizer.decode(tokens)
        print("Decode successful.")

        print("NumericalTokenizer tests PASSED.")

    except Exception as e:
        print(f"An unexpected Python-level error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40 + "\n")

def run_categorical_tests():
    """Minimal debug script for CategoryTokenizer."""
    print("--- Running CategoryTokenizer Debug Script ---")
    try:
        # 1. Initialization
        offset = 50
        tokenizer = CategoryTokenizer(offset=offset)
        print(f"Tokenizer initialized with offset: {offset}")

        # 2. Fitting
        fit_data = ["apple", "banana", "cherry", "apple"]
        # The C-API expects a NumPy array of objects for strings
        fit_array = np.array(fit_data, dtype=object)
        
        print("Fitting categorical tokenizer...")
        tokenizer.fit(fit_array)
        
        # After fitting, the vocabulary should be unique and sorted: apple, banana, cherry
        # Total tokens = 3 categories + 2 sentinels (missing/unknown)
        assert tokenizer.vocab_size == (3 + 2)
        print("Fit successful.")

        # 3. Encode/Decode Roundtrip
        print("Testing encode/decode roundtrip...")
        test_values = np.array([
            "banana",        # Known
            "apple",         # Known
            "date",          # Unknown
            ""               # Missing
        ], dtype=object)

        tokens = tokenizer.encode(test_values)
        print("Encode successful. Tokens:", tokens)
        
        print("Now attempting to decode...")
        decoded_values = tokenizer.decode(tokens)
        print("Decode successful. Decoded values:", decoded_values)

        # 4. Assertions
        expected_decoded_values = np.array(["banana", "apple", "Unknown", "Missing"])
        assert np.array_equal(decoded_values, expected_decoded_values), "Decoded values do not match expected"
        
        print("CategoricalTokenizer tests PASSED.")

    except Exception as e:
        print(f"An unexpected Python-level error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40 + "\n")

def run_timestamp_tests():
    """Minimal debug script for TimestampTokenizer."""
    print("--- Running TimestampTokenizer Debug Script ---")
    try:
        # 1. Initialization
        min_year, max_year, offset = 2020, 2030, 100
        tokenizer = TimestampTokenizer(min_year=min_year, max_year=max_year, offset=offset)
        print(f"Tokenizer initialized for range {min_year}-{max_year} with offset {offset}")

        # 2. Encode/Decode Roundtrip for valid data
        print("Testing encode/decode roundtrip...")
        valid_timestamps = np.array([
            "2023-05-15T14:37:29",
            "2020-01-01T00:00:00",
            "2030-12-31T23:59:59"
        ], dtype=object)

        tokens = tokenizer.encode(valid_timestamps)
        print("Encode successful. Now attempting to decode...")
        
        decoded_values = tokenizer.decode(tokens)
        print("Decode successful. Decoded values:", decoded_values)
        
        assert np.array_equal(decoded_values, valid_timestamps), "Decoded values do not match expected"
        print("Roundtrip test PASSED.")

        # 3. Test error handling for invalid inputs
        print("Testing error handling for invalid inputs...")
        invalid_timestamps = [
            ("out of range", np.array(["2019-01-01T00:00:00"], dtype=object)),
            ("malformed string", np.array(["not-a-timestamp"], dtype=object)),
            ("invalid component", np.array(["2023-02-30T10:00:00"], dtype=object)),
        ]

        for name, data in invalid_timestamps:
            try:
                tokenizer.encode(data)
                # This line should not be reached
                print(f"FAIL: Invalid timestamp ({name}) did not raise a ValueError.")
            except ValueError as e:
                print(f"PASS: Correctly caught expected error for '{name}': {e}")

        print("TimestampTokenizer tests PASSED.")

    except Exception as e:
        print(f"An unexpected Python-level error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40 + "\n")


run_numerical_tests()

run_categorical_tests()

run_timestamp_tests()