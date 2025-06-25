 # Zeichenformer: Blazing-Fast Data Tokenization for Machine Learning

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

**Industrial-strength tokenization for numerical, categorical, and timestamp data** - optimized for performance and designed for modern ML workflows.

## Why Tokenization Matters

Tokenization is the critical first step in preparing data for machine learning models, particularly for Transformer architectures. While pre-trained tokenizers work well for text, most tabular and time-series data requires custom tokenization approaches because:

1. **Domain-Specific Patterns**: Your data has unique distributions and relationships that generic tokenizers can't capture
2. **Information Preservation**: Custom tokenization maintains the original data semantics without embedding distortions
3. **Cold-Start Advantage**: Enables effective modeling from scratch without requiring pre-trained embeddings
4. **Computational Efficiency**: Specialized tokenizers can be orders of magnitude faster than general-purpose solutions

## Key Features

### 🚀 Performance Optimized
- Process 1 million numerical values in <500ms
- Categorical encoding with O(log n) lookup complexity
- Zero memory allocation during inference (after initialization)

### 🔢 Three Specialized Tokenizers

1. **Numerical Tokenizer**
   - Recursive interval bisection for optimal binning
   - Automatic range detection during fitting
   - Configurable precision (4-16 bits typical)

2. **Categorical Tokenizer**
   - Alphabetical sorting for binary search efficiency
   - Built-in handling of missing/unknown categories
   - Constant-time decoding

3. **Timestamp Tokenizer**
   - ISO 8601 compliant parsing
   - Component-wise validation (years, months, etc.)
   - Configurable valid year ranges

### 🔌 Seamless Integration
- NumPy-native input/output
- Batch processing support
- Pure Python interface with C acceleration

## Tradeoffs to Consider

✔ **Pros**:
- Sub-second processing for million-row datasets
- Minimal dependencies (just NumPy + C compiler)
- Exact reconstruction capability
- MIT licensed for unrestricted use

✖ **Cons**:
- Timestamps become high-dimensional (6+ tokens per value)
- No built-in sub-word tokenization
- Requires re-fitting to handle new data distributions
- Single-core implementation

## Installation

1. **Prerequisites**:
   - Python 3.7+
   - GCC (or other C compiler)
   - NumPy (will be installed automatically)

2. **Install from source**:
   ```bash
   git clone https://github.com/pfluxa/zeichenformer.git
   cd zeichenformer
   pip install -e .
   ```

## Quick Start

```python
from zeichenformer import BinaryTokenizer, CategoryTokenizer, TimestampTokenizer

# Numerical data
num_tokenizer = BinaryTokenizer(num_bits=8)
num_tokenizer.fit([0.0, 100.0])
tokens = num_tokenizer.encode([25.3, 50.1, 75.8])  # array of token arrays

# Categorical data
cat_tokenizer = CategoryTokenizer()
cat_tokenizer.fit(["red", "green", "blue"])
tokens = cat_tokenizer.encode("green")  # array([3], dtype=int32)

# Timestamps
time_tokenizer = TimestampTokenizer(min_year=2000, max_year=2030)
tokens = time_tokenizer.encode("2025-12-31T23:59:58")  # 6-component array
```

## Advanced Usage

For maximum performance with large datasets:
```python
# Process DataFrame columns efficiently
import pandas as pd
df = pd.read_parquet("large_dataset.parquet")

# Batch encode entire columns
num_tokens = num_tokenizer.encode(df["values"].to_numpy())
cat_tokens = cat_tokenizer.encode(df["categories"].to_list())
```

## Benchmarks

| Operation              | 1M Samples | Notes                          |
|------------------------|------------|--------------------------------|
| **Numerical Data**     |            |                                |
| - Fit                  | 5.9ms      | One-time calibration           |
| - Encode               | 221.1ms    | ~4.5M ops/second              |
| - Decode               | 64.7ms     | ~15.5M ops/second             |
| **Categorical Data**   |            |                                |
| - Fit                  | 718.9ms    | Includes vocabulary building   |
| - Encode               | 500.4ms    | ~2M ops/second                |
| - Decode               | 128.5ms    | ~7.8M ops/second              |
| **Timestamp Data**     |            |                                |
| - Encode               | 644.3ms    | ~1.55M ops/second             |
| - Decode               | 584.1ms    | ~1.7M ops/second              |

*Results collected on Apple Silicon (M3 Max) hardware - actual performance may vary*

## License

MIT License - Copyright (c) [Your Name] and ThoughtWorks. Free for commercial and personal use with attribution.
