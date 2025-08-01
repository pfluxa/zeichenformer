import numpy as np
import json
from pathlib import Path
from collections.abc import Iterable

# NOTE: This imports the C-extension classes we built.
from ._tokenizers import (
    NumericalTokenizer as _NumericalTokenizer,
    CategoryTokenizer as _CategoryTokenizer,
    TimestampTokenizer as _TimestampTokenizer
)


class NumericalTokenizer:
    """
    Tokenizes numerical data using recursive interval bisection.

    This tokenizer learns a min/max range from data and encodes any value
    as a sparse set of active "bits". Each bit corresponds to a sub-interval,
    allowing a model to learn non-linear responses to numerical features.

    The encoding process:
    1. For a given value, it finds its position within the fitted range [min, max].
    2. It recursively bisects this range `num_bits` times.
    3. At each step, it records which half the value falls into (0=lower, 1=upper).
    4. The final output is a sparse array of the integer indices of the "activated" bits.

    Special sentinel bits are reserved to handle values outside the fitted range or NaN.

    Args:
        num_bits (int, optional): The number of bisections to perform. This controls
            the precision of the encoding. Defaults to 8.
        offset (int, optional): An integer added to all output token indices, allowing
            multiple tokenizers to have non-overlapping vocabularies. Defaults to 0.

    Example:
        >>> tokenizer = NumericalTokenizer(num_bits=4)
        >>> tokenizer.fit([0.0, 1.0])
        >>> # 0.8 is in the upper half (1), then upper (1), then lower (0), then upper (1)
        >>> tokenizer.encode(0.8)
        array([1, 3, 6, 7], dtype=int32) # Sparse indices of active bits
    """
    def __init__(self, num_bits: int = 8, offset: int = 0):
        self._offset = offset
        self._tokenizer = _NumericalTokenizer(num_bits=num_bits, offset=offset)

    def fit(self, data: Iterable[float]) -> None:
        """
        Fits the tokenizer to determine the min/max range of the data.

        Args:
            data (Iterable[float]): A sequence of numbers used to find the
                min/max range for encoding. NaN values are ignored.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a 1D numpy array of floats.")
        self._tokenizer.fit(data)

    def encode(self, values: float | Iterable[float]) -> np.ndarray | list[np.ndarray]:
        """
        Encodes one or more numerical values into sparse token arrays.

        Args:
            values (float | Iterable[float]): A single float or a sequence of floats.

        Returns:
            np.ndarray | list[np.ndarray]:
            - If input is a single float, returns a 1D NumPy array of active token indices.
            - If input is a sequence, returns a list of such NumPy arrays.
        """
        if not isinstance(values, np.ndarray):
            raise TypeError("Input must be a 1D numpy array of floats.")
        return self._tokenizer.encode(values)

    def decode(self, tokens: Iterable[int] | Iterable[Iterable[int]]) -> float | np.ndarray:
        """
        Decodes one or more token sequences back into numerical values.

        Since encoding is lossy (quantization), the decoded value will be the
        center of the final sub-interval, not the exact original value.

        Args:
            tokens (Iterable[int] | Iterable[Iterable[int]]): A single token
                sequence or a sequence of token sequences.

        Returns:
            float | np.ndarray:
            - If input is a single token sequence, returns the reconstructed float.
            - If input is a sequence of sequences, returns a 1D NumPy array of floats.
        """
        # NOTE: Smarter type checking to guide the user.
        if not isinstance(tokens, (list, tuple, np.ndarray)):
            raise TypeError("Input must be a sequence of integer tokens.")
        
        # NOTE: Convert list output from C extension to NumPy array for consistency.
        decoded_values = self._tokenizer.decode(tokens)
        if isinstance(decoded_values, list):
            return np.array(decoded_values, dtype=np.float64)
        return decoded_values

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def num_tokens(self) -> int:
        """The number of bisections used for encoding (read-only)."""
        return self._tokenizer.num_tokens

    @property
    def vocab_size(self) -> int:
        """The total size of the token vocabulary, including the offset."""
        return self._tokenizer.vocab_size

    def save(self, filepath: Path):
        """Saves the tokenizer's state to a JSON file."""
        if not self._tokenizer.fitted:
            raise RuntimeError("Cannot save an unfitted tokenizer.")
        
        state = {
            'type': 'NumericalTokenizer',
            'num_tokens': self.num_tokens,
            'offset': self.offset,
            'min_val': self._tokenizer.min_val,
            'max_val': self._tokenizer.max_val
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, filepath: Path):
        """Loads a tokenizer from a JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        if state['type'] != 'NumericalTokenizer':
            raise ValueError(f"File {filepath} is not a NumericalTokenizer state file.")
            
        # Create a new instance
        instance = cls(num_bits=state['num_bits'], offset=state['offset'])
        
        # Manually set the fitted state
        instance._tokenizer.min_val = state['min_val']
        instance._tokenizer.max_val = state['max_val']
        instance._tokenizer.fitted = True
        
        return instance

class CategoryTokenizer:
    """
    Tokenizes categorical string data into unique integer tokens.

    This tokenizer learns a vocabulary of unique strings from data and maps each
    string to a unique integer. It uses special sentinel tokens for missing or
    unseen values.

    The token mapping is:
    - 0: Missing value (e.g., None, empty string)
    - 1: Unknown category (a string not seen during `fit`)
    - 2+: Learned categories, indexed alphabetically.

    Args:
        offset (int, optional): An integer added to all output token indices.
            Defaults to 0.
        categories (list[str], optional): A predefined list of categories. If
            provided, the tokenizer is fitted immediately, bypassing the need
            for a separate `.fit()` call. Defaults to None.

    Example:
        >>> tokenizer = CategoryTokenizer(categories=["apple", "banana", "cherry"])
        >>> tokenizer.encode("banana")
        array([3], dtype=int32) # 2 (base for categories) + 1 (index of "banana")
        >>> tokenizer.decode([0, 1, 3])
        ['Missing', 'Unknown', 'banana']
    """
    def __init__(self, offset: int = 0, categories: list[str] | None = None):
        self._offset = offset
        self._tokenizer = _CategoryTokenizer(offset=offset)
        # NOTE: Added user-friendly pre-fitting from the constructor.
        if categories is not None:
            self.fit(categories)

    def fit(self, values: Iterable[str]) -> None:
        """
        Builds the category vocabulary from the input data.

        Args:
            values (Iterable[str]): A sequence of strings to learn. Duplicates
                are handled, and the final vocabulary is sorted alphabetically.
        """
        if not isinstance(values, (np.ndarray)):
            raise TypeError("Input must be a numpy array of strings.")
        self._tokenizer.fit(values)

    def encode(self, values: str | Iterable[str]) -> np.ndarray:
        """
        Converts one or more category strings into integer tokens.

        Args:
            values (str | Iterable[str]): A single string or a sequence of strings.

        Returns:
            np.ndarray: A 1D NumPy array of integer tokens.
        """
        if not isinstance(values, (np.ndarray)):
            raise TypeError("Input must be a string or a sequence of strings.")
        return self._tokenizer.encode(values)

    def decode(self, tokens: int | Iterable[int]) -> str | list[str]:
        """
        Converts one or more integer tokens back to their original strings.

        Args:
            tokens (int | Iterable[int]): A single integer token or a sequence of them.

        Returns:
            str | list[str]: The decoded string or a list of strings. Special tokens
            are decoded to placeholder strings (e.g., 'Missing', 'Unknown').
        """
        if not isinstance(tokens, (np.ndarray)):
            raise TypeError("Input must be an int or a sequence of ints.")
        return self._tokenizer.decode(tokens)

    def save(self, filepath: Path):
        """Saves the tokenizer's state to a JSON file."""
        if not self._tokenizer.fitted:
            raise RuntimeError("Cannot save an unfitted tokenizer.")
            
        state = {
            'type': 'CategoryTokenizer',
            'offset': self.offset,
            'categories': self._tokenizer.get_categories()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, filepath: Path):
        """Loads a tokenizer from a JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        if state['type'] != 'CategoryTokenizer':
            raise ValueError(f"File {filepath} is not a CategoryTokenizer state file.")
            
        # Re-initialize with the saved categories
        return cls(categories=state['categories'], offset=state['offset'])

    @property
    def offset(self) -> int:
        return self._offset
    
    @property
    def vocab_size(self) -> int:
        """The total vocabulary size, including special tokens and the offset."""
        return self._tokenizer.vocab_size
    
    @property
    def num_tokens(self) -> int:
        """The number of bisections used for encoding (read-only)."""
        return self._tokenizer.num_tokens


class TimestampTokenizer:
    """
    Tokenizes ISO 8601 timestamps by decomposing them into 6 integer components.

    Each timestamp string is parsed and validated, and its components (year, month,
    day, hour, minute, second) are encoded into separate integer tokens. This allows
    a model to learn from time-based features independently.

    Args:
        min_year (int): The minimum allowed year (inclusive). Defaults to 1970.
        max_year (int): The maximum allowed year (inclusive). Defaults to 2070.
        offset (int, optional): An integer added to all output token indices.
            Defaults to 0.

    Example:
        >>> tokenizer = TimestampTokenizer(min_year=2020, max_year=2030)
        >>> # Returns a 2D array for a list of inputs
        >>> tokenizer.encode(["2025-12-31T23:59:58", "2021-01-01T00:00:00"])
        array([[5, 12, 31, 23, 59, 58],
               [1,  1,  1,  0,  0,  0]], dtype=int32)
        >>> tokenizer.decode([5, 12, 31, 23, 59, 58])
        '2025-12-31T23:59:58'
    """
    def __init__(self, min_year: int = 1970, max_year: int = 2070, offset: int = 0):
        if not min_year < max_year:
            raise ValueError("min_year must be less than max_year.")
        self._offset = offset
        self._tokenizer = _TimestampTokenizer(min_year=min_year, max_year=max_year, offset=offset)

    def encode(self, values: str | Iterable[str]) -> np.ndarray:
        """
        Converts one or more ISO 8601 timestamps into component tokens.

        Args:
            values (str | Iterable[str]): A single timestamp string or a sequence of them.
                Format: "YYYY-MM-DDTHH:MM:SS" (space separator also accepted).

        Returns:
            np.ndarray:
            - If input is a single string, returns a 1D NumPy array of shape (6,).
            - If input is a sequence, returns a 2D NumPy array of shape (n, 6).
        """
        if not isinstance(values, (np.ndarray)):
            raise TypeError("Input must be a string or a sequence of strings.")
        return self._tokenizer.encode(values)

    def decode(self, tokens: Iterable[int] | Iterable[Iterable[int]]) -> str | list[str]:
        """
        Reconstructs one or more timestamps from their 6-component token arrays.

        Args:
            tokens (Iterable): A token array of shape (6,) or a sequence of such arrays.

        Returns:
            str | list[str]: The reconstructed ISO 8601 timestamp string(s). Returns
            a placeholder for invalid or malformed token arrays.
        """
        if not isinstance(tokens, (np.ndarray)):
            raise TypeError("Input must be an array-like object of integer tokens.")
        return self._tokenizer.decode(tokens)

    def save(self, filepath: Path):
        """Saves the tokenizer's state to a JSON file."""
        state = {
            'type': 'TimestampTokenizer',
            'min_year': self._tokenizer.min_year,
            'max_year': self._tokenizer.max_year,
            'offset': self.offset
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, filepath: Path):
        """Loads a tokenizer from a JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        if state['type'] != 'TimestampTokenizer':
            raise ValueError(f"File {filepath} is not a TimestampTokenizer state file.")
            
        return cls(min_year=state['min_year'], max_year=state['max_year'], offset=state['offset'])

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def num_tokens(self) -> int:
        """The number of tokens generated per timestamp (always 6)."""
        return self._tokenizer.num_tokens

    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        return self._tokenizer.vocab_size