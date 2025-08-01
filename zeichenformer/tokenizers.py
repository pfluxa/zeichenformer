import numpy as np
import json
from pathlib import Path

from ._tokenizers import (
    BinaryTokenizer as _BinaryTokenizer,
    CategoryTokenizer as _CategoryTokenizer,
    TimestampTokenizer as _TimestampTokenizer
)


class NumericalTokenizer:
    """
    Tokenizes numerical data using recursive interval bisection.
    Now with JSON-based save/load functionality.
    """
    def __init__(self, num_bits: int = 8, offset: int = 0):
        self._offset = offset
        self._tokenizer = _BinaryTokenizer(num_bits=num_bits, offset=offset)

    def fit(self, data: np.ndarray) -> None:
        """Fits the tokenizer to the input data range."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        self._tokenizer.fit(data)

    def encode(self, values) -> list[np.ndarray]:
        """Encodes numerical values into bit position sequences."""
        return self._tokenizer.encode(values)

    def decode(self, tokens) -> np.ndarray:
        """Reconstructs original values from token sequences."""
        return self._tokenizer.decode(tokens)
    
    def save(self, filepath: Path):
        """Saves the tokenizer's state to a JSON file."""
        if not self._tokenizer.fitted:
            raise RuntimeError("Cannot save an unfitted tokenizer.")
        
        state = {
            'type': 'NumericalTokenizer',
            'num_bits': self.num_bits,
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

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def num_bits(self) -> int:
        return self._tokenizer.num_bits

    @property
    def num_tokens(self) -> int:
        return self._tokenizer.num_bits

    @property
    def vocab_size(self) -> int:
        return self.num_tokens

class CategoryTokenizer:
    """
    Tokenizes categorical string data.
    Now with JSON-based save/load functionality.
    """
    def __init__(self, categories: list[str] = None, offset: int = 0):
        self._offset = offset
        self._tokenizer = _CategoryTokenizer(offset=offset)
        if categories:
            self.fit(categories)

    def fit(self, values: list[str]) -> None:
        """Builds the category vocabulary from input data."""
        self._tokenizer.fit(values)

    def encode(self, values) -> list[np.ndarray]:
        """Converts category strings to integer tokens."""
        return self._tokenizer.encode(values)

    def decode(self, tokens) -> list[str]:
        """Converts tokens back to original category strings."""
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
    def num_categories(self) -> int:
        return self._tokenizer.num_categories

    @property
    def num_tokens(self) -> int:
        return self._tokenizer.num_bits

    @property
    def vocab_size(self) -> int:
        return self.num_tokens


class TimestampTokenizer:
    """
    Tokenizes ISO 8601 timestamps into discrete components.
    Now with JSON-based save/load functionality.
    """
    def __init__(self, min_year: int = 2000, max_year: int = 2100, offset: int = 0):
        self._offset = offset
        self._tokenizer = _TimestampTokenizer(min_year=min_year, max_year=max_year, offset=offset)

    def encode(self, values) -> list[np.ndarray]:
        """Converts ISO 8601 timestamps to component tokens."""
        return self._tokenizer.encode(values)

    def decode(self, tokens) -> list[str]:
        """Reconstructs timestamps from component tokens."""
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
        return self._tokenizer.num_bits
    
    @property
    def vocab_size(self) -> int:
        return self.num_tokens
