import numpy as np
from ._tokenizers import (
    BinaryTokenizer as _BinaryTokenizer,
    CategoryTokenizer as _CategoryTokenizer,
    TimestampTokenizer as _TimestampTokenizer
)


class NumericalTokenizer:
    """
    Tokenizes numerical data using recursive interval bisection (binary space partitioning).
    Each value is encoded as a sequence of bits representing its position within progressively
    smaller sub-intervals of the fitted range.

    The encoding process:
    1. Divides the fitted range [min_val, max_val] into two equal sub-intervals
    2. Records which sub-interval contains the value (0=lower, 1=upper)
    3. Recursively bisects the chosen sub-interval for `num_bits` iterations
    4. Returns the sequence of bit positions that were "activated"

    The decoding process reverses this by reconstructing the value from the bit positions.

    Args:
        num_bits (int, optional): Number of bisection iterations (bits) to use.
                                Higher values give more precision but longer token sequences.
                                Default: 8.

    Example:
        >>> tokenizer = BinaryTokenizer(num_bits=4)
        >>> tokenizer.fit([0.0, 1.0])
        >>> tokenizer.encode(0.75)
        array([0, 1, 1, 1], dtype=int32)  # Indicates upper half at each bisection
    """
    def __init__(self, num_bits: int = 8, offset: int = 0):
        self._offset = offset
        self._tokenizer = _BinaryTokenizer(num_bits=num_bits)

    def fit(self, data: np.ndarray) -> None:
        """
        Fits the tokenizer to the input data range.

        Parameters:
            data : np.ndarray[float]
                1D array of values used to determine the [min_val, max_val] range.
                All future encodes will be relative to this range.

        Implementation Notes:
        - Uses exact min/max from data (no epsilon padding)
        - NaN values are ignored during fitting
        - Empty input leaves tokenizer in unfitted state (encode/decode will return NaN)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        self._tokenizer.fit(data)

    def encode(self, values) -> np.ndarray:
        """
        Encodes numerical values into bit position sequences.

        Parameters:
            values : float | Iterable[float]
                Input value(s) to encode. Can be:
                - Single float -> returns 1D array
                - Sequence of floats -> returns list of 1D arrays

        Returns:
            np.ndarray[int32] | list[np.ndarray[int32]]
                For single input: 1D array of active bit positions (0 to num_bits-1)
                For multiple inputs: List of such arrays

        Implementation Details:
        - Values outside fitted range return empty arrays
        - NaN inputs return empty arrays
        - Each bisection level adds exactly 0 or 1 to the output sequence
        """
        
        token_list = self._tokenizer.encode(values)
        for tokens in token_list:
            tokens = tokens + self._offset
        return token_list

    def decode(self, tokens) -> np.ndarray:
        """
        Reconstructs original values from token sequences.

        Parameters:
            tokens : Sequence[int] | Iterable[Sequence[int]]
                Bit position sequences to decode. Can be:
                - Single sequence -> returns float
                - Multiple sequences -> returns array of floats

        Returns:
            float | np.ndarray[float]
                Reconstructed value(s). Returns NaN for:
                - Empty input sequences
                - Unfitted tokenizer
                - Invalid bit positions

        Algorithm:
        1. Starts with center of fitted range
        2. For each bit position in 0..num_bits-1:
        - If present in tokens: move toward upper sub-interval
        - Else: move toward lower sub-interval
        3. Final position is the decoded value
        """
        return self._tokenizer.decode(tokens - self._offset)
    
    @property
    def offset(self) -> int:
        return self._offset

    @property
    def num_bits(self) -> int:
        """
        Number of bisection levels used (read-only).
        
        Determines:
        - Maximum token sequence length
        - Theoretical error bound: (max_val-min_val)/2^num_bits
        """
        return self._tokenizer.num_bits

    @property
    def max_active_features(self) -> int:
        """
        Maximum possible active tokens per encode (equal to num_bits).
        """
        return self._tokenizer.max_active_features


class CategoryTokenizer:
    """
    Tokenizes categorical string data using a sorted vocabulary with sentinel tokens.
    Implements efficient binary search for encoding and provides special tokens for:
    - Missing values (empty/NULL strings)
    - Unknown categories (values not seen during fitting)
    - Invalid tokens (out-of-range values)

    The token mapping is:
    0: "__missing__" (reserved for empty/NULL inputs)
    1: "__unknown__" (reserved for unseen categories)
    2+: Actual categories (sorted alphabetically)

    Args:
        categories (list[str], optional): Predefined categories. If provided,
                                        bypasses the need to call fit().
                                        Defaults to None.

    Example:
        >>> tokenizer = CategoryTokenizer()
        >>> tokenizer.fit(["apple", "banana", "cherry"])
        >>> tokenizer.encode("banana")
        3  # 2 (offset) + 1 (alphabetical position)
        >>> tokenizer.decode([0, 1, 3])
        ["__missing__", "__unknown__", "banana"]
    """
    def __init__(self, offset: int = 0):
        self._offset = offset
        self._tokenizer = _CategoryTokenizer()

    def fit(self, values: list[str]) -> None:
        """
        Builds the category vocabulary from input data.

        Parameters:
            values : list[str]
                Raw category strings to learn. Duplicates are automatically removed.

        Implementation Notes:
        - Sorts categories alphabetically for O(log n) encoding
        - Empty strings/NULL values map to sentinel token 0
        - Subsequent unseen values map to token 1
        - Original strings are copied internally (safe to modify input after fitting)

        C-Level Behavior:
        1. Deduplicates input while preserving original case
        2. Uses qsort() with strcmp() for alphabetical ordering
        3. Allocates independent memory for category strings
        """
        self._tokenizer.fit(values)

    def encode(self, values) -> np.ndarray:
        """
        Converts category strings to integer tokens.

        Parameters:
            values : str | Iterable[str]
                Input(s) to encode. Can be:
                - Single string -> returns scalar array
                - Sequence of strings -> returns 1D array

        Returns:
            np.ndarray[int32]
                Token values where:
                - 0 = Missing/empty input
                - 1 = Unknown category
                - ≥2 = Valid category (offset by 2)

        Special Cases:
        - None/empty string → 0 ("__missing__")
        - Unseen category → 1 ("__unknown__")
        - Non-string input → TypeError
        """
        token_list = self._tokenizer.encode(values)
        for tokens in token_list:
            tokens = tokens + self._offset
        return token_list

    def decode(self, tokens) -> list[str]:
        """
        Converts tokens back to original category strings.

        Parameters:
            tokens : int | Iterable[int]
                Token(s) to decode. Can be:
                - Single int -> returns single string
                - Sequence -> returns list of strings

        Returns:
            list[str]
                Decoded strings with special cases:
                - 0 → "__missing__"
                - 1 → "__unknown__"
                - Invalid tokens → "__invalid__"

        Error Handling:
        - Returns placeholder strings for invalid tokens rather than raising
        - Non-integer inputs → TypeError
        """
        return self._tokenizer.decode(tokens - self._offset)

    @property
    def offset(self) -> int:
        return self._offset
    
    @property
    def num_categories(self) -> int:
        """
        The number of unique categories learned (excluding sentinels).
        Read-only after fitting.
        """
        return self._tokenizer.num_categories

    @property
    def num_bits(self) -> int:
        """
        Total token space size (num_categories + 2 sentinels).
        Useful for determining output dimension in ML models.
        """
        return self._tokenizer.num_bits

    @property
    def max_active_features(self) -> int:
        """
        Always returns 3 because:
        - 1 active category token (≥2)
        - 2 sentinel bits (for missing/unknown)
        """
        return 3


class TimestampTokenizer:
    """
    Tokenizes ISO 8601 timestamps into discrete components with validation.
    Each timestamp is decomposed into 6 integer tokens representing:
    1. Year   (with min/max bounds checking)
    2. Month  (1-12)
    3. Day    (1-31)
    4. Hour   (0-23)
    5. Minute (0-59)
    6. Second (0-59)

    Invalid components are automatically flagged with special tokens.
    The tokenizer is always "fitted" (no separate fit() needed).

    Args:
        min_year (int): Minimum allowed year (inclusive). Default: 2000
        max_year (int): Maximum allowed year (inclusive). Default: 2100

    Example:
        >>> tokenizer = TimestampTokenizer(min_year=2020, max_year=2030)
        >>> tokens = tokenizer.encode("2025-12-31T23:59:58")
        >>> tokens
        array([7, 14, 45, 69, 129, 189], dtype=int32)  # See breakdown below
        >>> tokenizer.decode(tokens)
        "2025-12-31T23:59:58"
    
    Notes:
    
    [0]: Year token (2 + (year - min_year))
    [1]: Month token (3 + (month - 1))
    [2]: Day token (15 + (day - 1))       # 15 = 3 (offset) + 12 (months)
    [3]: Hour token (46 + hour)           # 46 = previous offsets + 31 (days)
    [4]: Minute token (70 + minute)       # 70 = previous + 24 (hours)
    [5]: Second token (130 + second)      # 130 = previous + 60 (minutes)

    Special Values:
    - Year: 0=below min, 1=above max
    - Other components: Highest token = invalid (e.g., month=15)
    """
    def __init__(self, min_year: int = 2000, max_year: int = 2100, offset: int = 0):
        self._offset = offset
        self._tokenizer = _TimestampTokenizer(min_year=min_year, max_year=max_year)

    def encode(self, timestamps) -> np.ndarray:
        """
        Converts ISO 8601 timestamps to component tokens.

        Parameters:
            timestamps : str | Iterable[str]
                Input timestamp(s) in "YYYY-MM-DDTHH:MM:SS" format.
                Can be:
                - Single string -> returns (6,) array
                - Sequence -> returns list of (6,) arrays

        Returns:
            np.ndarray[int32] | list[np.ndarray[int32]]
                Array(s) of 6 tokens per timestamp, with:
                - Valid components: Mapped to token ranges shown above
                - Invalid components: Flagged with boundary values
                - Malformed input: All components marked invalid

        Example:
            >>> tokenizer.encode("2025-02-30T25:61:61")  # Invalid date/time
            array([7, 5, 46, 70, 130, 190], dtype=int32)  # Day/hour/minute/second invalid
        """
        token_list = self._tokenizer.encode(values)
        for tokens in token_list:
            tokens = tokens + self._offset
        return token_list

    def decode(self, tokens) -> list[str]:
        """
        Reconstructs timestamps from component tokens.

        Parameters:
            tokens : array-like | Iterable[array-like]
                Token sequence(s) to decode. Each must contain exactly 6 tokens.

        Returns:
            list[str]
                Reconstructed timestamps in ISO format. Invalid components return:
                - "__invalid__" for malformed token sequences.
                - Clamped values for out-of-bounds years
                - Best-effort reconstruction for other invalid components

        Example:
            >>> tokens = tokenizer.encode("2025-02-30T25:61:61") # Invalid date/time
            >>> tokenizer.decode(tokens)  # [[7, 5, 46, 70, 130, 190],]
            ["__invalid__"]
        """
        return self._tokenizer.decode(tokens - self._offset)
    
    @property
    def offset(self) -> int:
        return self._offset

    @property
    def num_bits(self) -> int:
        """
        Total token space size calculated as:
            3 (year sentinels) + 
            (max_year - min_year + 1) + 
            12 (months) + 
            31 (days) + 
            24 (hours) + 
            60 (minutes) + 
            60 (seconds)

            Example (2000-2100 default range):
            3 + 101 + 12 + 31 + 24 + 60 + 60 = 291
        """
        return self._tokenizer.num_bits

    @property
    def max_active_features(self) -> int:
        return self._tokenizer.max_active_features