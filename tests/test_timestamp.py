from zeichenformer import TimestampTokenizer

import time
import numpy as np

def test_basic():
    tokenizer = TimestampTokenizer(min_year=2020, max_year=2030, offset=17)
    
    # Test encode/decode
    iso = ["2023-05-15T14:37:29", "2023-05-15T14:37:30"]
    tokens = tokenizer.encode(iso)
    decoded = tokenizer.decode(tokens)
    assert sum([dec == i for (dec, i) in zip(decoded, iso)]) == 2
    # Test invalid
    invalid_iso = ["2023-05-15T14:61:29",]
    tokens = tokenizer.encode(invalid_iso)
    assert tokenizer.decode(tokens)[0] == "__invalid__"
    
    # Test out of range
    not_in_range = ["1989-12-15T00:00:00",]
    tokens = tokenizer.encode(not_in_range)
    assert tokenizer.decode(tokens)[0] == "__invalid__"
    
    # # Test missing part
    incomplete = ["2023-05-15",]
    tokens = tokenizer.encode(incomplete)
    assert tokenizer.decode(tokens)[0] == "__invalid__"

def generate_timestamps():
    start_time = np.datetime64('2000-01-01T00:00:00')
    all_seconds = np.arange(10_000_000)
    chosen_seconds = np.random.choice(all_seconds, 1_000_000, replace=False)
    timestamps = start_time + np.sort(chosen_seconds)
    timestamps_iso = timestamps.astype('M8[s]').astype('O').tolist()
    timestamps_iso = np.array([dt.isoformat() for dt in timestamps_iso])

    return timestamps_iso

def benchmark():

    timestamps = generate_timestamps()
    
    tokenizer = TimestampTokenizer()
    
    # Benchmark encode
    t0 = time.time()
    tokens = tokenizer.encode(timestamps)
    print(f"Encode (1E6 samples): {time.time() - t0:.4f}s")
    
    # Benchmark decode
    t0 = time.time()
    decoded = tokenizer.decode(tokens)
    print(f"Decode (1E6 samples): {time.time() - t0:.4f}s")
    assert np.all(decoded == timestamps)

if __name__ == "__main__":
    test_basic()
    print("Tests passed!")
    benchmark()
    print("Benchmark passed!")
