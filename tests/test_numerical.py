import numpy as np
import time

from zeichenformer import NumericalTokenizer

def test_basic():
    tokenizer = NumericalTokenizer(num_bits=24, offset=204)
    data = np.random.uniform(-1.0, 1.0, 1000)
    
    # Test fit
    tokenizer.fit(data)
    assert tokenizer.num_bits == 24
    
    # Test encode/decode
    values = [-10.0, -1.0, -0.245, 0.0, 0.173, 0.25, 1.0, 13.0]
    tokens = tokenizer.encode(values)
    decoded = tokenizer.decode(tokens)
    assert np.nansum(np.array(decoded) - values) < 1e-4
    
def benchmark():
    tokenizer = NumericalTokenizer(num_bits=24)
    data = np.random.uniform(-1.0, 1.0, 1_000_000)
    
    # Warmup
    tokenizer.fit(data)
    
    # Benchmark fit
    t0 = time.time()
    tokenizer.fit(data)
    print(f"Fit: {time.time() - t0:.4f}s")
    
    # Benchmark encode
    data = np.random.uniform(0, 100, 1_000_000)
    t0 = time.time()
    tokens = tokenizer.encode(data)
    print(f"Encode (1E6 samples): {time.time() - t0:.4f}s")
    
    # Benchmark decode
    t0 = time.time()
    decoded = tokenizer.decode(tokens)
    print(f"Decode (1E6 samples): {time.time() - t0:.4f}s")

    assert np.nansum(np.array(decoded) - data) < 1e-4

if __name__ == "__main__":
    test_basic()
    print("Tests passed!")
    benchmark()
    print("Benchmark passed!")
