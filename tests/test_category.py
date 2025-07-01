import numpy as np
import time

from zeichenformer import CategoryTokenizer

def test_basic():
    offset = 9
    tokenizer = CategoryTokenizer(offset=offset)
    original_data = sorted([
        "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
        "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "pear",
        "quince", "raspberry", "strawberry", "tangerine", "ugli fruit",
        "vanilla", "watermelon", "xigua", "yellow passionfruit", "zucchini",
        "almond", "broccoli", "carrot", "dill", "eggplant", "fennel", "garlic",
        "horseradish", "iceberg lettuce", "jalapeno", "kale", "leek", "mushroom",
        "nutmeg", "onion", "parsley", "quinoa", "radish", "spinach", "tomato",
        "uppama", "vinegar", "wasabi", "xylocarp", "yam", "ziti", "ant", "bear",
        "cat", "dog", "elephant", "fox", "giraffe", "hippopotamus", "iguana",
        "jaguar", "koala", "lion", "monkey", "newt", "octopus", "penguin",
        "quokka", "rhinoceros", "snake", "tiger", "urchin", "vulture", "walrus",
        "xerus", "yak", "zebra"
    ])
    data = [x for x in original_data] 
    # Test fit
    tokenizer.fit(data)
    assert tokenizer.num_bits == len(original_data) + 2 
    assert tokenizer.num_categories == len(original_data)
    
    # Test encode/decode
    assert tokenizer.encode("almond") == 2 + offset
    assert tokenizer.decode(2 + offset) == "almond"
    assert tokenizer.encode("unknown") == 1  # Unknown
    assert tokenizer.decode(1 + offset) == "__unknown__"
    assert tokenizer.decode(0 + offset) == "__missing__"
    
    # Test batch
    encoded = tokenizer.encode(data)
    decoded = tokenizer.decode(encoded)
    assert decoded == original_data

def benchmark():
    tokenizer = CategoryTokenizer()
    original_data = [
        "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
        "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "pear",
        "quince", "raspberry", "strawberry", "tangerine", "ugli fruit",
        "vanilla", "watermelon", "xigua", "yellow passionfruit", "zucchini",
        "almond", "broccoli", "carrot", "dill", "eggplant", "fennel", "garlic",
        "horseradish", "iceberg lettuce", "jalapeno", "kale", "leek", "mushroom",
        "nutmeg", "onion", "parsley", "quinoa", "radish", "spinach", "tomato",
        "uppama", "vinegar", "wasabi", "xylocarp", "yam", "ziti", "ant", "bear",
        "cat", "dog", "elephant", "fox", "giraffe", "hippopotamus", "iguana",
        "jaguar", "koala", "lion", "monkey", "newt", "octopus", "penguin",
        "quokka", "rhinoceros", "snake", "tiger", "urchin", "vulture", "walrus",
        "xerus", "yak", "zebra"
    ]
    data = np.random.choice(original_data, 1_000_000)
    
    # Warmup
    tokenizer.fit(data)
    
    # Benchmark fit
    t0 = time.time()
    tokenizer.fit(data)
    print(f"Fit: {time.time() - t0:.4f}s")
    
    # Benchmark encode
    data = np.random.choice(original_data, 1_000_000)
    t0 = time.time()
    tokens = tokenizer.encode(data)
    print(f"Encode (1E6 samples): {time.time() - t0:.4f}s")
    
    # Benchmark decode
    t0 = time.time()
    decoded = tokenizer.decode(tokens)  # Decode subset
    print(f"Decode (1E6 samples): {time.time() - t0:.4f}s")

if __name__ == "__main__":
    test_basic()
    print("Tests passed!")
    benchmark()
    print("Benchmark passed!")