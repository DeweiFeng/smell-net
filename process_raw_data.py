from collections import defaultdict
import os
import pandas as pd
import random

root_dir = "/Users/derre/Documents/workspace/smell-net/neurips-data-processed"
test_dir = "/Users/derre/Documents/workspace/smell-net/testing"
train_dir = "/Users/derre/Documents/workspace/smell-net/training"

data_paths = defaultdict(list)
min_len = float('inf')  # Track minimum length across all series

ingredients = defaultdict(list)

# Load and trim data
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".csv"):
            cur_path = os.path.join(root, filename)
            df = pd.read_csv(cur_path)

            cur_ingredient = filename.split(".")[0]

            ingredients[cur_ingredient].append((filename, df))

for ingredient in ingredients:
    test_ix = random.randint(0, 5)

    os.makedirs(os.path.join(train_dir, ingredient), exist_ok=True)
    os.makedirs(os.path.join(test_dir, ingredient), exist_ok=True)

    for ix, (filename, df) in enumerate(ingredients[ingredient]):
        if ix != test_ix:
            df.to_csv(os.path.join(train_dir, ingredient, filename), index=False)
        else:
            df.to_csv(os.path.join(test_dir, ingredient, filename), index=False)
