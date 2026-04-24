import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from finance.ingestion import process_tick_data

# Process only a small subset of the file
DATA_PATH = '/Users/robertalexandrou/Projects/quant-svm-smo/data/SP.csv'
# Use pandas to just read the first 200k lines
df = pd.read_csv(DATA_PATH, nrows=200000)
df.to_csv('/Users/robertalexandrou/Projects/quant-svm-smo/data/SP_subset.csv', index=False)

# Now run the pipeline on the subset
data = process_tick_data('/Users/robertalexandrou/Projects/quant-svm-smo/data/SP_subset.csv')
print(f"Subset processing complete. {len(data)} bars created.")
