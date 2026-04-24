import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ml.svm import SVM
from finance.indicators import calculate_rsi, calculate_bollinger_bands
from finance.ingestion import process_tick_data

DATA_PATH = '/Users/robertalexandrou/Projects/quant-svm-smo/data/SP.csv'

if not os.path.exists(DATA_PATH):
    print(f"Please place the S&P 500 tick data at: {DATA_PATH}")
    sys.exit(1)

# 1. Pipeline: Load and Resample Raw Tick Data
data = process_tick_data(DATA_PATH)

# 2. Feature Engineering
data['rsi'] = calculate_rsi(data['close'], window=14)
data['bb_width'] = calculate_bollinger_bands(data['close'], window=20)
data = data.dropna()

# 3. Create Targets: Regime based on returns
data['returns'] = data['close'].pct_change()
data = data.dropna()
data['target'] = np.where(data['returns'] > 0, 1, -1)

# Prepare X, y
X = data[['rsi', 'bb_width']].values
y = data['target'].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 4. Train Optimized SVM
model = SVM(kernel='rbf', C=1.0, gamma=0.5)
model.fit(X, y)

# 5. Results
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)

print(f"\nSPY Tick Data Regime Detection Accuracy: {accuracy:.2%}")
