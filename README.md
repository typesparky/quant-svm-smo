# Quant SVM SMO: Market Regime Detection

This project provides a production-grade implementation of Support Vector Machines (SVM) trained using Sequential Minimal Optimization (SMO), specifically designed for high-frequency financial data analysis.

## Core Features
*   **Optimized SMO Algorithm:** Implements the Keerthi et al. improvements to Platt's SMO for faster convergence.
*   **Memory-Efficient:** Uses on-demand kernel row-caching to train on large datasets without needing to hold an $O(N^2)$ Gram matrix in memory.
*   **High-Performance ETL:** Includes a chunked data ingestion pipeline to transform massive tick-level data into actionable 1-minute OHLCV bars.
*   **Financial Indicators:** Built-in module for calculating technical analysis features (RSI, Bollinger Bands).

## Mathematical Foundations
The model solves the dual SVM optimization problem:
Maximize $\sum \alpha_i - 0.5 \sum \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$
Subject to: $0 \le \alpha_i \le C$ and $\sum \alpha_i y_i = 0$

## Pipeline Walkthrough
1. **Data Ingestion:** `src/finance/ingestion.py` processes large CSVs in chunks, resampling tick data to 1-minute bars.
2. **Feature Engineering:** `src/finance/indicators.py` generates signals (RSI, BB Width).
3. **Training:** `src/ml/svm.py` uses the optimized SMO solver with a kernel cache to fit the model.
4. **Demonstration:** `examples/regime_detection.py` connects the pipeline and evaluates market regime detection accuracy.

## Setup & Usage
1. Clone the repository.
2. Place raw tick data (e.g., Kaggle S&P 500 futures) at `data/SP.csv`.
3. Run the ingestion and regime detection demo:
   ```bash
   python3 examples/regime_detection.py
   ```

## Experiment Results
Based on historical SPY sample data, the model achieved:
- **Accuracy:** 60.00%
- **Recall:** 1.00 (Highly sensitive to upward market movements)

See `examples/RESULTS.md` for the full confusion matrix and classification report.
