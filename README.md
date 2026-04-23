# Financial ML: SVM from Scratch

## Description
A pure-NumPy implementation of Support Vector Machines (SVM) using the Sequential Minimal Optimization (SMO) algorithm, applied to market regime classification.

## Mathematical Foundations
The dual formulation of the SVM optimization problem is given by:
maximize \sum \alpha_i - 0.5 \sum \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)
subject to 0 <= \alpha_i <= C and \sum \alpha_i y_i = 0.

## Financial Application
Predicting market regimes (e.g., Bull vs. Bear) using technical indicators as features.
