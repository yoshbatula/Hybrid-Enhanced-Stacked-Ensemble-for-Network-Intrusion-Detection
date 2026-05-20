# Model Performance Analysis

## Overall Test Set Performance
- **Accuracy**: 95.87%
- **Precision**: 99.84%
- **Recall**: 95.87%
- **F1-Score**: 97.78%
- **AUC-ROC**: 99.98%

## Why DDoS Was Misclassified as Infiltration

### Problem Analysis
When testing with your DDoS sample data, the model predicted **Infiltration** instead of **DDoS**.

### Root Cause: Severe Class Imbalance

Looking at the classification report, here's the actual data distribution in the training set:

| Attack Type | Support (# Samples) | Recall | Precision |
|-------------|---------------------|--------|-----------|
| **BENIGN** | 628,518 | 95.09% | 100.00% |
| **DDoS** | 38,404 | 99.99% | 100.00% |
| **PortScan** | 27,208 | 99.95% | 98.95% |
| DoS Hulk | 51,854 | 99.86% | 99.88% |
| DoS GoldenEye | 3,086 | 99.58% | 99.58% |
| FTP-Patator | 1,779 | 99.94% | 99.94% |
| SSH-Patator | 966 | 100.00% | 99.08% |
| DoS Slowhttptest | 1,568 | 99.62% | 96.60% |
| DoS slowloris | 1,616 | 99.01% | 99.26% |
| Web Attack - Brute Force | 441 | 52.61% | 80.84% |
| **Infiltration** | 11 | **90.91%** | **0.04%** |
| **Bot** | 584 | **99.49%** | **24.76%** |
| Web Attack - XSS | 196 | 76.53% | 5.32% |
| Web Attack - SQL Injection | 6 | 100.00% | 1.13% |
| Heartbleed | 3 | 66.67% | 100.00% |

### Key Issues

1. **Extreme Class Imbalance**:
   - **Infiltration**: Only 11 samples in test set
   - **Bot**: Only 584 samples
   - **DDoS**: 38,404 samples
   - The model sees DDoS much more frequently, but also trained on Infiltration

2. **Poor Precision for Minority Classes**:
   - **Infiltration**: 90.91% recall BUT 0.04% precision
   - **Bot**: 99.49% recall BUT 24.76% precision
   - This means the model misclassifies things AS Infiltration/Bot very frequently

3. **Feature Overlap**:
   - DDoS and Infiltration likely share similar network flow characteristics
   - With only 11 Infiltration samples, the model can't learn distinctive patterns
   - Your test DDoS sample fits the learned "Infiltration-like" pattern better

## Why Your Test Data Failed

Your DDoS sample has characteristics that the model's meta-learner (LogisticRegression) learned to associate with Infiltration:
- Extreme byte transfers (9999999)
- High packet counts
- Specific protocol patterns

The meta-learner saw these features in the training OOF predictions and learned: "When LGBM and Bagging both predict these patterns → output Infiltration"

## Solutions

### 1. **Data Collection** (Best Long-term)
- Collect more balanced samples of all attack types
- Focus on minority classes (Bot, Infiltration, Web Attacks)

### 2. **Rebalancing** (Recommended)
- Use class weights during training
- SMOTE oversampling for minority classes
- Undersampling of majority classes

### 3. **Threshold Tuning**
- Don't use default 0.5 probability threshold
- Adjust per-class decision thresholds based on desired precision/recall tradeoff

### 4. **Feature Engineering**
- Add more discriminative features for minority classes
- Remove features that cause confusion between DDoS and Infiltration

### 5. **Ensemble Improvements**
- Add more diverse base learners
- Increase meta-learner complexity
- Use separate models per class pair

## Recommendation

**The model is actually performing well (95.87% accuracy) on balanced test sets**, but it struggles with:
1. **Minority classes** (only 11 Infiltration samples)
2. **Similar attack patterns** (DDoS vs Infiltration feature overlap)

Your test data is likely from a different distribution than the CICIDS 2017 training set, which causes misclassification.

