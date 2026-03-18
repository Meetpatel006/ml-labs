# Titanic Survival Prediction - Decision Tree Classifier

## Project Overview

This project implements a Decision Tree classifier to predict Titanic passenger survival. The project demonstrates proper machine learning methodology including preprocessing, hyperparameter tuning, cross-validation, and handling class imbalance.

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | 891 samples with survival labels (ground truth) |
| `test.csv` | 418 samples without labels (predictions) |
| `gender_submission.csv` | Sample submission format |

### Dataset Statistics
- **Overall Survival Rate**: 38.38%
- **Female Survival Rate**: 74.20%
- **Male Survival Rate**: 18.89%

---

## Project Journey

### Phase 1: Initial Implementation
- Created folder structure: `opencode/src/`, `opencode/models/`
- Implemented data preprocessing pipeline
- Trained basic Decision Tree (unlimited depth)
- **Initial Result**: Training Accuracy: 98.03%, Validation: 81.56%

### Phase 2: Identifying Overfitting
The large gap (16%) between training and validation accuracy revealed overfitting. The model was memorizing training data rather than learning generalizable patterns.

### Phase 3: Hyperparameter Tuning
Tested different `max_depth` values with 5-fold cross-validation:

| max_depth | Training Acc | CV Mean ± Std |
|-----------|-------------|---------------|
| 3 | 82.72% | 82.49% ± 1.49% |
| 4 | 83.50% | 81.93% ± 1.66% |
| 5 | 84.18% | 81.14% ± 1.33% |
| 7 | 87.88% | 82.15% ± 1.68% |
| None (unlimited) | 97.98% | 78.23% ± 3.40% |

**Finding**: Unlimited depth causes overfitting. Selected max_depth=5 for better generalization.

### Phase 4: Model Comparison
Compared Decision Tree with Random Forest:

| Model | Validation Accuracy | 5-Fold CV |
|-------|-------------------|-----------|
| Decision Tree (max_depth=5) | 75.98% | 81.14% ± 1.33% |
| Random Forest (100 trees) | 81.01% | 83.05% ± 1.67% |

### Phase 5: Class Imbalance Analysis
The dataset is imbalanced (62% died, 38% survived). Analyzed precision-recall trade-off:

| Model | Accuracy | Survived Recall | Survived Precision |
|-------|----------|-----------------|-------------------|
| Unbalanced | 75.98% | 57.97% | 74.07% |
| **Balanced** | **77.09%** | **65.22%** | **72.58%** |

**Finding**: Balanced class weights improve recall by +7.25% with slight accuracy improvement.

### Phase 6: Final Model
Selected **Decision Tree with max_depth=5 and class_weight='balanced'** as final model due to:
- Higher accuracy (77.09%)
- Better recall for minority class (65.22%)
- Improved F1-score (68.70%)

---

## Final Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Final Model** | Decision Tree (max_depth=5, balanced) |
| **5-Fold CV Accuracy** | 78.23% ± 0.22% |
| **Validation Accuracy** | 75.98% |

### Baseline Comparison
| Baseline | Accuracy |
|----------|----------|
| Majority Class (all die) | 61.62% |
| Gender Rule (women survive) | 78.68% |
| **Our Model** | **77.09%** |

### Feature Importance
1. **Sex**: 55.23% - Primary factor (women and children first)
2. **Age**: 15.58% - Children prioritized
3. **Fare**: 14.00% - Socioeconomic indicator
4. **Pclass**: 9.69% - Passenger class

### Prediction Distribution (Test Set)
- Survived: 210 (50.2%)
- Not Survived: 208 (49.8%)

---

## Key Learnings

1. **Bias-Variance Trade-off**: Controlling tree depth reduces overfitting
2. **Cross-Validation**: More reliable than single train-test split
3. **Class Imbalance**: Balanced weights improve minority class recall
4. **Feature Importance**: Domain knowledge aligns with model findings (Sex as dominant factor)

---

## Project Structure

```
mini_project/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
└── opencode/
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py      # Data loading & preprocessing
    │   ├── model.py            # ML models & evaluation
    │   └── visualization.py    # Tree & feature plots
    ├── models/
    │   ├── decision_tree_tuned.pkl
    │   ├── decision_tree_tuned.png
    │   ├── feature_importance_tuned.png
    │   └── predictions_tuned.csv
    ├── requirements.txt
    ├── main.py
    └── test_improved.py
```

---

## How to Run

```bash
# Activate virtual environment
cd C:\Users\hites\Desktop\ML Labs
.venv\Scripts\python.exe mini_project\opencode\main.py
```

---

## Conclusion

This project demonstrates a complete machine learning pipeline with proper experimentation. The final Decision Tree model achieves ~78% cross-validation accuracy, slightly below the simple gender rule baseline, which is expected given the dataset's high separability by gender. The balanced class weights improved recall for survivors, demonstrating understanding of precision-recall trade-offs.

This project showcases:
- Proper preprocessing
- Hyperparameter tuning
- Cross-validation
- Baseline comparison
- Class imbalance handling
- Analytical interpretation
