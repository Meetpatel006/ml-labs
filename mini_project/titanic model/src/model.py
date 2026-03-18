from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import numpy as np


def train_model(X_train, y_train, max_depth=None, min_samples_leaf=1, min_samples_split=2, class_weight=None, random_state=42):
    """Train a Decision Tree Classifier."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train):
    """Evaluate model performance on training data."""
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    return accuracy, y_pred


def train_and_evaluate(X, y, test_size=0.2, max_depth=None, random_state=42):
    """Split data, train model, and evaluate."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model = train_model(X_train, y_train, max_depth=max_depth, random_state=random_state)
    
    train_acc, _ = evaluate_model(model, X_train, y_train)
    val_acc, val_pred = evaluate_model(model, X_val, y_val)
    
    return model, X_train, X_val, y_train, y_val, train_acc, val_acc


def hyperparameter_tuning(X, y, max_depths=[3, 4, 5, 6, 7, None], cv_folds=5):
    """Perform hyperparameter tuning with cross-validation."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - Decision Tree")
    print("="*60)
    
    results = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        train_model_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
        train_model_temp.fit(X, y)
        train_acc = accuracy_score(y, train_model_temp.predict(X))
        
        results.append({
            'max_depth': depth,
            'train_acc': train_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        depth_str = str(depth) if depth else "None (unlimited)"
        print(f"max_depth={depth_str:>8} | Train: {train_acc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    best_result = max(results, key=lambda x: x['cv_mean'])
    print(f"\n>>> Best: max_depth={best_result['max_depth']} with CV accuracy: {best_result['cv_mean']:.4f}")
    
    return results, best_result


def cross_validate_model(X, y, model, cv_folds=5):
    """Perform K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return cv_scores


def compare_class_weights(X, y, max_depth=5, test_size=0.2, random_state=42):
    """Compare models with different class weights."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("\n" + "="*60)
    print("CLASS WEIGHT COMPARISON")
    print("="*60)
    
    results = {}
    
    for weight in [None, 'balanced']:
        name = "Unbalanced" if weight is None else "Balanced"
        model = train_model(X_train, y_train, max_depth=max_depth, class_weight=weight, random_state=random_state)
        
        y_pred = model.predict(X_val)
        prf = precision_recall_fscore_support(y_val, y_pred, average=None)
        
        acc = accuracy_score(y_val, y_pred)
        
        results[weight] = {
            'accuracy': acc,
            'precision_0': prf[0][0],
            'recall_0': prf[1][0],
            'precision_1': prf[0][1],
            'recall_1': prf[1][1],
            'f1_0': prf[2][0],
            'f1_1': prf[2][1]
        }
        
        print(f"\n{name} (class_weight={weight}):")
        print(f"  Accuracy:                 {acc:.4f}")
        print(f"  Class 0 (Not Survived):   Precision: {prf[0][0]:.4f}, Recall: {prf[1][0]:.4f}, F1: {prf[2][0]:.4f}")
        print(f"  Class 1 (Survived):       Precision: {prf[0][1]:.4f}, Recall: {prf[1][1]:.4f}, F1: {prf[2][1]:.4f}")
    
    print(f"\n>>> Recall for Survived improved: {results['balanced']['recall_1'] - results[None]['recall_1']:.2%}")
    print(f">>> Accuracy changed: {results['balanced']['accuracy'] - results[None]['accuracy']:.2%}")
    
    return results


def compare_models(X, y, test_size=0.2, random_state=42):
    """Compare Decision Tree vs Random Forest."""
    print("\n" + "="*60)
    print("MODEL COMPARISON: Decision Tree vs Random Forest")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Decision Tree (best from tuning)
    dt_model = train_model(X_train, y_train, max_depth=5, random_state=random_state)
    dt_train_acc, _ = evaluate_model(dt_model, X_train, y_train)
    dt_val_acc, _ = evaluate_model(dt_model, X_val, y_val)
    dt_cv_scores = cross_validate_model(X, y, dt_model)
    
    print(f"\nDecision Tree (max_depth=5):")
    print(f"  Training Accuracy:    {dt_train_acc:.4f}")
    print(f"  Validation Accuracy:  {dt_val_acc:.4f}")
    print(f"  5-Fold CV:            {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}")
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=5, random_state=random_state)
    rf_train_acc, _ = evaluate_model(rf_model, X_train, y_train)
    rf_val_acc, _ = evaluate_model(rf_model, X_val, y_val)
    rf_cv_scores = cross_validate_model(X, y, rf_model)
    
    print(f"\nRandom Forest (n_estimators=100, max_depth=5):")
    print(f"  Training Accuracy:    {rf_train_acc:.4f}")
    print(f"  Validation Accuracy:  {rf_val_acc:.4f}")
    print(f"  5-Fold CV:            {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    
    print(f"\n>>> Random Forest improves validation by: {(rf_val_acc - dt_val_acc)*100:.2f}%")
    
    return {
        'dt': {'train': dt_train_acc, 'val': dt_val_acc, 'cv': dt_cv_scores.mean(), 'cv_std': dt_cv_scores.std()},
        'rf': {'train': rf_train_acc, 'val': rf_val_acc, 'cv': rf_cv_scores.mean(), 'cv_std': rf_cv_scores.std()}
    }


def save_model(model, filepath):
    """Save model to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """Load model from a pickle file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
