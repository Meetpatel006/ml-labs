import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os


def plot_decision_tree(model, feature_names, output_path=None, figsize=(20, 10)):
    """Plot and save the Decision Tree visualization."""
    
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['Not Survived', 'Survived'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Titanic Survival Decision Tree', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Decision Tree saved to: {output_path}")
    
    plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, output_path=None):
    """Plot feature importance chart."""
    
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance saved to: {output_path}")
    
    plt.show()
    plt.close()
