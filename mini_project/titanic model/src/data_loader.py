import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir):
    """Load training and test datasets."""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess the data for Decision Tree model."""
    
    # Combine for consistent preprocessing
    combined = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0)
    
    # Features to use
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Handle missing values
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    combined['Sex'] = le_sex.fit_transform(combined['Sex'])
    combined['Embarked'] = le_embarked.fit_transform(combined['Embarked'])
    
    # Select features
    X = combined[features]
    
    # Split back to train and test
    X_train = X.iloc[:len(train_df)]
    X_test = X.iloc[len(train_df):]
    y_train = train_df['Survived']
    
    return X_train, X_test, y_train, features


def get_feature_names():
    """Return feature names used in the model."""
    return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
