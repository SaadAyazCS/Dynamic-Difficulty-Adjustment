# Full implementation of Dynamic Difficulty Adjustment (DDA) system
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Step 1: Simulate player behavior data with logical difficulty assignments (increased to 500 samples)
    np.random.seed(42)
    data = {
        'ReactionTime': np.random.normal(1.2, 0.3, 500),
        'Accuracy': np.random.randint(50, 100, 500),
        'TimeSpent': np.random.randint(60, 300, 500),
        'EnemiesDefeated': np.random.randint(0, 30, 500),
        'Score': np.random.randint(100, 1000, 500)
    }
    df = pd.DataFrame(data)

    def assign_difficulty(row):
        if row['Accuracy'] > 80 and row['ReactionTime'] < 1.1 and row['Score'] > 650:  # Slightly relaxed thresholds
            return 'Hard'
        elif row['Accuracy'] > 60 and row['ReactionTime'] < 1.6 and row['Score'] > 350:  # Adjusted for balance
            return 'Medium'
        else:
            return 'Easy'

    df['DifficultyLabel'] = df.apply(assign_difficulty, axis=1)
    print("Figure 4: Terminal Output of Model Evaluation - Sample Data:\n", df.head())

    # Step 2: Encode target labels
    le = LabelEncoder()
    df['DifficultyEncoded'] = le.fit_transform(df['DifficultyLabel'])

    # Step 3: Select features and target
    X = df[['ReactionTime', 'Accuracy', 'TimeSpent', 'EnemiesDefeated', 'Score']]
    y = df['DifficultyEncoded']

    # Step 4: Split data with stratification (adjusted for larger dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 5: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict and evaluate
    y_pred = model.predict(X_test)
    print("\nFigure 4: Terminal Output of Model Evaluation - Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nFigure 4: Terminal Output of Model Evaluation - Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Step 7: Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFigure 4: Terminal Output of Model Evaluation - Feature Importance:\n", feature_importance)

    # Step 8: Visualize Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Figure 2: Confusion Matrix Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 9: Visualize Accuracy by Difficulty Level
    recall_scores = [cm[i, i] / cm[i].sum() for i in range(len(le.classes_))]
    plt.figure(figsize=(8, 6))
    plt.bar(le.classes_, recall_scores, color=['#36A2EB', '#FFCE56', '#FF6384'])
    plt.title('Figure 3: Accuracy by Difficulty Level')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Recall (Accuracy per Class)')
    plt.ylim(0, 1)
    for i, v in enumerate(recall_scores):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")