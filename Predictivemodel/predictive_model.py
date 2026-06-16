import os
import shutil
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def setup_dataset():
    """
    Cleans up any directory named 'Titanic-Dataset.csv' and downloads the
    actual Titanic CSV file if not already present.
    """
    filepath = "Titanic-Dataset.csv"
    if os.path.exists(filepath):
        if os.path.isdir(filepath):
            print(f"Cleaning up: '{filepath}' is a directory. Removing to download the dataset...")
            shutil.rmtree(filepath)
        else:
            print(f"Dataset file '{filepath}' already exists locally.")
            return filepath

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print(f"Downloading dataset from {url}...")
    try:
        # Use a request with user-agent header to avoid blocked requests
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Dataset downloaded and saved as 'Titanic-Dataset.csv'.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise e
    return filepath

def preprocess_data(df):
    """
    Cleans the Titanic dataset, performs feature engineering, and encodes categorical variables.
    """
    print("\n--- Data Preprocessing & Feature Engineering ---")
    df = df.copy()
    
    # 1. Handle Missing Values
    # Impute Age using median of Pclass and Sex groups
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Impute Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Impute Fare (if missing) with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # 2. Feature Engineering
    # Title extraction from Passenger Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. Categorical Encoding
    # Map Sex to binary
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encoding for Title and Embarked
    df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)
    
    # 4. Feature Selection
    # Drop irrelevant features
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    print(f"Preprocessed dataset shape: {df.shape}")
    return df

def train_and_evaluate():
    # Setup dataset file
    filepath = setup_dataset()
    
    # Load data
    df = pd.read_csv(filepath)
    print("\n--- Initial Dataset Info ---")
    print(f"Total Rows: {len(df)}")
    print(f"Survival Rate: {df['Survived'].mean() * 100:.2f}%")
    
    # Preprocess
    df_clean = preprocess_data(df)
    
    # Split features and target
    X = df_clean.drop(columns=['Survived'])
    y = df_clean['Survived']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6)
    }
    
    results = {}
    y_probs = {}
    y_preds = {}
    
    print("\n--- Model Training and Evaluation ---")
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        y_preds[name] = preds
        y_probs[name] = probs
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'model_object': model
        }
        
        print(f"{name:20} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
    # Find Best Model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
    print(f"\n>>> Best Model: {best_model_name} (F1-Score: {results[best_model_name]['F1-Score']:.4f})")
    
    # Visualizations
    visualize_performance(results, best_model_name, y_test, y_preds, y_probs, X.columns)
    
    # Print comparison table
    df_results = pd.DataFrame(results).T.drop(columns=['model_object'])
    print("\n--- Model Comparison Summary ---")
    print(df_results.to_string())

def visualize_performance(results, best_model_name, y_test, y_preds, y_probs, feature_names):
    """
    Generates a beautifully styled, high-quality composite performance dashboard
    and saves it to 'model_performance.png'.
    """
    print("\n--- Generating Visualizations ---")
    # Set premium plotting styles
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'figure.titlesize': 18,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle('Titanic Survival Prediction - Model Performance Dashboard', y=0.98, weight='bold', color='#1e293b')
    
    # 1. ROC Curve
    ax1 = fig.add_subplot(1, 3, 1)
    colors = {'Logistic Regression': '#6366f1', 'Decision Tree': '#f59e0b', 'Random Forest': '#10b981'}
    for name in results.keys():
        fpr, tpr, _ = roc_curve(y_test, y_probs[name])
        auc_val = results[name]['ROC-AUC']
        ax1.plot(fpr, tpr, color=colors[name], lw=2.5, label=f'{name} (AUC = {auc_val:.3f})')
    
    ax1.plot([0, 1], [0, 1], color='#cbd5e1', lw=1.5, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', labelpad=10, weight='semibold', color='#475569')
    ax1.set_ylabel('True Positive Rate', labelpad=10, weight='semibold', color='#475569')
    ax1.set_title('ROC Curves Comparison', pad=15, weight='bold', color='#334155')
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#e2e8f0')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Confusion Matrix for the Best Model
    ax2 = fig.add_subplot(1, 3, 2)
    cm = confusion_matrix(y_test, y_preds[best_model_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['Perished', 'Survived'],
                yticklabels=['Perished', 'Survived'])
    ax2.set_xlabel('Predicted Label', labelpad=10, weight='semibold', color='#475569')
    ax2.set_ylabel('True Label', labelpad=10, weight='semibold', color='#475569')
    ax2.set_title(f'Confusion Matrix: {best_model_name}', pad=15, weight='bold', color='#334155')
    
    # 3. Feature Importance (using Random Forest Classifier)
    ax3 = fig.add_subplot(1, 3, 3)
    rf_model = results['Random Forest']['model_object']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Get top 10 features
    top_n = min(10, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    sns.barplot(x=top_importances, y=top_features, ax=ax3, palette='viridis', hue=top_features, legend=False)
    ax3.set_xlabel('Relative Importance', labelpad=10, weight='semibold', color='#475569')
    ax3.set_title('Top 10 Feature Importances (Random Forest)', pad=15, weight='bold', color='#334155')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'model_performance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Performance plots successfully saved to: {output_path}")

if __name__ == '__main__':
    train_and_evaluate()