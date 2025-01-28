# -------------------- Imports --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import scipy.stats as stats

# -------------------- Data Loading & Preparation --------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=column_names, na_values="?")

# Handle missing values
for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(float)
df['target'] = (df['target'] > 0).astype(int)

# -------------------- Visualization Data Preparation --------------------
df_vis = df.copy()
df_vis['cp'] = df_vis['cp'].map({
    0: 'Typical Angina', 
    1: 'Atypical Angina', 
    2: 'Non-anginal', 
    3: 'Asymptomatic'
})
df_vis['restecg'] = df_vis['restecg'].map({
    0: 'Normal', 
    1: 'ST-T Abnormality', 
    2: 'LV Hypertrophy'
})
df_vis['thal'] = df_vis['thal'].map({
    3.0: 'Normal', 
    6.0: 'Fixed Defect', 
    7.0: 'Reversible Defect'
})

# -------------------- Visualizations --------------------
sns.set(style="whitegrid", palette="pastel")

# 1. Target Distribution
plt.figure(figsize=(8, 5))
df['target'].value_counts().plot.pie(autopct='%1.1f%%', 
                                    labels=['Healthy', 'Heart Disease'],
                                    colors=['#66b3ff','#ff9999'],
                                    explode=(0.1, 0))
plt.title('Heart Disease Distribution', fontsize=16)
plt.ylabel('')
plt.savefig('target_distribution.png', bbox_inches='tight')
plt.close()

# 2. Numerical Features Distribution
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=feature, hue='target', 
                 kde=True, palette={0: '#66b3ff', 1: '#ff9999'},
                 element='step', stat='density', common_norm=False)
    plt.title(f'{feature.title()} Distribution', fontsize=12)
plt.tight_layout()
plt.savefig('numerical_distributions.png', bbox_inches='tight')
plt.close()

# 3. Categorical Features Analysis
cat_features = ['cp', 'restecg', 'thal']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df_vis, x=feature, hue='target', 
                  palette={0: '#66b3ff', 1: '#ff9999'})
    plt.title(f'{feature.upper()} Distribution', fontsize=12)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('categorical_distributions.png', bbox_inches='tight')
plt.close()

# 4. Correlation Matrix
plt.figure(figsize=(12, 8))
corr = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask, 
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.close()

# -------------------- Model Training --------------------
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing setup
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
cat_cols = ['cp', 'restecg', 'thal']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Outlier removal
z_scores = np.abs(stats.zscore(X_train[num_cols]))
train_filter = (z_scores < 3).all(axis=1)
X_train_clean, y_train_clean = X_train.loc[train_filter], y_train.loc[train_filter]

# Model pipelines
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
]).fit(X_train_clean, y_train_clean)

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=5,
        min_samples_split=10,
        random_state=42
    ))
]).fit(X_train_clean, y_train_clean)

# -------------------- Model Evaluation --------------------
def save_model_metrics(y_true, y_pred, y_proba, model_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Healthy', 'Predicted Disease'],
                yticklabels=['Actual Healthy', 'Actual Disease'])
    plt.title(f'{model_name} Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_metrics.png', bbox_inches='tight')
    plt.close()
    
    # Save classification report and ROC AUC
    report = classification_report(y_true, y_pred)
    
    with open(f'{model_name.lower().replace(" ", "_")}_report.txt', 'w') as f:
        f.write(report)
        f.write(f'\nROC AUC: {roc_auc:.4f}\n')

# Evaluate models on test set
models = [
    (lr_pipeline, 'Logistic Regression'),
    (dt_pipeline, 'Decision Tree')
]

for model, name in models:
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    save_model_metrics(y_test, y_test_pred, y_test_proba, f'{name} Test')

print("""
=== Execution Complete ===
Saved output files:
- Visualizations:
  - target_distribution.png
  - numerical_distributions.png
  - categorical_distributions.png
  - correlation_matrix.png
- Model metrics:
  - logistic_regression_test_metrics.png
  - decision_tree_test_metrics.png
  - logistic_regression_test_report.txt
  - decision_tree_test_report.txt
""")