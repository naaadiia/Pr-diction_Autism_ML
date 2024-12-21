#!/usr/bin/env python
# coding: utf-8

# In[1]:


#arbre decesion 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement et préparation des données (même code jusqu'à SMOTE)
df = pd.read_csv(r"C:\Users\ksiin\OneDrive\Bureau\autism_VF.csv")
df.rename(columns={'Class/ASD': 'Class_ASD'}, inplace=True)
df['Class_ASD'] = df['Class_ASD'].map({'NO': 0, 'YES': 1})

X = df.drop('Class_ASD', axis=1)
y = df['Class_ASD']

categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    X[col] = X[col].fillna(X[col].mode()[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Préprocesseurs
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prétraitement et SMOTE
X_train_preprocessed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Decision Tree avec GridSearchCV
dt_pipeline = Pipeline(steps=[
    ('classifier', DecisionTreeClassifier(random_state=42))
])

dt_param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

dt_grid_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
dt_grid_search.fit(X_train_resampled, y_train_resampled)
best_dt_model = dt_grid_search.best_estimator_

# Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    best_dt_model, X_train_resampled, y_train_resampled, 
    cv=StratifiedKFold(n_splits=5), scoring='f1', 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve (Decision Tree)")
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('F1-score')
plt.legend(loc='best')
plt.grid()
plt.show()

# Évaluation
# Pour chaque modèle (après les prédictions)
X_test_preprocessed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_preprocessed)

print(f"--------------------{model_name}--------------------")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['0', '1'], digits=2))

# Matrice de confusion
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion - {model_name}')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')
plt.show()

# Matrice de confusion
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion - Decision Tree')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')
plt.show()


# In[ ]:


#random forest
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement et préparation des données (même code jusqu'à SMOTE)
df = pd.read_csv(r"C:\Users\ksiin\OneDrive\Bureau\autism_VF.csv")
df.rename(columns={'Class/ASD': 'Class_ASD'}, inplace=True)
df['Class_ASD'] = df['Class_ASD'].map({'NO': 0, 'YES': 1})

X = df.drop('Class_ASD', axis=1)
y = df['Class_ASD']

categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    X[col] = X[col].fillna(X[col].mode()[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Préprocesseurs
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prétraitement et SMOTE
X_train_preprocessed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Random Forest avec GridSearchCV
rf_pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
rf_grid_search.fit(X_train_resampled, y_train_resampled)
best_rf_model = rf_grid_search.best_estimator_

# Courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X_train_resampled, y_train_resampled, 
    cv=StratifiedKFold(n_splits=5), scoring='f1', 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve (Random Forest)")
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('F1-score')
plt.legend(loc='best')
plt.grid()
plt.show()

# Évaluation
X_test_preprocessed = preprocessor.transform(X_test)
y_pred = best_rf_model.predict(X_test_preprocessed)

print("--------------------Random Forest--------------------")
print(f"Best Hyperparameters: {rf_grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-score: {f1_score(y_test, y_pred):.2f}")

# Matrice de confusion
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion - Random Forest')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')
plt.show()

