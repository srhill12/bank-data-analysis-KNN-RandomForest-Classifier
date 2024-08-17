
# Bank Marketing Data Analysis

This project aims to analyze and model bank marketing data using machine learning techniques. The project involves data preprocessing, encoding categorical variables, applying dimensionality reduction, and training various machine learning models.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation

To run the project, you'll need Python and the necessary libraries installed. You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:

```plaintext
├── bank_marketing_analysis.ipynb  # Jupyter notebook with data analysis and model training
├── ml_utils.py                    # Python file with utility functions for preprocessing and encoding
├── README.md                      # Project documentation
└── requirements.txt               # List of required libraries
```

## Data Preparation

### Importing the Data

The data is imported directly from a URL as a CSV file into a Pandas DataFrame:

```python
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/bank_marketing.csv')
```

### Splitting the Data

The data is split into training and testing sets using the `train_test_split_marketing` function:

```python
X_train, X_test, y_train, y_test = train_test_split_marketing(df)
```

### Handling Missing Values

Missing values in categorical columns are filled using various strategies provided in the `ml_utils.py`:

```python
X_train_filled = fill_missing(X_train)
X_test_filled = fill_missing(X_test)
```

### Encoding Categorical Variables

Categorical variables are encoded using One-Hot Encoding and Ordinal Encoding strategies defined in `ml_utils.py`:

```python
encoders = build_encoders(X_train_filled)
X_train_encoded = encode_categorical(X_train_filled, encoders)
X_test_encoded = encode_categorical(X_test_filled, encoders)
```

### Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset:

```python
from sklearn.decomposition import PCA

pca_model = PCA(n_components=10)
pca_model.fit(X_train_encoded)

X_train_pca = pd.DataFrame(pca_model.transform(X_train_encoded))
X_test_pca = pd.DataFrame(pca_model.transform(X_test_encoded))
```

## Model Training

### Random Forest Classifier

A Random Forest model is trained using the PCA-transformed data:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_pca, y_train_encoded)
```

### K-Nearest Neighbors Classifier

Hyperparameter tuning is performed on the K-Nearest Neighbors model using `RandomizedSearchCV`:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['uniform', 'distance'],
    'leaf_size': [10, 50, 100, 500]
}
random_knn = RandomizedSearchCV(KNeighborsClassifier(), param_grid, verbose=3)
random_knn.fit(X_train_pca, y_train_encoded)
```

## Hyperparameter Tuning

The project includes hyperparameter tuning for both the Random Forest and K-Nearest Neighbors models to optimize performance.

### Random Forest Tuning

The `max_depth` and `n_estimators` parameters of the Random Forest are tuned:

```python
# Tuning max_depth
for depth in range(1,10):
    model = RandomForestClassifier(n_estimators=100, max_depth=depth)
    model.fit(X_train_pca, y_train_encoded)

# Tuning n_estimators
for n in [50, 100, 500, 1000]:
    model = RandomForestClassifier(n_estimators=n, max_depth=7)
    model.fit(X_train_pca, y_train_encoded)
```

## Evaluation

The models are evaluated using the `balanced_accuracy_score` metric on the test dataset:

```python
from sklearn.metrics import balanced_accuracy_score

y_test_pred = model.predict(X_test_pca)
print(balanced_accuracy_score(y_test_encoded, y_test_pred))
```

## Results

The project's results include balanced accuracy scores for the different models and configurations. The final scores will help determine the best model and configuration for predicting the target variable in the bank marketing dataset.

---

This README provides a structured overview of the project, including how the data is processed, how the models are trained, and how they are evaluated. If you need further customization or additional sections, feel free to ask!
