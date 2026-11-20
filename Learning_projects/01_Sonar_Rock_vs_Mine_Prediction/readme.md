# Sonar Rock vs Mine Prediction using Logistic Regression

## 1. Problem Statement

The objective of this project is to build a Machine Learning classification model that can distinguish between **underwater rocks and metal mines** using sonar signal data.

Submarines use sonar technology to detect and identify objects underwater by transmitting sound waves and analysing the reflected signals.

The challenge is to determine whether an object detected by sonar is a rock (R) or a mine (M) based on the characteristics of the reflected signals.

This project uses **Logistic Regression**, a supervised Machine Learning algorithm, to perform binary classification.

## 2. Dataset Description

The project uses the Sonar dataset, which contains sonar signal measurements along with their corresponding object classifications.

The dataset consists of 61 columns:

- **60 independent features:** Numerical values representing the energy of sonar signals at different frequencies.
- **1 target variable:** The classification label indicating whether the detected object is a Rock (R) or a Mine (M).

### Target Classes

| Label | Description |
|---|---|
| R | Rock |
| M | Mine |

The objective is to train a model that learns patterns from the 60 sonar measurements and predicts the corresponding target class.

## 3. Technologies and Libraries Used

- Python
- Pandas – Data manipulation and preprocessing
- NumPy – Numerical computations
- Scikit-learn – Model development, dataset splitting and evaluation
- Jupyter Notebook / Google Colab – Development environment

## 4. Project Workflow

### Step 1: Data Loading

Loaded the sonar dataset into a Pandas DataFrame for exploration and processing.

### Step 2: Separating Features and Labels

Separated the independent variables (X) and dependent variable (Y).

```python
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

- X contains the 60 sonar signal measurements.
- Y contains the target labels (Rock or Mine).

### Step 3: Train-Test Split

Split the dataset into training and testing sets using Scikit-learn's `train_test_split()`.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=1
)
```

The dataset was divided into:

- 90% Training data
- 10% Testing data

`stratify=Y` maintains approximately the same class distribution in both datasets.

`random_state=1` ensures reproducibility of the split.

### Step 4: Model Creation and Training

Created and trained a Logistic Regression classification model.

```python
model = LogisticRegression()

model.fit(X_train, Y_train)
```

The model learns the relationship between sonar signal measurements and their corresponding classifications.

### Step 5: Model Evaluation

Evaluated the model's performance using the accuracy score from Scikit-learn.

**Training Accuracy:**

```python
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction, Y_train
)
```

**Testing Accuracy:**

```python
X_test_prediction = model.predict(X_test)

testing_data_accuracy = accuracy_score(
    X_test_prediction, Y_test
)
```

Training accuracy measures how well the model performs on the data used for learning.

Testing accuracy measures how well the model performs on previously unseen data.

## 5. Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

In this project, it predicts whether an underwater object belongs to one of two classes:

- Rock (R)
- Mine (M)

The algorithm estimates the probability of an observation belonging to a particular class and uses a decision threshold to determine its predicted classification.

## 6. Results

The model was evaluated using training and testing accuracy.

| Evaluation Metric | Result |
|---|---|
| Training Accuracy | [Add your result] |
| Testing Accuracy | [Add your result] |

*Note: Replace the placeholders with the actual accuracy values obtained when running the notebook.*

## 7. Conclusion

This project demonstrates the implementation of a supervised Machine Learning classification workflow using Logistic Regression.

It covers dataset loading, feature and target separation, train-test splitting, model training, prediction and accuracy evaluation.

The project provided practical experience with Scikit-learn and the fundamentals of building and evaluating a binary classification model using Python.

## 8. Future Improvements

Potential improvements to this project include:

- Evaluating the model using a confusion matrix.
- Calculating precision, recall and F1-score.
- Comparing Logistic Regression with other classification algorithms such as SVM, Decision Trees and Random Forest.
- Performing hyperparameter tuning.
- Using cross-validation to obtain a more reliable estimate of model performance.

---

**Project Type:** Supervised Machine Learning | Binary Classification

**Algorithm:** Logistic Regression

**Language:** Python
