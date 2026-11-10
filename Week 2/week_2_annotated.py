"""
===============================================================================
Topic: Ensemble Learning Methods (Boosting, Bagging, Hard & Soft Voting Ensembles)
Libraries: scikit-learn, pandas, numpy, matplotlib
===============================================================================
"""

# Import Core Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Datasets & Generation
from sklearn.datasets import load_breast_cancer, make_classification

# Data Splitting, Cross-Validation & Preprocessing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)
from sklearn.preprocessing import LabelEncoder

# Evaluation Metrics
from sklearn.metrics import accuracy_score, f1_score

# Base Learners (Weak & Base Classifiers)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensemble Frameworks
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier
)


# =============================================================================
# ACTIVITY 1 — BOOSTING WITH ADABOOST (ADAPTIVE BOOSTING)
# =============================================================================
# CONCEPT: Boosting trains weak learners SEQUENTIALLY. Each subsequent model
# focuses more on instances misclassified by previous models by re-weighting data.
# Goal: Reduce BIAS (and variance).

# Step 1: Load Dataset & Target Encoding
breast_cancer = load_breast_cancer()

# Create Feature DataFrame
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

# Convert integer targets to categorical strings (e.g., 'malignant', 'benign')
y = pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)

# Encode string labels back to numeric integers for scikit-learn compatibility
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("--- Activity 1: Dataset Overview ---")
print("Dataset shape:", X.shape)
print("Target classes:", encoder.classes_)  # Returns array of mapped class names


# Step 2: Train/Test Split with Stratification
# EXAM TIP: Always use `stratify=y` on classification tasks to ensure both train
# and test sets maintain identical class label proportions (crucial for imbalanced data).
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1,
    stratify=y
)
print(f"Training observations: {X_train.shape[0]} | Testing observations: {X_test.shape[0]}")


# Step 3: Build & Train AdaBoost Model
# A "Decision Stump" is a Decision Tree constrained to max_depth=1 (a single split).
# Weak learners must perform slightly better than random guessing (accuracy > 50%).
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=1
    ),
    n_estimators=200,  # Number of sequential boosting rounds/stumps
    random_state=1
)
boosting_model.fit(X_train, y_train)

# Model Evaluation
boost_predictions = boosting_model.predict(X_test)
boost_accuracy = accuracy_score(y_test, boost_predictions)
boost_f1 = f1_score(y_test, boost_predictions)

print("\n--- AdaBoost Performance ---")
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")


# =============================================================================
# ACTIVITY 2 — BAGGING WITH RANDOM FOREST (BOOTSTRAP AGGREGATING)
# =============================================================================
# CONCEPT: Bagging trains weak/base learners IN PARALLEL on independent bootstrap 
# samples (random sampling with replacement). Aggregates via majority vote/averaging.
# Goal: Reduce VARIANCE (prevents overfitting).

# Random Forest adds feature randomness (subsampling random features at each split)
# on top of row bootstrapping.
bagging_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=1,       # Constrained depth to match decision stumps for controlled comparison
    random_state=1,
    n_jobs=-1          # n_jobs=-1 utilizes all available CPU cores for parallel execution
)
bagging_model.fit(X_train, y_train)

# Model Evaluation
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
bagging_f1 = f1_score(y_test, bagging_predictions)

print("\n--- Random Forest Performance ---")
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

# Model Comparison Summary:
# AdaBoost vs. Random Forest comparison table structure:
# -----------------------------------------------------------------------------
# Model         | Accuracy | F1 Score | Learning Strategy
# -----------------------------------------------------------------------------
# AdaBoost      |  0.96    |   0.97   | Sequential (Focus on misclassified instances)
# Random Forest |  0.92    |   0.94   | Parallel (Bootstrap + Random Feature Subsets)
# -----------------------------------------------------------------------------


# =============================================================================
# ACTIVITY 3 — BAGGING WITH NON-TREE BASE LEARNERS (KNN)
# =============================================================================
# CONCEPT: Decision trees are high-variance learners, so bagging excels with them.
# However, BaggingClassifier can wrap ANY base algorithm (e.g., KNN, SVM).

# Step 1: Generate Synthetic Classification Dataset
X_knn, y_knn = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=5
)
print("\n--- Activity 3: KNN Bagging Dataset Shape ---", X_knn.shape)


# Step 2: Create KNN Bagging Classifier
# API NOTE: Use `estimator=` (newer scikit-learn syntax); avoid `base_estimator=`.
knn_bagging = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)


# Step 3: Repeated Stratified Cross-Validation Evaluation
# EXAM TIP: RepeatedStratifiedKFold runs K-Fold CV multiple times with different random splits,
# giving a much more reliable estimate of mean performance and stability (low std dev).
cv = RepeatedStratifiedKFold(
    n_splits=10,       # 10 folds per repeat
    n_repeats=3,        # 3 repeats -> 30 total evaluation runs
    random_state=1
)

knn_scores = cross_val_score(
    knn_bagging,
    X_knn,
    y_knn,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    error_score="raise"
)

print("\n--- KNN Bagging Cross-Validation ---")
print(f"Mean Accuracy: {knn_scores.mean():.3f}")
print(f"Standard Deviation: {knn_scores.std():.3f}")
# Lower Std Dev indicates high model stability across varying data subsets.


# =============================================================================
# ACTIVITY 4 — SOFT VOTING ENSEMBLE (PROBABILITY-WEIGHTED VOTING)
# =============================================================================
# CONCEPT: Soft voting sums predicted class PROBABILITIES from each classifier 
# and selects the class with the highest average probability.
# Key Advantage: Gives greater weight to models that are highly confident in their prediction.
# REQUIREMENT: All base models MUST support probability estimation (e.g., SVC(probability=True)).

# Step 1: Synthetic Dataset for SVM Ensembles
X_svm, y_svm = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=2
)


# Step 2: Define Soft Voting Ensemble Builder
# Uses 5 Support Vector Machines with increasing Polynomial Kernel degrees (1 through 5).
def create_soft_voting_model():
    models = [
        ("svm1", SVC(probability=True, kernel="poly", degree=1, random_state=1)),
        ("svm2", SVC(probability=True, kernel="poly", degree=2, random_state=1)),
        ("svm3", SVC(probability=True, kernel="poly", degree=3, random_state=1)),
        ("svm4", SVC(probability=True, kernel="poly", degree=4, random_state=1)),
        ("svm5", SVC(probability=True, kernel="poly", degree=5, random_state=1))
    ]
    ensemble = VotingClassifier(
        estimators=models,
        voting="soft"   # Combines predicted probabilities: P_avg(y=c) = (1/M) * sum(P_m(y=c))
    )
    return ensemble


# Step 3: Package Individual Models & Ensemble for Benchmark Comparison
def create_soft_voting_models():
    models = {
        "SVM-1": SVC(probability=True, kernel="poly", degree=1, random_state=1),
        "SVM-2": SVC(probability=True, kernel="poly", degree=2, random_state=1),
        "SVM-3": SVC(probability=True, kernel="poly", degree=3, random_state=1),
        "SVM-4": SVC(probability=True, kernel="poly", degree=4, random_state=1),
        "SVM-5": SVC(probability=True, kernel="poly", degree=5, random_state=1),
        "Soft Voting": create_soft_voting_model()
    }
    return models


# Step 4: Evaluate Individual SVMs vs. Soft Voting Ensemble via Cross-Validation
print("\n--- Activity 4: Soft Voting SVM Evaluation ---")
models_soft = create_soft_voting_models()
results_soft = []
names_soft = []

for name, model in models_soft.items():
    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )
    results_soft.append(scores)
    names_soft.append(name)
    print(f"{name:12s} Mean = {scores.mean():.3f} | Std = {scores.std():.3f}")


# Step 5: Visualize Soft Voting Results with Boxplot
plt.figure(figsize=(10, 5))
plt.boxplot(
    results_soft,
    tick_labels=names_soft,
    showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Soft Voting vs Individual SVM Models (Polynomial Kernels)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# =============================================================================
# ACTIVITY 5 — HARD VOTING ENSEMBLE (MAJORITY CLASS VOTING)
# =============================================================================
# CONCEPT: Hard voting counts predicted CLASS LABELS from each classifier and 
# picks the class receiving the simple majority of votes.
# Best Practice: Use an ODD number of base classifiers to avoid tie votes.

# Step 1: Define Hard Voting KNN Ensemble (k = 1, 3, 5, 7, 9)
def create_hard_voting_model():
    models = [
        ("knn1", KNeighborsClassifier(n_neighbors=1)),
        ("knn3", KNeighborsClassifier(n_neighbors=3)),
        ("knn5", KNeighborsClassifier(n_neighbors=5)),
        ("knn7", KNeighborsClassifier(n_neighbors=7)),
        ("knn9", KNeighborsClassifier(n_neighbors=9))
    ]
    ensemble = VotingClassifier(
        estimators=models,
        voting="hard"   # Counts discrete class labels: y_pred = mode(y_1, y_2, ..., y_M)
    )
    return ensemble


# Step 2: Package Models for Benchmark Comparison
def create_hard_voting_models():
    models = {
        "KNN-1": KNeighborsClassifier(n_neighbors=1),
        "KNN-3": KNeighborsClassifier(n_neighbors=3),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "KNN-7": KNeighborsClassifier(n_neighbors=7),
        "KNN-9": KNeighborsClassifier(n_neighbors=9),
        "Hard Voting": create_hard_voting_model()
    }
    return models


# Step 3: Evaluate Individual KNNs vs. Hard Voting Ensemble
print("\n--- Activity 5: Hard Voting KNN Evaluation ---")
models_hard = create_hard_voting_models()
results_hard = []
names_hard = []

for name, model in models_hard.items():
    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )
    results_hard.append(scores)
    names_hard.append(name)
    print(f"{name:12s} Mean = {scores.mean():.3f} | Std = {scores.std():.3f}")


# Step 4: Visualize Hard Voting Results
plt.figure(figsize=(10, 5))
plt.boxplot(
    results_hard,
    tick_labels=names_hard,
    showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Hard Voting vs Individual KNN Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# =============================================================================

# =============================================================================
"""
-------------------------------------------------------------------------------
1. BAGGING (Bootstrap Aggregating)
   - Mechanism: Parallel training on random bootstrap samples (with replacement).
   - Primary Goal: Reduce Variance (prevents overfitting).
   - Ideal For: High-variance base models (e.g., deep Decision Trees).
   - Scikit-Learn Class: BaggingClassifier(estimator=..., n_estimators=...)

2. BOOSTING (e.g., AdaBoost, Gradient Boosting)
   - Mechanism: Sequential training; updates sample weights so subsequent models
     focus on errors/misclassifications made by prior models.
   - Primary Goal: Reduce Bias (builds strong model from weak learners).
   - Ideal For: Weak base learners (e.g., Decision Stumps with max_depth=1).
   - Scikit-Learn Class: AdaBoostClassifier(estimator=..., n_estimators=...)

3. HARD VOTING
   - Mechanism: Majority vote over predicted discrete class labels.
   - Rule: Requires an odd number of models to avoid tie breaks.
   - Scikit-Learn Class: VotingClassifier(estimators=..., voting='hard')

4. SOFT VOTING
   - Mechanism: Weighted/unweighted average of predicted class probabilities.
   - Advantage: Leverages prediction confidence; usually outperforms Hard Voting.
   - Rule: All base models must support `.predict_proba()` (e.g., SVC(probability=True)).
   - Scikit-Learn Class: VotingClassifier(estimators=..., voting='soft')
-------------------------------------------------------------------------------
"""
