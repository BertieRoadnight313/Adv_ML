# Import Core Libraries
import numpy as np
import pandas as pd

# Visualisation
from matplotlib import pyplot as plt

# Datasets
from sklearn.datasets import load_breast_cancer, make_classification

#Data splitting and validation
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)

# Pre-processing
from sklearn.preprocessing import LabelEncoder

# Evaluation
from sklearn.metrics import accuracy_score, f1_score

# Base learners

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensemble methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier
)

### 1- Boosting with AdaBoost

# Load dataset
breast_cancer = load_breast_cancer()

# Create feature DataFrame
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)
# Create categorical target
y = pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)
# Convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Dataset shape:", X.shape)
print("Target classes:", encoder.classes_)



## 2 - Create/ train test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1,
    stratify=y #Why stratify=y?It helps maintain approximately the same class proportions in the training and test sets.
)
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])

#3 — Build the AdaBoost model
# We will use a decision stump as the weak learner.
# A decision stump is simply a decision tree with:
# max_depth=1

boosting_model = AdaBoostClassifier(
 estimator=DecisionTreeClassifier(
    max_depth=1,
    random_state=1
 ),
 n_estimators=200,
 random_state=1
)
boosting_model.fit(X_train, y_train)
boost_predictions = boosting_model.predict(X_test)

boost_accuracy = accuracy_score(
 y_test,
 boost_predictions
)
boost_f1 = f1_score(
 y_test,
 boost_predictions
)
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")

## Activity 2 — Bagging with Random Forest
#Step 1 — Create the Random Forest For a controlled comparison, we will again use shallow decision trees.

bagging_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=1,
    random_state=1,
    n_jobs=-1
)
bagging_model.fit(X_train, y_train)

bagging_predictions = bagging_model.predict(X_test)

bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)

bagging_f1 = f1_score(
    y_test,
    bagging_predictions
)

print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

##Compare the two models
#make into table
#Model Accuracy / F1 /Learning strategy
#AdaBoost
#Random Forest

#Activity 3 — Bagging with an Alternative Base Learner ( Decision trees are commonly used in bagging because they can be high-variance learners.
#However:
#Can bagging be used with other algorithms?
#Yes.
#In this activity, we use K-nearest neighbours (KNN) as the base learner.

#Step 1 — Generate a classification dataset For this experiment we will use a synthetic dataset.
X_knn, y_knn = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=5
)

print("Dataset shape:", X_knn.shape)

#tep 2 — Create KNN Bagging
#Important API note
#Current versions of scikit-learn use:
#estimator=
#rather than the older:
#base_estimator=
# Therefore, use the following version

knn_bagging = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

#Step 3 — Repeated Stratified Cross-Validation
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
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
print(f"Mean Accuracy: {knn_scores.mean():.3f}")
print(f"Standard Deviation: {knn_scores.std():.3f}")

#A lower standard deviation generally indicates that the model's performance is more
#consistent across the cross-validation folds.

#Activity 4 — Soft Voting Ensemble

#Soft voting combines the predicted class probabilities from several classifiers and selects the
#class with the highest combined probability.
#The lecture notes that soft voting can give more influence to classifiers that are more confident
#in their predictions. Ensemble methods

#Step 1 — Create the dataset
X_svm, y_svm = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=2
)

#Step 2 — Create the soft-voting ensemble
#We will create five SVM models using polynomial kernels.
#The polynomial degrees are:
#1, 2, 3, 4, 5
def create_soft_voting_model():
 models = [
    ("svm1", SVC(
     probability=True,
     kernel="poly",
     degree=1,
     random_state=1
    )),

    ("svm2", SVC(
         probability=True,
         kernel="poly",
         degree=2,
         random_state=1
    )),

     ("svm3", SVC(
         probability=True,
         kernel="poly",
         degree=3,
         random_state=1
     )),
     ("svm4", SVC(
         probability=True,
         kernel="poly",
         degree=4,
         random_state=1
     )),
     ("svm5", SVC(
         probability=True,
         kernel="poly",
         degree=5,
         random_state=1
     ))
 ]
 ensemble = VotingClassifier(
     estimators=models,
     voting="soft"
 )
 return ensemble

#Step 3 — Create models for comparison
def create_soft_voting_models():

    models = {
        "SVM-1": SVC(
            probability=True,
            kernel="poly",
            degree=1,
            random_state=1
        ),

        "SVM-2": SVC(
            probability=True,
            kernel="poly",
            degree=2,
            random_state=1
        ),
        "SVM-3": SVC(
            probability=True,
            kernel="poly",
            degree=3,
            random_state=1
        ),
        "SVM-4": SVC(
            probability=True,
            kernel="poly",
            degree=4,
            random_state=1
        ),
        "SVM-5": SVC(
            probability=True,
            kernel="poly",
            degree=5,
            random_state=1
        ),
        "Soft Voting": create_soft_voting_model()
    }
    return models

#Step 4 — Evaluate using repeated cross-validation

models = create_soft_voting_models()

results = []
names = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
)

results.append(scores)
names.append(name)

print(
    f"{name:12s} "
    f"Mean = {scores.mean():.3f} | "
    f"Std = {scores.std():.3f}"
)

#Step 5 — Visualise the results
plt.figure(figsize=(10, 5))

plt.boxplot(
    results,
    tick_labels=names,
    showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Soft Voting vs Individual SVM Models")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

##Activity 5 — Hard Voting Ensemble
#Hard voting uses the predicted class labels from each classifier and selects the class receiving
#the majority vote.



#Step 1 — Create the KNN voting ensemble
#We will use five KNN models:
#k = 1, 3, 5, 7, 9
#Odd values reduce the possibility of ties

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
        voting="hard"
    )
    return ensemble

#Step 2 — Create the models for comparison

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

#Step 3 — Evaluate the models

models = create_hard_voting_models()

results = []
names = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )

    results.append(scores)
    names.append(name)

    print(
        f"{name:12s} "
        f"Mean = {scores.mean():.3f} | "
        f"Std = {scores.std():.3f}"
    )

# Step 4 — Visualise

plt.figure(figsize=(10, 5))

plt.boxplot(

    results,
    tick_labels=names,
    showmeans=True
)

plt.ylabel("Accuracy")
plt.title("Hard Voting vs Individual KNN Models")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#Bagging
#Train multiple models independently using different bootstrap samples and aggregate their
#predictions.
#The main objective is generally to reduce variance. Ensemble methods (1)

#Boosting
#Train weak learners sequentially, allowing later learners to focus on difficult observations.
#The aim is to progressively build a stronger predictive model. Ensemble methods (1)

#Hard Voting
#Combine the predicted class labels and select the majority class.

#Soft Voting
#Combine class probabilities and select the class with the highest combined probability.