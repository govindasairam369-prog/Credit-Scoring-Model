import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==========================================
# STEP 1: Load Dataset
# ==========================================
df = pd.read_csv("data/credit_risk_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 2: Separate Features and Target
# ==========================================
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ==========================================
# STEP 3: Define Columns
# ==========================================
numeric_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

# ==========================================
# STEP 4: Preprocessing
# ==========================================

# Fill missing numeric values and scale them
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Fill missing categorical values and encode them
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# ==========================================
# STEP 5: Split Dataset
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==========================================
# STEP 6: Define Models
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(
        solver="saga",
        max_iter=5000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

# ==========================================
# STEP 7: Train and Evaluate
# ==========================================
for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nProject completed successfully!")