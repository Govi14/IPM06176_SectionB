# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("student-mat.csv", sep=";")

print(df.head())
print(df.info())

# ==============================
# 3. Exploratory Data Analysis
# ==============================

# Summary statistics
print(df.describe())

# Study time vs final grade
plt.figure()
sns.boxplot(x=df["studytime"], y=df["G3"])
plt.title("Study Time vs Final Grade")
plt.show()

# Absences vs final grade
plt.figure()
sns.scatterplot(x=df["absences"], y=df["G3"])
plt.title("Absences vs Final Grade")
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

# ==============================
# 4. Feature Preparation
# ==============================

# Create Pass/Fail column
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Encode categorical variables
df_encoded = df.copy()

label = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = label.fit_transform(df_encoded[col])

# ==============================
# 5. Clustering (Learning Profiles)
# ==============================

cluster_features = df_encoded[[
    "studytime",
    "absences",
    "failures",
    "G1",
    "G2"
]]

# Scale data
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(cluster_scaled)

print(df["cluster"].value_counts())

# ==============================
# 6. Label Clusters
# ==============================

cluster_profile = df.groupby("cluster")[[
    "studytime","absences","failures","G3"
]].mean()

print(cluster_profile)

# Example interpretation
cluster_labels = {
    0: "Consistent Learners",
    1: "Irregular Learners",
    2: "Struggling Students"
}

df["learning_profile"] = df["cluster"].map(cluster_labels)

print(df[["cluster","learning_profile"]].head())

# ==============================
# 7. Supervised Learning
# ==============================

X = df_encoded.drop(["G3","pass"], axis=1)
y = df_encoded["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 8. Logistic Regression
# ==============================

log_model = LogisticRegression(max_iter=2000)

log_model.fit(X_train, y_train)

pred_log = log_model.predict(X_test)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, pred_log))

print(classification_report(y_test, pred_log))

# Cross Validation
log_cv = cross_val_score(log_model, X, y, cv=5)

print("Logistic Regression CV Accuracy:", log_cv.mean())

# ==============================
# 9. Random Forest
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:")
print(accuracy_score(y_test, pred_rf))

print(classification_report(y_test, pred_rf))

# Cross Validation
rf_cv = cross_val_score(rf_model, X, y, cv=5)

print("Random Forest CV Accuracy:", rf_cv.mean())

# ==============================
# 10. Feature Importance
# ==============================

importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure()
importances.head(10).plot(kind='bar')
plt.title("Top Features Affecting Student Performance")
plt.show()

# ==============================
# Model Comparison Table
# ==============================

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logistic Regression metrics
log_accuracy = accuracy_score(y_test, pred_log)
log_precision = precision_score(y_test, pred_log)
log_recall = recall_score(y_test, pred_log)
log_f1 = f1_score(y_test, pred_log)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, pred_rf)
rf_precision = precision_score(y_test, pred_rf)
rf_recall = recall_score(y_test, pred_rf)
rf_f1 = f1_score(y_test, pred_rf)

# Create comparison table
comparison_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [log_accuracy, rf_accuracy],
    "Precision": [log_precision, rf_precision],
    "Recall": [log_recall, rf_recall],
    "F1 Score": [log_f1, rf_f1],
    "Cross Validation Score": [log_cv.mean(), rf_cv.mean()]
})

print(comparison_table)
