# ------------------------ CUSTOMER INTELLIGENCE & REVENUE OPTIMIZATION SYSTEM ------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, classification_report

# VISUAL STYLE 
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9,6)

# LOAD DATA
df = pd.read_csv("D:\\Mall_Customers.csv")

print(df.columns)  

# FEATURE SELECTION 
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# DATA SCALING 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  PCA 
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Variance Retained (%):", round(pca.explained_variance_ratio_.sum()*100, 2))

# CUSTOMER SEGMENTATION 
sil_scores = []
K = range(2, 11)

for k in K:
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, labels))

optimal_k = K[np.argmax(sil_scores)]

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Segment"] = kmeans.fit_predict(X_pca)

print("Silhouette Score:", round(silhouette_score(X_pca, df["Segment"]), 3))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X_pca, df["Segment"]), 3))

# REGRESSION: SPENDING PREDICTION 
y_reg = df['Spending Score (1-100)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

print("Regression R2 Score:", round(r2_score(y_test_reg, y_pred_reg), 3))
print("Regression RMSE:", round(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)), 2))

#  CLASSIFICATION: CHURN RISK
df["Churn"] = (df['Spending Score (1-100)'] < 40).astype(int)

y_cls = df["Churn"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_scaled, y_cls, test_size=0.2, random_state=42
)

cls_model = LogisticRegression(max_iter=1000)
cls_model.fit(X_train_cls, y_train_cls)

y_pred_cls = cls_model.predict(X_test_cls)

print("\nChurn Classification Report:\n")
print(classification_report(y_test_cls, y_pred_cls))

#SEGMENT PROFILING 
segment_profile = df.groupby("Segment")[features].mean()
print("\nSegment Profile:\n")
print(segment_profile)

#------------------VISUALIZATIONS ------------------

# 1. CUSTOMER SEGMENTATION MAP (PCA)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1],
                hue=df["Segment"], palette="Set2", s=120)
plt.title("1. Customer Segmentation Map (PCA)")
plt.xlabel("Behavior Component 1")
plt.ylabel("Behavior Component 2")
plt.show()

# 2. CUSTOMER VALUE MATRIX (INCOME vs SPENDING)
sns.scatterplot(data=df,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                hue="Segment", palette="Set2", s=120)
plt.title("2. Customer Value Matrix (Income vs Spending)")
plt.show()

# 3. CUSTOMER SEGMENT DISTRIBUTION
sns.countplot(x="Segment", hue="Segment",
              data=df, palette="Set2", legend=False)
plt.title("3. Customer Distribution by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Customer Count")
plt.show()

# 4. REGRESSION PERFORMANCE
plt.scatter(y_test_reg, y_pred_reg, color="green")
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.title("4. Spending Prediction Performance (Regression)")
plt.show()

# 5. CUSTOMER CHURN DISTRIBUTION
sns.countplot(x="Churn", hue="Churn",
              data=df, palette="coolwarm", legend=False)
plt.title("5. Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Customer Count")
plt.show()

# 6. CUSTOMER SEGMENT PROFILING HEATMAP
sns.heatmap(segment_profile, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("6. Customer Segment Behavior Heatmap")
plt.show()
