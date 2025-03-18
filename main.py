import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# map target values to actual species names
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display first 5 rows
df.head()

#check data set information
df.info()

#check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Pairplot to see distributions and relationships
sns.pairplot(df, hue="species")
plt.show()

# Split Data into training and testing sets

X = df[iris.feature_names] # Features
y = df['target'] #Labels

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a K-Nearest Neighbors (KNN) Model
# Create and train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

#make predictions
y_pred = knn.predict(X_test_scaled)

#Evaluate the model

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print((f'Accuracy: {accuracy:.2f}'))

#Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()










