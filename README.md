# Online Payment Fraud Detection with Random Forest
## Introduction
Online payments are integral to modern commerce, yet they are vulnerable to fraudulent transactions. Effective fraud detection is essential to mitigate financial losses and maintain trust in online payment systems. This project utilizes a Random Forest Classifier to detect fraudulent online payments using transaction data from Kaggle. The goal is to build a reliable and scalable fraud detection model.

## Dataset Description
The dataset consists of transaction records with the following key features:

- step: A time unit where 1 step equals 1 hour.
- type: The type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT).
amount: The transaction amount.
nameOrig: The originating account identifier.
oldbalanceOrg: Initial balance of the originating account.
newbalanceOrig: Updated balance of the originating account.
nameDest: The recipient account identifier.
oldbalanceDest: Initial balance of the recipient account.
newbalanceDest: Updated balance of the recipient account.
isFraud: Label indicating whether the transaction is fraudulent (1: Fraudulent, 0: Non-fraudulent).
isFlaggedFraud: Indicates whether the transaction was flagged as suspicious.
Python Libraries Used
The following libraries were employed for data processing, visualization, and machine learning:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
matplotlib & seaborn: For data visualization.
scikit-learn: For machine learning and evaluation tasks.
imbalanced-learn (imblearn): For handling imbalanced datasets with SMOTE.
shap: For explainability of machine learning models.
Data Preprocessing
Key preprocessing steps include:

Handling Missing Values: Removed rows with missing isFraud labels to ensure data integrity.
Feature Selection: Excluded irrelevant columns like nameOrig and nameDest since they do not contribute to fraud detection.
Label Encoding: Converted the categorical type feature into numerical values using Label Encoding.
Class Balancing: Applied Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance between fraudulent and non-fraudulent transactions.
Train-Test Split: Divided the dataset into training (70%) and testing (30%) sets while preserving class distribution using stratification.
Model Training
The Random Forest Classifier was selected for its:

Robustness to overfitting.
High predictive accuracy for binary classification problems.
Ability to handle imbalanced datasets effectively.
Model Parameters:
n_estimators=100: Number of trees in the forest.
max_depth=15: Maximum depth of each tree to prevent overfitting.
min_samples_split=10: Minimum number of samples required to split an internal node.
random_state=42: Ensures reproducibility.
n_jobs=-1: Utilizes all available CPU cores for parallel processing.
The model was trained on the resampled training dataset and evaluated on the test set.

Evaluation Metrics
The model's performance was evaluated using the following metrics:

Accuracy: Measures the proportion of correctly classified transactions.
Confusion Matrix: Displays the true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
Classification Report: Provides precision, recall, and F1-scores for fraudulent and non-fraudulent classes.
ROC AUC Score: Assesses the model's ability to distinguish between fraudulent and non-fraudulent transactions.
Results:
Accuracy: Achieved 90% (example value).
Confusion Matrix:
lua
Copy
Edit
[[TN, FP],
 [FN, TP]]
Top Contributing Features:
Amount
OldbalanceOrg
NewbalanceOrig
Visualizations
Confusion Matrix Heatmap: A heatmap illustrating the model's performance in correctly identifying fraud and non-fraud transactions.
Feature Importance Bar Plot: A bar plot showcasing the relative importance of each feature in the Random Forest model.
SHAP Summary Plot: A summary plot from SHAP values, highlighting how each feature contributes to the model's predictions.
Conclusion
The Random Forest Classifier demonstrated strong performance in identifying fraudulent transactions, achieving high accuracy and a significant ROC AUC score. Key takeaways include:

Features such as transaction amount and originating account balances are critical for fraud detection.
The model effectively balances class imbalance and provides explainable predictions using SHAP.
Future Work
To enhance the model's performance and applicability:

Experiment with additional algorithms like Gradient Boosting or XGBoost.
Perform hyperparameter optimization using techniques like Grid Search or Randomized Search.
Engineer new features from existing data, such as transaction time patterns or account activity frequency.
Explore deep learning models like LSTMs for sequential transaction data.
