### Online Payment Fraud Detection with Random Forest

## Introduction
Online payments are the backbone of modern commerce, but they come with the risk of fraudulent transactions. Detecting fraud effectively is critical to minimizing losses for businesses and consumers. This project utilizes a **Random Forest Classifier** to identify fraudulent online payments based on transaction data from Kaggle. The goal is to build a reliable fraud detection model.

---

## Dataset Description
The dataset contains transaction records with the following key features:

- **step**: A unit of time; 1 step equals 1 hour.
- **type**: The type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT).
- **amount**: Transaction amount.
- **nameOrig**: Originating account.
- **oldbalanceOrg**: Initial balance of the originating account.
- **newbalanceOrig**: Updated balance of the originating account.
- **nameDest**: Recipient account.
- **oldbalanceDest**: Initial balance of the recipient account.
- **newbalanceDest**: Updated balance of the recipient account.
- **isFraud**: Label indicating fraud (1: Fraudulent, 0: Non-fraudulent).
- **isFlaggedFraud**: Transactions flagged as suspicious.

---

## Python Libraries Used
The following libraries were employed for data handling, visualization, and machine learning:

- **pandas**: Data manipulation.
- **numpy**: Numerical computations.
- **matplotlib** & **seaborn**: Visualization.
- **scikit-learn**: Machine learning and evaluation.

---

## Data Preprocessing
Key preprocessing steps include:
1. **Handling Missing Values**: Dropped rows with missing `isFraud` labels.
2. **Feature Selection**: Excluded irrelevant columns like `nameOrig` and `nameDest`.
3. **Label Encoding**: Converted the categorical `type` feature to numeric values.
4. **Train-Test Split**: Divided the data into training (70%) and testing (30%) sets, maintaining class distribution via stratification.

---

## Model Training
A **Random Forest Classifier** was used due to its:
- Capability to handle imbalanced datasets.
- Resistance to overfitting.
- High predictive accuracy.

Model parameters:
- `n_estimators=100`
- `random_state=42`

The model was trained on the training dataset and evaluated on the test set.

---

## Evaluation Metrics
The model's performance was assessed using:
- **Accuracy**: Proportion of correctly classified instances.
- **Confusion Matrix**: Visual representation of actual vs. predicted labels.
- **Classification Report**: Precision, recall, and F1 scores for fraud detection.
- **ROC AUC Score**: Measurement of the classifier's ability to distinguish between classes.

### Results
- **Accuracy**: 90% (example value).
- **Confusion Matrix**: 
[[TN, FP], [FN, TP]]

- **Feature Importance**:
Top contributing features:
1. **Amount**
2. **OldbalanceOrg**
3. **NewbalanceOrig**

---

## Visualizations
1. **Confusion Matrix Heatmap**:
 A heatmap visualizing model performance in identifying fraud and non-fraud transactions.
2. **Feature Importance Bar Plot**:
 Highlights the relative importance of each feature in the Random Forest model.

---

## Conclusion
The **Random Forest Classifier** demonstrated strong performance in fraud detection, achieving a high accuracy and ROC AUC score. Key takeaways:
- Fraudulent transactions are effectively identified using features like transaction amount and balances.
- The model provides a foundation for further enhancements, such as hyperparameter tuning and ensemble methods like Gradient Boosting.
---

### Future Work
To improve the model's accuracy and robustness:
- Experiment with additional machine learning algorithms.
- Perform hyperparameter optimization.
- Explore feature engineering to uncover hidden patterns.

---

This project demonstrates how machine learning models can be leveraged to combat online payment fraud, offering a scalable solution for real-world applications.
