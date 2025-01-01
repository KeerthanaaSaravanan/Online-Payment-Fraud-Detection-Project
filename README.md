# Online-Payment-Fraud-Detection-Project
### Online Payment Fraud Detection with Random Forest
---
#### **Introduction**
Online payment is the most popular transaction method in the world today. However, with an increase in online payments also comes a rise in payment fraud. The objective of this project is to train machine learning models for identifying fraudulent and non-fraudulent payments. The dataset, sourced from Kaggle, contains historical information about fraudulent transactions, which can be leveraged to detect fraud in online payments effectively.

---

#### **Dataset Description**
The dataset consists of 10 key variables:

- **step**: Represents a unit of time where 1 step equals 1 hour.
- **type**: Type of online transaction.
- **amount**: The amount of the transaction.
- **nameOrig**: Customer initiating the transaction.
- **oldbalanceOrg**: Balance before the transaction.
- **newbalanceOrig**: Balance after the transaction.
- **nameDest**: Recipient of the transaction.
- **oldbalanceDest**: Initial balance of the recipient before the transaction.
- **newbalanceDest**: The new balance of the recipient after the transaction.
- **isFraud**: Indicates if the transaction is fraudulent (1) or not (0).

---

#### **Python Libraries**
The following Python libraries were utilized in this project:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **seaborn & matplotlib**: For data visualization.
- **sklearn**: For implementing the Random Forest model and evaluating performance.

---

#### **Machine Learning Model**
A **Random Forest Classifier** was used to identify fraudulent and non-fraudulent payments. The model was selected for its ability to handle large datasets, deal with overfitting, and provide better performance by combining multiple decision trees.

---

#### **Steps in the Project**
1. **Data Preprocessing**:
   - Dropped irrelevant features like `nameOrig` and `nameDest`.
   - Encoded the categorical variable `type`.
   - Split the data into training and testing datasets.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of transaction types.
   - Examined the correlation between numerical features using a heatmap.

3. **Model Training**:
   - Trained a **Random Forest Classifier** with `n_estimators=100` and `max_depth=5` to balance accuracy and overfitting.

4. **Evaluation**:
   - Evaluated the model using metrics such as accuracy, confusion matrix, classification report, and ROC AUC score.
   - Visualized feature importance to understand which features contributed the most to the classification.

---

#### **Conclusion**
The **Random Forest Classifier** achieved improved results in detecting online payment fraud compared to simpler models like Decision Trees. 

- **Evaluation Metrics**:
  - Accuracy: ~90% (example value; actual performance depends on data).
  - The feature importance analysis showed which variables have the most impact on fraud detection.

While the Random Forest model performs well, further improvements can be made using hyperparameter tuning or by exploring ensemble methods like Gradient Boosting for even better performance.

---

#### **Visualization**
The project includes visualizations for:
1. Transaction type distribution.
2. Correlation heatmap of numerical features.
3. Confusion matrix of the Random Forest model.
4. Feature importance bar plot to interpret which variables contribute the most.

---
