# ANN on Imbalanced Dataset

This project aims to build an Artificial Neural Network (ANN) to predict customer churn based on various features. The dataset used is imbalanced, which means that the number of instances for the majority class is significantly higher than the minority class. To address this issue, four different methods for handling imbalanced datasets were employed: Undersampling, Oversampling, Synthetic Minority Over-sampling Technique (SMOTE), and Ensemble with Undersampling. Each method has been explained below:

## 1. Undersampling

### Description
Undersampling reduces the number of instances in the majority class to match the number of instances in the minority class. This is done to balance the class distribution and mitigate the bias towards the majority class.

### Steps
- Separate the dataset into majority and minority classes.
- Randomly sample from the majority class to match the number of instances in the minority class.
- Concatenate the undersampled majority class with the minority class to form a balanced dataset.
- Split the data into training and testing sets.
- Train the ANN on the balanced dataset.

### Results
This method improved the recall and F1-score for the minority class but at the cost of losing valuable information from the majority class.

## 2. Oversampling

### Description
Oversampling increases the number of instances in the minority class by randomly replicating them until the class distribution is balanced.

### Steps
- Separate the dataset into majority and minority classes.
- Randomly replicate the minority class instances to match the number of instances in the majority class.
- Concatenate the majority class with the oversampled minority class to form a balanced dataset.
- Split the data into training and testing sets.
- Train the ANN on the balanced dataset.

### Results
This method also improved the recall and F1-score for the minority class, providing a better balance between precision and recall.

## 3. Synthetic Minority Over-sampling Technique (SMOTE)

### Description
SMOTE generates synthetic samples for the minority class by interpolating between existing minority class instances. This helps in creating more diverse synthetic instances and reduces overfitting.

### Steps
- Separate the dataset into features (X) and target (y).
- Apply SMOTE to generate synthetic samples for the minority class.
- Concatenate the original dataset with the synthetic samples to form a balanced dataset.
- Split the data into training and testing sets.
- Train the ANN on the balanced dataset.

### Results
SMOTE provided a significant improvement in the recall and F1-score for the minority class, indicating better performance in predicting customer churn.

## 4. Ensemble with Undersampling

### Description
This method involves creating multiple subsets of the majority class, each combined with the minority class, and training separate models on each subset. The final prediction is made by averaging the predictions of all models.

### Steps
- Separate the dataset into features (X) and target (y).
- Split the training data into multiple batches of majority class instances and combine each batch with the minority class.
- Train a separate ANN model on each batch.
- Aggregate the predictions of all models to make the final prediction.

### Results
This method improved the recall and F1-score for the minority class but required more computational resources due to the training of multiple models.

## Conclusion

Handling imbalanced datasets is crucial for building effective machine learning models. Each method has its advantages and trade-offs. The choice of method depends on the specific use case and the balance between precision, recall, and computational efficiency. In this project, SMOTE and Ensemble with Undersampling showed significant improvements in predicting customer churn, providing a more balanced performance across all metrics.

## Dependencies

- Python
- pandas
- numpy
- scikit-learn
- tensorflow
- imbalanced-learn
