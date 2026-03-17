```markdown
# Binary Classification Project: Feature Analysis and Decision Tree Modeling

## Project Overview
This project focuses on performing binary classification using a given dataset. The process involves loading the data, performing initial exploration and preprocessing, visualizing key features, and finally training and evaluating a Decision Tree Classifier.

## Dataset
The dataset `Lab_Exam_binary_classification_dataset.csv` contains three columns:
- `Feature1`: A numerical feature.
- `Feature2`: Another numerical feature.
- `Target`: The binary target variable ('Yes' or 'No').

## Steps Performed

### 1. Data Loading and Initial Inspection
The dataset was loaded into a Pandas DataFrame. The first 5 rows, basic information (`df.info()`), and descriptive statistics (`df.describe()`) were displayed to understand the data's structure, types, and summary statistics.

### 2. Data Preprocessing
- **Missing Values**: Missing values in the 'Target' column were identified and handled by dropping the corresponding rows.
- **Target Encoding**: The 'Target' column was encoded from categorical ('Yes', 'No') to numerical (1, 0) for model compatibility.
- **Outlier Handling**: An outlier in 'Feature1' with an exceptionally large value was identified and removed from the dataset to prevent skewed model training.

### 3. Exploratory Data Analysis (EDA)
A scatter plot was generated to visualize the relationship between 'Feature1' and 'Feature2', colored by the 'Target' variable. This visualization helps in understanding the separability of the classes based on the features.

### 4. Data Splitting
The dataset was split into training and testing sets (70% training, 30% testing) to evaluate the model's performance on unseen data. `X` contained 'Feature1' and 'Feature2', and `y` contained the 'Target' variable.

### 5. Model Training and Evaluation
- **Model**: A Decision Tree Classifier (`DecisionTreeClassifier`) was chosen for this binary classification task.
- **Training**: The model was trained using the training data (`X_train`, `y_train`).
- **Prediction**: Predictions were made on the test set (`X_test`).
- **Evaluation**: The model's performance was evaluated using:
    - **Accuracy Score**: A measure of overall correct predictions.
    - **Classification Report**: Provides precision, recall, f1-score, and support for each class.

### 6. Decision Boundary Visualization
A custom function `plot_decision_boundary` was used to visualize the decision regions learned by the Decision Tree model on the training data, providing insight into how the model separates the classes.

## Results
The Decision Tree Classifier achieved an accuracy of **0.943** on the test set. The classification report showed strong performance, especially for class 0 (precision 1.00, recall 0.93), and reasonable performance for class 1 (precision 0.79, recall 0.98), indicating that the model generalizes well to new data after the preprocessing steps.

## How to Run
To reproduce this analysis, execute the cells in the provided Jupyter notebook sequentially. Ensure all required libraries (pandas, scikit-learn, matplotlib, seaborn, numpy) are installed.
```
