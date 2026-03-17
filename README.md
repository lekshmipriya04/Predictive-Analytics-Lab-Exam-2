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
- **Missing Values**: Missing values in the 'Target' column were identified and handled by dropping the corresponding rows, ensuring data integrity for the target variable.
- **Target Encoding**: The 'Target' column was encoded from categorical ('Yes', 'No') to numerical (1, 0) for model compatibility, as most machine learning algorithms require numerical input.
- **Outlier Handling**: An outlier in 'Feature1' with an exceptionally large value (identified as `10000.0` at index `132`) was detected and removed from the dataset. This step is crucial to prevent the model from being unduly influenced by extreme values, which could skew results and impair performance.

### 3. Exploratory Data Analysis (EDA)
A scatter plot was generated to visualize the relationship between 'Feature1' and 'Feature2', with points colored according to the 'Target' variable. This visualization helped in understanding the separability of the classes and identifying any obvious patterns or challenges for classification.

### 4. Data Splitting
The cleaned and prepared dataset was split into training and testing sets using `train_test_split` with a 70% to 30% ratio, respectively, and `random_state=42` for reproducibility. This ensures that the model is trained on one part of the data and evaluated on unseen data to assess its generalization capabilities.

### 5. Model Selection and Training
- **Model Choice**: A Decision Tree Classifier (`DecisionTreeClassifier`) was chosen for this binary classification task. Decision Trees are intuitive, easy to interpret, and capable of capturing non-linear relationships within the data. Their interpretability makes it easier to understand how decisions are made based on feature values. They are also relatively robust to outliers after the initial cleaning steps, and can handle both numerical and categorical data effectively without extensive feature scaling.
- **Training**: The `DecisionTreeClassifier` was initialized with `random_state=42` for reproducibility and trained using the `X_train` (features) and `y_train` (target) data.

### 6. Model Evaluation
- **Prediction**: The trained model was used to make predictions (`dt_y_pred`) on the `X_test` dataset.
- **Performance Metrics**: The model's performance was rigorously evaluated using:
    - **Accuracy Score**: The overall proportion of correctly classified instances. The Decision Tree model achieved an accuracy of **0.955** on the test set.
    - **F1 Score**: The f1 score is calculated and achieved **0.901** on the test set.
    - **Confusion Matrix**: The confusion matrix is another performance evaluation tool used in this model. It showed how well the model predicts each class by comparing actual vs predicted values.
    This shows strong performance and good accuracy suggesting the model is effective at identifying instances of the positive class.

### 7. Decision Boundary Visualization
A custom `plot_decision_boundary` function was utilized to visually represent the decision regions created by the trained Decision Tree model on the training data. This plot provides a clear graphical insight into how the model segments the feature space to classify instances, making the model's logic more transparent.

## How to Run
To reproduce this analysis, execute the cells in the provided Jupyter notebook sequentially. Ensure all required Python libraries (pandas, scikit-learn, matplotlib, seaborn, numpy) are installed in your environment.
