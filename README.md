# Heart Disease Prediction

This repository contains Python code for building and evaluating machine learning models to predict heart disease using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

**Data:**

- The dataset includes features such as age, sex, blood pressure, cholesterol, and other medical indicators.
- The target variable is binary, indicating the presence or absence of heart disease.

**Methodology:**

1. **Data Loading and Preparation:**
   - Load the dataset from the UCI repository.
   - Handle missing values.
   - Prepare the data for visualization and modeling.

2. **Exploratory Data Analysis (EDA):**
   - Visualize the distribution of heart disease cases.
   - Analyze the distribution of numerical and categorical features.
   - Explore the correlation between features.

3. **Model Training:**
   - Split the data into training and testing sets.
   - Preprocess the data using appropriate techniques (e.g., scaling, one-hot encoding).
   - Train two machine learning models:
     - Logistic Regression
     - Decision Tree Classifier

4. **Model Evaluation:**
   - Evaluate the performance of each model on the test set using metrics such as:
     - Confusion Matrix
     - ROC Curve
     - Classification Report
     - ROC AUC

**Output:**

- The script generates various visualizations (e.g., histograms, bar plots, heatmaps) to understand the data.
- Model evaluation metrics (confusion matrices, ROC curves, classification reports) are saved as images and text files.

**To run the code:**

1. Make sure you have the following libraries installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - scipy

2. Run the Python script.

**Note:**

- This is a basic example and can be further improved by:
   - Hyperparameter tuning
   - Feature engineering
   - Trying different machine learning models (e.g., Random Forest, Support Vector Machines)
   - Implementing more sophisticated data preprocessing techniques
