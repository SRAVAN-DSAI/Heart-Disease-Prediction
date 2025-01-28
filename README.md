## Heart-Disease-Prediction
This project aims to predict the presence of heart disease based on various medical features using machine learning models. The dataset contains medical attributes such as age, sex, cholesterol levels, blood pressure, and more, which are used to determine whether a patient has heart disease.

Project Overview
1. Data Collection and Preprocessing
The dataset is sourced from the UCI Machine Learning Repository. It contains 14 attributes, including medical features like age, cholesterol, and resting blood pressure.
Any missing values represented as "?" are handled by replacing them with the most frequent value (mode) of the respective columns.
Categorical features like ca (number of major vessels colored by fluoroscopy) and thal (thalassemia status) are converted into numeric values using label encoding.
2. Outlier Detection and Removal
Outliers in the dataset are detected using the Z-score method. Data points with a Z-score greater than 3 are considered outliers and are removed.
3. Model Building
Logistic Regression and Decision Tree Classifier models are used to predict the presence of heart disease.
These models are trained on the data to classify whether a patient has heart disease (1) or not (0).
4. Model Evaluation
Both models are evaluated using classification metrics, including:
Precision: The accuracy of positive predictions.
Recall: The ability of the model to capture all the positive instances.
F1-Score: The balance between precision and recall.
Accuracy: The overall correctness of the model.
Data Description
The dataset used in this project contains the following features:

age: Age in years

sex: Sex (1 = male, 0 = female)

cp: Chest pain type

trestbps: Resting blood pressure

chol: Serum cholesterol

fbs: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)

restecg: Resting electrocardiographic results

thalach: Maximum heart rate achieved

exang: Exercise induced angina (1 = yes, 0 = no)

oldpeak: Depression induced by exercise relative to rest

slope: Slope of the peak exercise ST segment

ca: Number of major vessels colored by fluoroscopy

thal: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)

target: Presence or absence of heart disease (1 = presence, 0 = absence)

Steps to Reproduce
1. Install Dependencies
To run the project, first install the required Python libraries.
2. Download the Data
The dataset is automatically downloaded from the UCI Machine Learning Repository.

3. Preprocess the Data
Handle missing values by replacing "?" with the mode of the respective columns.
Label encode categorical columns like ca and thal.
4. Outlier Detection and Removal
Use the Z-score method to identify and remove outliers (Z-score > 3).

5. Train the Model
Split the dataset into training and test sets, and then train the machine learning models (Logistic Regression and Decision Tree) using the training data.

6. Evaluate the Model
Use classification metrics to evaluate the model's performance, including precision, recall, F1-score, and accuracy.

Results
After training and evaluation, the models' performance is measured using the classification report, which provides key metrics for model comparison. Both Logistic Regression and Decision Tree models are assessed, showing the following:

Precision: How many of the predicted positives are actually positive.
Recall: How many of the actual positives were correctly predicted.
F1-Score: The harmonic mean of precision and recall, balancing the two metrics.
Accuracy: The overall correctness of the model in predicting both positive and negative cases.
