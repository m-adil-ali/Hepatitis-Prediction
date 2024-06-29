# Hepatitis Prediction Project.
[Streamlit live App link](https://hepatitis-prediction-webapp.streamlit.app/)

### 📊 Overview
- This project focuses on predicting Hepatitis cases using Supervised Machine Learning techniques. We’ll walk through the steps from data preparation to model deployment.

# Steps Taken
1. Data Acquisition\
    Obtained the dataset from the [Kaggle](https://www.kaggle.com/datasets/mragpavank/hepatitis-dataset).\
    Explored the metadata to understand its structure and features.
2. Data Cleaning\
    Cleaned the dataset by handling missing values, duplicates, and outliers.\
    Ensured data consistency and integrity.
3. Exploratory Data Analysis (EDA)\
    Conducted EDA to gain insights into the data.\
    Visualized distributions, correlations, and patterns.
4. Preprocessing\
    Imputed missing values using appropriate techniques (mean, median, etc.).\
    Checked for duplicate records.\
    Prepared the data for model training.
5. Model Selection\
    Split the dataset into training and testing sets.\
    Normalized features to ensure consistent scaling.\
    Experimented with various machine learning models:
     - Random Forest Classifier
     - Logistic Regression
     - Support Vector Machine
6. Model Evaluation\
    Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.\
    Selected the best-performing model based on validation results.
7. Model Deployment\
    Saved the best model as a .pkl file for future use.\
    Created a .py file for user interface using Streamlit and made a multi-page web app to allow real-time predictions based on user input.

# Usage

Clone this repository and install the required dependencies.\
Load the pre-trained model using the .pkl file.\
Use the .py file to make a webpage to predict Hepatitis cases based on real-time user input.\
Feel free to customize this template with specific details about your dataset, features, and findings.\
Good luck with your project! �
