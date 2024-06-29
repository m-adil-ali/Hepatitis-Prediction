import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import statsmodels as stats
import joblib

#@st.cache

st.sidebar.title("Navigation")
page= st.sidebar.radio("Go to", ["EDA and Model Training", "Prediction", "Hepatitis Awareness", "Contact us"])

if page == "EDA and Model Training":

    # Load the dataset
    st.title("Hepatitis Awareness and Prediction")
    st.image("data\hepatitis.jpg")

    st.write("If you want to know more about the Hepatitis dataset, here is the [link](https://www.kaggle.com/datasets/mragpavank/hepatitis-dataset)")
    file = "data/hepatitis.csv"

    if file is not None:
        df = pd.read_csv(file, na_values="?")
        st.write("### Data Preview", df.head())
        
        # Basic data statistics
        st.write("### Data Summary")
        st.write(df.describe())
        
        # Handling missing values
        st.write("### Missing Values")
        st.write(df.isnull().sum())

        # Data preprocessing
        st.write("### Data Preprocessing")

        # Handling missing values for categorical and numerical variables

        # Convert appropriate columns to numeric, errors='coerce' will handle any improper conversion
        numerical_cols = ['age', 'bili', 'alk', 'sgot', 'albu', 'protime']
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

        # Identify the actual categorical columns based on your metadata
        categorical_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 
                            'anorexia', 'liverBig', 'liverFirm', 'spleen', 'spiders',
                            'ascites', 'varices', 'histology']

        # Convert these categorical columns to type 'category'
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Handling missing values for numerical and categorical variables

        # Fill missing numerical values with the median of each column
        for col in numerical_cols:
            if df[col].isnull().any():  # Only fill if there are missing values
                df[col] = df[col].fillna(df[col].median())

        # Fill missing categorical values with the mode of each column
        for col in categorical_cols:
            if df[col].isnull().any():  # Only fill if there are missing values
                mode_value = df[col].mode()
                if not mode_value.empty:  # Check if mode exists
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    # If mode is not found (e.g., all values are NaN), fill with a placeholder
                    df[col] = df[col].fillna('Unknown')

        # After handling missing values, confirm the missing values are handled
        st.write("### Missing Values After Treatment")
        st.write(df.isnull().sum())

        
        # Feature Engineering - Selecting top features based on correlation
        corr = df.corr()
        top_features = corr['target'].abs().sort_values(ascending=False).index[1:7]
        st.write("### Top Features", top_features)
        
        # Data Visualization
        st.write("### Exploratory Data Analysis")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, feature in enumerate(top_features):
            sns.histplot(df[feature], kde=True, ax=axes[i//3, i%3])
            axes[i//3, i%3].set_title(f"Distribution of {feature}")
        st.pyplot(fig)

        ### Plotly
    

        # Ensure data types are correct
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Animated Plotly Bar Plot
        # Let's visualize the count of patients by 'Bilirubin' across 'Age' with animation
        st.write("### Animated Plotly Scatter Graph - Bilirubin Levels Across Age")

        # Remove rows with missing 'bili' or 'age' for this visualization
        df_plot = df.dropna(subset=['age', 'bili'])

        # Create a scatter plot
        fig = px.scatter(df_plot, x='age', y='bili', trendline = 'ols',
                        title='Bilirubin Levels Across Age',
                        labels={'Bilirubin': 'Bilirubin Level', 'age': 'Age'})
        st.plotly_chart(fig)



    ### ______________________________________    
        # Train-test split
        X = df.drop(['ID','target'], axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #st.write("test", X.head())
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(scaler, 'scaler.pkl')
        
        # Model Training Using 3 for now
        st.write("### Model Training and Evaluation")
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC()
        }
        
        # Evaluating models and then comparing their metrics

        model_performance = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            model_performance[name] = [accuracy, precision, recall, f1]
        
        performance_df = pd.DataFrame(model_performance, index=["Accuracy", "Precision", "Recall", "F1 Score"])
        st.write(performance_df)
        st.write("### Model Performance Comparison")
        
        performance_df_long = performance_df.reset_index().melt(id_vars='index', var_name='Model', value_name='Score')
        
        fig = px.bar(performance_df_long, x='Model', y='Score', color='index', barmode='group', title="Model Performance Comparison")
        
        st.plotly_chart(fig)
        
        # Save the model as a pickle file
        joblib.dump(models["Logistic Regression"], 'logistic_regression_model.pkl')
        
elif page == "Prediction":
    st.title("Hepatitis Prediction")
    
    # Taking input from user and then using that input for prediction
    st.write("### Prediction Inputs")
    
    # Load the model from the file
    model = joblib.load('logistic_regression_model.pkl')
    
    # Load the StandardScaler from the file
    scaler = joblib.load('scaler.pkl')

    with st.form(key='my_form'):
        age = st.number_input(label='Enter age', step=1 , min_value=18, max_value=100, value=30)
        gender = st.selectbox('Select gender (1=male, 2= female)', options=[1, 2])
        steroid = st.selectbox('Steroid use (1= yes, 2= no)', options=[1, 2])
        antivirals = st.selectbox('Antivirals use (1= yes, 2= no)', options=[1, 2])
        fatigue = st.selectbox('Fatigue (1= yes, 2= no)', options=[1, 2])
        malaise = st.selectbox('Malaise (1= yes, 2= no)', options=[1, 2])
        anorexia = st.selectbox('Anorexia (1= yes, 2= no)', options=[1, 2])
        liverBig = st.selectbox('Liver big (1= yes, 2= no)', options=[1, 2])
        liverFirm = st.selectbox('Liver firm (1= yes, 2= no)', options=[1, 2])
        spleen = st.selectbox('Spleen palpable (1= yes, 2= no)', options=[1, 2])
        spiders = st.selectbox('Spiders (1= yes, 2= no)', options=[1, 2])
        ascites = st.selectbox('Ascites (1= yes, 2= no)', options=[1, 2])
        varices = st.selectbox('Varices (1= yes, 2= no)', options=[1, 2])
        bili = st.number_input(label='Enter bilirubin level')
        alk = st.number_input(label='Enter alk level', step=1)
        sgot = st.number_input(label='Enter sgot level', step=1)
        albu = st.number_input(label='Enter albu level')
        protime = st.number_input(label='Enter protime level', step=1)
        histology = st.selectbox('Histology (1= yes, 2= no)', options=[1, 2])
        
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        user_input = [age, gender, steroid, antivirals, fatigue, malaise, anorexia, liverBig, 
                    liverFirm, spleen, spiders, ascites, varices, bili, alk, sgot, albu, 
                    protime, histology]
        
        #st.write("Before reshaping: ", user_input)
        
        # Convert the list to a 2D array
        user_input = np.array(user_input).reshape(-1) 
        
        #st.write("After reshaping to 1D: ", user_input.shape, user_input)
        
        # Reshape the 1D array to a 2D array with one row
        user_input = user_input.reshape(1, -1)
        
        #st.write("After reshaping to 2D: ", user_input.shape , user_input)   
       
        # scale the input array
        full_input_scaled = scaler.transform(user_input)
        
        # Predict with the 
        #st.write("Input to Logistic Regression: ", user_input.shape)
        
        prediction = model.predict(full_input_scaled)
        st.write('---')
        st.write('### Predictions based on your given input data:')
        if prediction == 2:
            st.write("You are not likely to have hepatitis and you will survive.")
        else:
            st.write("You are likely to to have hepatitis and you may not survive as per this model.")
        
        prediction_proba = model.predict_proba(full_input_scaled)
        st.subheader('Prediction Probability')
        st.write('1 means likelihood to survive and 0 means likelihood to not survive')
        st.write(prediction_proba)
        

    
elif page == "Hepatitis Awareness":
    st.title("Hepatitis Awareness")
    st.write(
"""
    ### What is Hepatitis?
    Hepatitis refers to an inflammatory condition of the liver. It’s commonly caused by a viral infection, but there are other possible causes of hepatitis. These include autoimmune hepatitis and hepatitis that occurs as a secondary result of medications, drugs, toxins, and alcohol.

    Autoimmune hepatitis is a disease that occurs when your body makes antibodies against your liver tissue.

    Your liver is located in the right upper area of your abdomen. It performs many critical functions that affect metabolism throughout your body, including:

    - Bile production, which is essential to digestion
    - Filtering of toxins from your body
    - Excretion of bilirubin (a product of broken-down red blood cells), cholesterol, hormones, and drugs
    - Metabolism of carbohydrates, fats, and proteins
    - Activation of enzymes, which are specialized proteins essential to body functions
    - Storage of glycogen (a form of sugar), minerals, and vitamins (A, D, E, and K)
    - Synthesis of blood proteins, such as albumin
    - Synthesis of clotting factors

    When the liver is inflamed or damaged, its functions can be affected.

    ### Types of Hepatitis
    - **Hepatitis A:** Hepatitis A is caused by consuming contaminated food or water. This type is often acute and resolves without treatment.
    - **Hepatitis B:** Hepatitis B is spread through contact with infectious body fluids. It can be both acute and chronic.
    - **Hepatitis C:** Hepatitis C is transmitted through direct contact with infected body fluids. This type is typically chronic.
    - **Hepatitis D:** Hepatitis D is a secondary infection that only occurs in people infected with Hepatitis B.
    - **Hepatitis E:** Hepatitis E is typically transmitted through consuming contaminated water. It is usually acute and resolves on its own.

    ### Prevention
    - Vaccinations are available for Hepatitis A and B.
    - Practice good hygiene.
    - Avoid sharing needles or other personal items.
    - Use condoms during sexual intercourse.
    - Avoid consumption of contaminated food or water.
    """
    )


elif page == "Contact us":
    ("""
     ### Get in Touch
    We would love to hear from you! You can reach us through the following channels:

    - **Website:** [www.hepatitis-awareness.org](https://www.hepatitis-awareness.org)
    - **Twitter:** [@HepatitisAware](https://twitter.com/HepatitisAware)
    - **Email:** contact@hepatitis-awareness.org

    Whether you have a question about our resources, need support, or just want to share your thoughts, feel free to reach out!
     """)