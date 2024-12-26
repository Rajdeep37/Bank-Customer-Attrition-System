import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Title and description
st.title("Churn Prediction Model Comparison")
st.markdown("""
This application allows you to upload a customer data CSV file, 
compare various machine learning models for churn prediction, 
and make predictions on new customer data.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    # Preprocess data
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Label Encoding for categorical columns
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"])
    data["Geography"] = le.fit_transform(data["Geography"])

    x = data.drop('Exited', axis=1)
    y = data['Exited']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=0)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Initialize models for comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(probability=True),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    # Train and evaluate models
    accuracy_results = {}
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_results[name] = accuracy
            
            progress_bar.progress((i + 1) / len(models))

    st.success("Training complete!")

    # Display results in a table format with improved visuals
    st.subheader("Model Comparison")
    
    results_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])
    results_df.sort_values('Accuracy', ascending=False, inplace=True)
    
    st.write(results_df.style.highlight_max(axis=0))

    # Visualize results with improved aesthetics
    st.subheader("Model Accuracy Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax, color='skyblue')
    
    plt.title("Model Accuracy Comparison", fontsize=16)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    st.pyplot(fig)

    # Best model display with additional details
    if accuracy_results:  # Ensure there are results to avoid KeyError
        best_model_name = results_df.index[0]
        best_model = models[best_model_name]  # Assign the best model after training
        
        st.subheader(f"Best Model: {best_model_name}")
        st.write(f"Accuracy: {results_df.iloc[0]['Accuracy']:.3f}")

        # Make predictions section with improved layout and instructions
        st.subheader("Make Predictions")
        
        input_data = {}
        
        # Define mappings for categorical variables to display meaningful labels in select boxes.
        geography_mapping = {0: 'France', 1: 'Spain', 2: 'Germany'}
        gender_mapping = {0: 'Female', 1: 'Male'}

        for column in x.columns:
            if column in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
                input_data[column] = st.number_input(f"Enter {column}", value=float(x[column].mean()), format="%.2f")
            elif column == 'Geography':
                input_data[column] = st.selectbox(f"Select Geography", options=list(geography_mapping.keys()), format_func=lambda x: geography_mapping[x])
            elif column == 'Gender':
                input_data[column] = st.selectbox(f"Select Gender", options=list(gender_mapping.keys()), format_func=lambda x: gender_mapping[x])
            else:
                input_data[column] = st.checkbox(f"{column}")

        if st.button("Predict Churn"):
            try:
                input_df = pd.DataFrame([input_data])
                input_df['Gender'] = le.transform(input_df['Gender'])
                input_df['Geography'] = le.transform(input_df['Geography'])
                input_scaled = scaler.transform(input_df)
                
                prediction = best_model.predict(input_scaled)
                probability = best_model.predict_proba(input_scaled)[0][1]
                
                if prediction[0] == 1:
                    st.error(f"The customer is likely to churn. Probability: {probability:.2f}")
                else:
                    st.success(f"The customer is likely to stay. Probability: {1 - probability:.2f}")
            
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

else:
    st.info("Please upload a CSV file to begin.")
