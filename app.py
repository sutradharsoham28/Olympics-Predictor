import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Set up basic page configuration
st.set_page_config(page_title='Olympic Medal Predictor', layout='wide')

# 2. Add a main title for the application
st.title('Olympic Medal Predictor: Analyzing Historical Data')

# 3. Add a brief project description
st.markdown(
    """ 
    This application predicts the type of Olympic medal (Gold, Silver, or Bronze) an athlete might win based on historical data. 
    It leverages machine learning models trained on various features like year, country demographics, and sport characteristics. 
    Explore the factors influencing medal outcomes and evaluate potential predictions. 
    """
)

st.write("Application structure created. Further components will be added here.")

# Load preprocessing artifacts
# The best_model.pkl and scaler.pkl will be loaded in later steps.
# For now, we only need top_categories_map for input fields.
top_categories_map = joblib.load('top_categories_map.pkl')

# --- Sidebar for Feature Input ---
st.sidebar.header('Feature Input')

# Numerical Features
user_year = st.sidebar.number_input('Year', min_value=1896, max_value=2014, value=2000)
user_population = st.sidebar.number_input('Population', min_value=1000, max_value=2000000000, value=50000000)
user_gdp_per_capita = st.sidebar.number_input('GDP per Capita', min_value=100, max_value=150000, value=30000)

# Low-Cardinality Categorical Features (Boolean after encoding)
user_season = st.sidebar.selectbox('Season', ['Summer', 'Winter'])
user_gender = st.sidebar.selectbox('Gender', ['Men', 'Women'])

# High-Cardinality Categorical Features
high_cardinality_cols = ['City', 'Sport', 'Discipline', 'Athlete', 'Country_Code', 'Event', 'Country_Name']
user_categorical_inputs = {}

for col in high_cardinality_cols:
    options = top_categories_map[col] + ['Other'] # Add 'Other' option
    # Convert column name to a more user-friendly format for display
    display_name = col.replace('_', ' ').replace('Code', 'Country Code').replace('Name', 'Country Name').title()
    user_categorical_inputs[col] = st.sidebar.selectbox(f'{display_name}', options)

# --- Load Preprocessing Artifacts and Model Features (Global for efficiency) ---
scaler = joblib.load('scaler.pkl')
top_categories_map = joblib.load('top_categories_map.pkl')
best_model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Reconstruct model_features (column names of X_train) to ensure consistency
# This involves reloading the cleaned data and re-applying the feature dropping steps
# --- Preprocessing Function for User Inputs ---
def preprocess_inputs(user_year, user_population, user_gdp_per_capita, user_season, user_gender, user_categorical_inputs, model_features, scaler, top_categories_map):
    # Initialize a DataFrame with all model_features as columns and one row, all zeros/False
    processed_input_df = pd.DataFrame(False, index=[0], columns=model_features)

    # 1. Handle numerical features (Year, Population, GDP_per_Capita)
    processed_input_df['Year'] = user_year

    # Scale Population and GDP_per_Capita
    input_for_scaling = pd.DataFrame([[
        user_population, 
        user_gdp_per_capita
    ]], columns=['Population', 'GDP_per_Capita'])
    scaled_values = scaler.transform(input_for_scaling)
    processed_input_df['Population'] = scaled_values[0, 0]
    processed_input_df['GDP_per_Capita'] = scaled_values[0, 1]

    # 2. Handle boolean features (Season, Gender)
    processed_input_df['Season_Winter'] = (user_season == 'Winter')
    processed_input_df['Gender_Women'] = (user_gender == 'Women')

    # 3. Handle high-cardinality categorical features
    high_cardinality_cols = ['City', 'Sport', 'Discipline', 'Athlete', 'Country_Code', 'Event', 'Country_Name']
    for col in high_cardinality_cols:
        user_val = user_categorical_inputs[col]
        effective_category = user_val if user_val in top_categories_map[col] else 'Other'
        
        ohe_col_name = f"{col}_{effective_category}"
        # Set the corresponding one-hot encoded column to True if it exists in model_features
        if ohe_col_name in model_features:
            processed_input_df[ohe_col_name] = True
            
    return processed_input_df

# --- Prediction Logic ---

# Button to trigger prediction
if st.sidebar.button('Predict Medal Type'):
    # Collect user inputs into a dictionary for categorical processing
    current_user_categorical_inputs = {}
    for col in high_cardinality_cols:
        current_user_categorical_inputs[col] = user_categorical_inputs[col]

    # Preprocess user inputs
    processed_input_df = preprocess_inputs(
        user_year,
        user_population,
        user_gdp_per_capita,
        user_season,
        user_gender,
        current_user_categorical_inputs,
        model_features,
        scaler,
        top_categories_map
    )

    # Ensure the order of columns matches the training data
    # This is crucial for the model to make correct predictions
    processed_input_df = processed_input_df[model_features]

    # Make prediction
    prediction = best_model.predict(processed_input_df)
    predicted_medal = label_encoder.inverse_transform(prediction)[0]

    st.subheader('Prediction Result:')
    st.write(f"Based on the input features, the predicted medal type is: **{predicted_medal}**")

    st.markdown(
        """
        --- 
        ### Important Considerations for Prediction:
        *   **Model Limitations:** The model achieved a moderate accuracy (~62%), indicating that predicting the exact medal type (Gold, Silver, or Bronze) is challenging with the available macro-level features.
        *   **Feature Granularity:** The model relies on historical, demographic, and broad sport-level data. Micro-level factors like individual athlete form, injury status, or event-specific conditions, which are highly influential in determining medal outcomes, are not included.
        *   **Context is Key:** While this prediction offers an estimate, real-world Olympic success is complex and multifactorial.
        """
    )

# --- Prediction Logic ---

# Button to trigger prediction
if st.sidebar.button('Predict Medal Type'):
    try:
        # Collect user inputs into a dictionary for categorical processing
        current_user_categorical_inputs = {}
        for col in high_cardinality_cols:
            current_user_categorical_inputs[col] = user_categorical_inputs[col]

        # Preprocess user inputs
        processed_input_df = preprocess_inputs(
            user_year,
            user_population,
            user_gdp_per_capita,
            user_season,
            user_gender,
            current_user_categorical_inputs,
            model_features,
            scaler,
            top_categories_map
        )

        # Ensure the order of columns matches the training data
        processed_input_df = processed_input_df[model_features]

        # Make prediction
        prediction = best_model.predict(processed_input_df)
        predicted_medal = label_encoder.inverse_transform(prediction)[0]

        # Get prediction probabilities
        prediction_proba = best_model.predict_proba(processed_input_df)[0]
        proba_df = pd.DataFrame({
            'Medal Type': label_encoder.classes_,
            'Probability': prediction_proba
        }).sort_values(by='Probability', ascending=False)

        st.subheader('Prediction Result:')
        st.write(f"Based on the input features, the predicted medal type is: **{predicted_medal}**")

        st.subheader('Prediction Probabilities:')
        st.write(proba_df.set_index('Medal Type'))
        st.bar_chart(proba_df.set_index('Medal Type'))

        st.markdown(
            """
            ---
            ### Important Considerations for Prediction:
            *   **Model Limitations:** The model achieved a moderate accuracy (~62%), indicating that predicting the exact medal type (Gold, Silver, or Bronze) is challenging with the available macro-level features.
            *   **Feature Granularity:** The model relies on historical, demographic, and broad sport-level data. Micro-level factors like individual athlete form, injury status, or event-specific conditions, which are highly influential in determining medal outcomes, are not included.
            *   **Context is Key:** While this prediction offers an estimate, real-world Olympic success is complex and multifactorial.
            """
        )

    except Exception as e:
        st.error(f'An error occurred during prediction: {e}. Please check your inputs.')
