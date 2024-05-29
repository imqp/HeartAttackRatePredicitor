import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Define values for categorical features
sex_mapping = {'Female': 0, 'Male': 1}
cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
fbs_mapping = {'False': 0, 'True': 1}
restecg_mapping = {'Normal': 0, 'Having ST-T Wave Abnormality': 1, 'Showing Probable or Definite Left Ventricular Hypertrophy': 2}
exng_mapping = {'No': 0, 'Yes': 1}
slp_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
caa_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
thall_mapping = {'Null': 0, 'Fixed Defect': 1, 'Normal': 2, 'Reversible Defect': 3}

# Define categorical features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# Define numerical features
numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# Read in the data
# df = pd.read_csv('heart.csv')

# Create a sidebar for user inputs
st.sidebar.header('User Input Features')

def map_user_input_features(features):
    # Map user inputs to real values
    features['sex'] = sex_mapping[features['sex'][0]]
    features['cp'] = cp_mapping[features['cp'][0]]
    features['fbs'] = fbs_mapping[features['fbs'][0]]
    features['restecg'] = restecg_mapping[features['restecg'][0]]
    features['exng'] = exng_mapping[features['exng'][0]]
    features['slp'] = slp_mapping[features['slp'][0]]
    features['caa'] = caa_mapping[str(features['caa'][0])]
    features['thall'] = thall_mapping[features['thall'][0]]

    return features

def user_input_features():
    age = st.sidebar.text_input('Age', value=str(60))
    sex = st.sidebar.selectbox('Sex', options=['Female', 'Male'])
    cp = st.sidebar.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], index=2)
    trtbps = st.sidebar.text_input('Resting Blood Pressure (mmHg)', value=str(120)) # 
    chol = st.sidebar.text_input('Cholesterol (mg/dl)', value=str(200)) # 
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['True', 'False'], index=1) # 
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', options=['Normal', 'Having ST-T Wave Abnormality', 'Showing Probable or Definite Left Ventricular Hypertrophy']) #
    thalachh = st.sidebar.text_input('Max Heart Rate Achieved (bpm)', value=str(80)) # 
    exng = st.sidebar.selectbox('Exercise Induced Angina', options=['No', 'Yes'])
    oldpeak = st.sidebar.text_input('Oldpeak', value=str(1))
    slp = st.sidebar.selectbox('Slope', options=['Upsloping', 'Flat', 'Downsloping'])
    caa = st.sidebar.selectbox('Number of Major Vessels', options=['0', '1', '2', '3', '4'], index=4)
    thall = st.sidebar.selectbox('Thalassemia', options=['Null', 'Normal', 'Fixed Defect', 'Reversible Defect'], index=0)


    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall}
    features = pd.DataFrame(data, index=[0])


    return map_user_input_features(features)

input_df = user_input_features()


# Read in saved classification model
load_clf = pickle.load(open('rf_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)[0]
prediction_proba = load_clf.predict_proba(input_df)

# print(prediction[0])

st.subheader('Prediction')
heart_attack = ['low chance of heart attack', 'high chance of heart attack']
st.write('You have a ', heart_attack[prediction], ' (Probability:', prediction_proba[0][prediction] * 100, ')')

# Create a TreeExplainer for tree-based models
explainer = shap.TreeExplainer(load_clf)
base_value = explainer.expected_value[1]

# Calculate Shap values
choosen_instance = input_df
shap_values = explainer.shap_values(choosen_instance)
# shap.initjs()
shap.plots.force(base_value, shap_values[0][:, 1], choosen_instance, matplotlib=True)

# Display plot in Streamlit
st.subheader('Feature Importance')
st.pyplot(plt.gcf())

# Suggestions
suggestions = []
st.subheader('Suggestions')

high_cholesterol_suggestions = ['- Eat heart-healthy foods', 
                                '- Exercise regularly', 
                                '- Quit smoking', 
                                '- Maintain a healthy weight', 
                                '- Limit the amount of alcohol you drink']

high_blood_pressure_suggestions = ['- Exercise regularly',
                                    '- Limit the amount of alcohol you drink',
                                    '- Quit smoking',
                                    '- Eat a healthy diet',
                                    '- Maintain a healthy weight',
                                    '- Reduce sodium in your diet',
                                    '- Reduce caffeine',
                                    '- Try some relaxation techniques to reduce stress, such as deep breathing or meditation']

high_fasting_blood_sugar_suggestions = ['- Exercise regularly',
                                        '- Eat a healthy diet',
                                        '- Maintain a healthy weight',
                                        '- Get enough sleep',
                                        '- Try some relaxation techniques to reduce stress, such as deep breathing or meditation']

if prediction == 1:
    threshold = np.sort(shap_values[0][:, 1])[6]
    # print(threshold)
    if shap_values[0][:, 1][3] < threshold and shap_values[0][:, 1][4] < threshold and shap_values[0][:, 1][5] < threshold and shap_values[0][:, 1][7] < threshold:
        suggestions.append('You are at high risk of heart attack. Please consult a doctor immediately.')

    if shap_values[0][:, 1][4] > threshold:
        suggestions.extend(high_cholesterol_suggestions)

    if shap_values[0][:, 1][3] > threshold or shap_values[0][:, 1][7] > threshold:
        suggestions.extend(high_blood_pressure_suggestions)

    if shap_values[0][:, 1][5] > threshold:
        suggestions.extend(high_fasting_blood_sugar_suggestions)

else :
    suggestions.append('- You are healthy! Keep up the good work!')

for suggestion in suggestions:
    st.markdown(suggestion)