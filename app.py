import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('artifacts\preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('artifacts\model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title("ML App")
gender=st.selectbox("Gender",["male","female"])
race_ethnicity=st.selectbox("Race Ethnicity",["group A","group B","group C","group D","group E"])
parental_level_of_education=st.selectbox("Parental level of Education",["bachelor's degree","some college","master's degree","associate's degree","high school","some high school"])
lunch=st.selectbox("Lunch",["standard","free/reduced"])
test_preparation_course=st.selectbox("Test preparation course",["none","completed"])
reading_score=st.number_input("Enter Reading Score")
writing_score=st.number_input("Enter Writing Score")

data = {
    'gender':[gender],
    'race_ethnicity':[race_ethnicity],
    'parental_level_of_education':[parental_level_of_education],
    'lunch':[lunch],
    'test_preparation_course':[test_preparation_course],
    'reading_score':[reading_score],
    'writing_score':[writing_score]
}
data=pd.DataFrame(data)
st.write(data)

scaled_data = preprocessor.transform(data)
prediction = model.predict(scaled_data)
st.write(f"Predicted Maths Score is: {prediction}")