import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.pkl')
loaded_encoder = joblib.load('encodingFile.pkl')
loaded_scaler = joblib.load('scalingFile.pkl')

def input_to_df(input):
  data = [input]
  df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
  return df

def encode(df):
  for column in df.columns:
    if df[column].dtype == "object":
      df[column] = loaded_encoder.fit_transform(df[column])
  return df

def normalize(df):
  df = loaded_scaler.transform(df)
  return df

def predict_with_model(model, user_input):
  prediction = model.predict(user_input)
  return prediction[0]

def main():
  st.title('Diabetes Classification')
  st.info("This app use machine learning to classify diabetes levels.")

  st.subheader("Dataset")
  df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
  x, y = split_x_y(df)
  with st.expander("**Raw Data**"):
    st.dataframe(df.head(50))

  with st.expander("**Input Data**"):
    st.dataframe(x.head(50))

  with st.expander("**Output Data**"):
    st.dataframe(y.head(50))

  st.subheader("Height vs Weight With Obesity Level")
  with st.expander('**Data Visualization**'):
    st.scatter_chart(data=df, x = 'Height', y = 'Weight', color='NObeyesdad')

  # input data by user
  st.subheader("Input Patient Data")
  Age = st.slider('Age', min_value = 10, max_value = 65, value = 25)
  Height = st.slider('Height', min_value = 1.45, max_value = 2.00, value = 1.75)
  Weight = st.slider('Weight', min_value = 30, max_value = 180, value = 70)
  FCVC = st.slider('FCVC', min_value = 1, max_value = 3, value = 2)
  NCP = st.slider('NCP', min_value = 1, max_value = 4, value = 3)
  CH2O = st.slider('CH2O', min_value = 1, max_value = 3, value = 2)
  FAF = st.slider('FAF', min_value = 0, max_value = 3, value = 1)
  TUE = st.slider('TUE', min_value = 0, max_value = 2, value = 1)
  
  Gender = st.selectbox('Gender', ('Male', 'Female'))
  family_history_with_overweight = st.selectbox('Family history with overweight', ('yes', 'no'))
  FAVC = st.selectbox('FAVC', ('yes', 'no'))
  CAEC = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'no'))
  SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
  SCC = st.selectbox('SCC', ('yes', 'no'))
  CALC = st.selectbox('CALC', ('Sometimes', 'no', 'Frequently', 'Always'))
  MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

  input_data = [Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]

  user_df = convert_input_to_df(input_data)

  st.subheader("Inputted Patient Data")
  st.dataframe(user_df)

  user_df = encode_features(user_df)
  user_df = normalize_features(user_df)

  prediction = predict_classification(user_df)
  proba = classification_proba(user_df)

  st.subheader("Prediction Result")
  st.dataframe(proba)
  st.write('The predicted output is: ', prediction)
  

if __name__ == "__main__":
  main()
