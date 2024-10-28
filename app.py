import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# Loading the Model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geography = pickle.load(file)
with open('Scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

## Streamlit APP

st.title('Customer Churn Prediction')

Geography = st.selectbox('Geography',onehot_encoder_geography.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data={
    'CreditScore':credit_score,
    'Geography':Geography,
    'Gender' :Gender,
    'Age':age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}
input_data = pd.DataFrame([input_data])

## Geography Enocoder
geo_encoded = onehot_encoder_geography.transform([input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
input_df = input_data.drop('Geography',axis=1)
scaled_df = Scaler.transform(input_df)

prediction = model.predict(scaled_df)
pred_prob = prediction[0][0]

st.write(f"Churn Probabilty : {pred_prob:.2f}")
if  pred_prob > .5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer will not churn")