import streamlit as st
st.set_page_config(page_title="Vehicle Registration Dashboard", layout='wide', page_icon='🚗')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("DATASET.csv")
    date_columns = ["regvalidfrom", "regvalidto", "fromdate", "todate"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    return df

df = load_data()

# --- PAGE CONFIGURATION ---

# --- TITLE ---
st.title("🚘 Vehicle Registration Analytics Dashboard")
st.markdown("Explore vehicle registrations, predict fuel type, and get personalized recommendations!")

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔧 Filters")
fuel_filter = st.sidebar.multiselect("Select Fuel Type", df['fuel'].unique(), default=df['fuel'].unique())

# --- FILTER DATA ---
df_filtered = df[df['fuel'].isin(fuel_filter)]

# --- VEHICLES REGISTERED BY LOCATION ---
st.subheader("📍 Number of Vehicles Registered by Location")
location_counts = df_filtered['OfficeCd'].value_counts().reset_index()
location_counts.columns = ['Location', 'Count']
fig_location = px.bar(location_counts, x='Location', y='Count', title="Registrations by Location", color_discrete_sequence=['#4CAF50'])
st.plotly_chart(fig_location, use_container_width=True)

# --- FUEL TYPE DISTRIBUTION ---
st.subheader("⛽ Fuel Type Distribution")
fuel_counts = df_filtered['fuel'].value_counts().reset_index()
fuel_counts.columns = ['Fuel Type', 'Count']
fig_fuel = px.pie(fuel_counts, names='Fuel Type', values='Count', title="Fuel Type Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig_fuel, use_container_width=True)

# --- EV VEHICLE COUNT ---
st.subheader("⚡ Number of EV Vehicles Registered")
ev_count = df_filtered[df_filtered['fuel'].str.contains("Electric", case=False, na=False)].shape[0]
st.metric(label="Total EV Registrations", value=ev_count)

# --- MACHINE LEARNING MODELS ---
st.subheader("🔍 Predict Insurance Validity & Fuel Type")

# Encode categorical features
label_encoder = LabelEncoder()
df['fuel_encoded'] = label_encoder.fit_transform(df['fuel'])

# Prepare dataset for ML
features = ['cc', 'hp', 'seatCapacity', 'cylinder']
X = df[features].fillna(0)

# Linear Regression for Insurance Prediction
y_insurance = df['todate'].fillna(pd.Timestamp("2025-12-31"))
y_insurance = y_insurance.astype("int64") // 10**9
X_train, X_test, y_train, y_test = train_test_split(X, y_insurance, test_size=0.2, random_state=42)
insurance_model = LinearRegression()
insurance_model.fit(X_train, y_train)

# Random Forest for Fuel Type Prediction
y_fuel = df['fuel_encoded']
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X, y_fuel, test_size=0.2, random_state=42)
fuel_model = RandomForestClassifier()
fuel_model.fit(X_train_fuel, y_train_fuel)

# --- USER INPUT FOR PREDICTIONS ---
st.sidebar.subheader("🚘 Enter Vehicle Details")
cc_input = st.sidebar.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1500)
hp_input = st.sidebar.number_input("Horsepower (hp)", min_value=10, max_value=1000, value=100)
seat_input = st.sidebar.number_input("Seat Capacity", min_value=1, max_value=20, value=5)
cylinder_input = st.sidebar.number_input("Cylinders", min_value=1, max_value=12, value=4)

# Location input
location_input = st.sidebar.selectbox("Select RTA Location", df['OfficeCd'].unique())

# --- PREDICTIONS & RECOMMENDATIONS ---
if st.sidebar.button("🔮 Predict"):
    user_input = pd.DataFrame([[cc_input, hp_input, seat_input, cylinder_input]], columns=features)
    predicted_insurance = insurance_model.predict(user_input)[0]
    predicted_fuel = fuel_model.predict(user_input)[0]
    fuel_type = label_encoder.inverse_transform([predicted_fuel])[0]

    st.subheader("🔧 Predictions")
    st.write(f"Predicted Insurance Validity: **{pd.to_datetime(predicted_insurance, unit='s').date()}**")
    st.write(f"Predicted Fuel Type: **{fuel_type}**")

    # Vehicle Recommendation based on location and user preferences
    st.subheader("🚘 Recommended Vehicles")
    recommended_cars = df[(df['cc'] <= cc_input) & (df['hp'] <= hp_input) & 
                          (df['seatCapacity'] == seat_input) & (df['cylinder'] == cylinder_input) &
                          (df['OfficeCd'] == location_input)]
    if not recommended_cars.empty:
        st.write(recommended_cars[['makerName', 'modelDesc', 'fuel']].head(5))
    else:
        st.error("❌ No matching vehicles found for the given criteria.")

# --- PREDICTION BY REGISTRATION NUMBER ---
st.sidebar.subheader("🔍 Check Vehicle Details by Registration Number")
reg_no_input = st.sidebar.text_input("Enter Vehicle Registration Number")
if st.sidebar.button("🔍 Get Details") and reg_no_input:
    vehicle_info = df[df['registrationNo'] == reg_no_input]
    if not vehicle_info.empty:
        st.success(f"✅ Insurance Validity for {reg_no_input}: {vehicle_info['todate'].iloc[0].date()}")
        st.write(vehicle_info[['fuel', 'modelDesc', 'bodyType', 'cc', 'OfficeCd']].iloc[0])
    else:
        st.error("❌ No vehicle found with that registration number.")

st.write("📅 Data last updated from: ", df['regvalidfrom'].max())
