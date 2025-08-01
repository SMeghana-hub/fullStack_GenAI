import pandas as pd
import streamlit as st
import joblib
import datetime
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="🔋 Energy Predictor", layout="centered")

# --- HEADER WITH BACKGROUND IMAGE ---
background_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1470&q=80"
st.markdown(f"""
    <style>
    .header-bg {{
        position: relative;
        background-image: url('{background_url}');
        background-size: cover;
        background-position: center;
        height: 250px;
        border-radius: 15px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        margin-bottom: 25px;
    }}
    .header-bg h1 {{
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }}
    </style>

    <div class="header-bg">
        <h1>🏠 Home Energy Consumption Predictor</h1>
    </div>
""", unsafe_allow_html=True)

# --- MODEL SELECTION ---
model_choice = st.selectbox("🤖 Choose a Prediction Model", ["Random Forest", "Linear Regression"])

# --- MODEL LOAD ---
if model_choice == "Random Forest":
    model = joblib.load("Random-Forest-model.pkl")
else:
    model = joblib.load("Linear.pkl")

# --- MODEL COLUMNS (28 features expected by both models) ---
model_columns = [
    'num_occupants', 'house_size_sqft', 'monthly_income', 'outside_temp_celsius',
    'year', 'month', 'day',
    'heating_type_Electric', 'heating_type_Gas', 'heating_type_None',
    'cooling_type_AC', 'cooling_type_Fan', 'cooling_type_None',
    'manual_override_Y', 'manual_override_N', 'is_weekend', 'temp_above_avg',
    'income_per_person', 'square_feet_per_person', 'high_income_flag', 'low_temp_flag',
    'season_spring', 'season_summer', 'season_fall', 'season_winter',
    'day_of_week_0', 'day_of_week_6', 'energy_star_home'
]

# --- SIDEBAR INPUT FIELDS ---
st.sidebar.header("🧮 Input Parameters")

num_occupants = st.sidebar.number_input("👨‍👩‍👧‍👦 Number of Occupants", min_value=1, value=4)
house_size_sqft = st.sidebar.number_input("📐 House Size (sqft)", min_value=100, value=2000)
monthly_income = st.sidebar.number_input("💰 Monthly Income ($)", min_value=1000, value=30000)
outside_temp_celsius = st.sidebar.number_input("🌡️ Outside Temp (°C)", min_value=-10, max_value=50, value=26)
year = st.sidebar.number_input("📆 Year", min_value=2020, max_value=2100, value=2025)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1)

# --- Derived Values ---
try:
    date_obj = datetime.date(year, month, day)
except ValueError:
    st.error("Invalid date selected.")
    st.stop()

day_of_week = date_obj.weekday()
is_weekend = int(day_of_week >= 5)

season_label = (
    "winter" if month in [12, 1, 2]
    else "spring" if month in [3, 4, 5]
    else "summer" if month in [6, 7, 8]
    else "fall"
)

heating_type = st.sidebar.selectbox("🔥 Heating Type", ["Electric", "Gas", "None"])
cooling_type = st.sidebar.selectbox("❄️ Cooling Type", ["AC", "Fan", "None"])
manual_override = st.sidebar.radio("🛠️ Manual Override", ["Y", "N"])
energy_star_home = st.sidebar.checkbox("🏡 Certified Energy Star Home")

# Additional features
temp_above_avg = int(outside_temp_celsius > 28)
income_per_person = monthly_income / num_occupants
square_feet_per_person = house_size_sqft / num_occupants
high_income_flag = int(monthly_income > 40000)
low_temp_flag = int(outside_temp_celsius < 15)

# --- Input Feature Dictionary ---
input_features = {
    'num_occupants': num_occupants,
    'house_size_sqft': house_size_sqft,
    'monthly_income': monthly_income,
    'outside_temp_celsius': outside_temp_celsius,
    'year': year,
    'month': month,
    'day': day,
    'heating_type_Electric': int(heating_type == "Electric"),
    'heating_type_Gas': int(heating_type == "Gas"),
    'heating_type_None': int(heating_type == "None"),
    'cooling_type_AC': int(cooling_type == "AC"),
    'cooling_type_Fan': int(cooling_type == "Fan"),
    'cooling_type_None': int(cooling_type == "None"),
    'manual_override_Y': int(manual_override == "Y"),
    'manual_override_N': int(manual_override == "N"),
    'is_weekend': is_weekend,
    'temp_above_avg': temp_above_avg,
    'income_per_person': income_per_person,
    'square_feet_per_person': square_feet_per_person,
    'high_income_flag': high_income_flag,
    'low_temp_flag': low_temp_flag,
    'season_spring': int(season_label == "spring"),
    'season_summer': int(season_label == "summer"),
    'season_fall': int(season_label == "fall"),
    'season_winter': int(season_label == "winter"),
    'day_of_week_0': int(day_of_week == 0),
    'day_of_week_6': int(day_of_week == 6),
    'energy_star_home': int(energy_star_home)
}

# --- DataFrame for Prediction ---
input_df = pd.DataFrame([input_features])[model_columns]

# --- PREDICT BUTTON ---
if st.button("🚀 Predict Energy Consumption"):
    prediction = model.predict(input_df)[0]

    # --- Display Result ---
    st.markdown(f"""
        <div style='background-color:#E8F5E9;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:#1B5E20;'>🔋 Estimated Energy Usage:</h2>
            <h1 style='color:#004D40;'>{prediction:.2f} kWh</h1>
        </div>
    """, unsafe_allow_html=True)

    # --- Extra Visuals for Random Forest ---
    st.subheader("📊 Breakdown of Derived Features")
    bar_data = pd.DataFrame({
        'Feature': ['Income/Person', 'Sqft/Person', 'High Income Flag', 'Low Temp Flag'],
        'Value': [income_per_person, square_feet_per_person, high_income_flag, low_temp_flag]
    })
    st.bar_chart(bar_data.set_index('Feature'))

    st.subheader("📉 Energy Usage vs. Average")
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.barh(['Your Usage'], [prediction], color='#4caf50')
    ax.axvline(50, color='red', linestyle='--', label="Avg Usage (50 kWh)")
    ax.set_xlim(0, max(100, prediction + 10))
    ax.set_xlabel("Energy Usage (kWh)")
    ax.legend()
    st.pyplot(fig)

    st.image("https://cdn-icons-png.flaticon.com/512/2622/2622338.png", width=100, caption="Eco-Friendly Living")
