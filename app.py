# app.py
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")

# ============================================================
# Custom styling
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Reduce top padding — multiple selectors for cross-version compatibility */
    .block-container,
    section.main > div.block-container,
    div[data-testid="stAppViewContainer"] > .main .block-container,
    div[data-testid="stMain"] .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 950px;
    }

    /* Page background */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #eef2ff 0%, #f5f7fa 100%);
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 10px 0 25px 0;
    }
    .hero h1 {
        font-size: 42px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .hero p {
        font-size: 16px;
        color: #64748b;
    }

    /* Form card */
    div[data-testid="stForm"] {
        background: white;
        padding: 35px 40px;
        border-radius: 18px;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.07);
        border: 1px solid #eef0f5;
    }

    /* Section subheaders inside form */
    .section-title {
        font-size: 17px;
        font-weight: 600;
        color: #1f6feb;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Submit button */
    div.stButton > button, button[kind="formSubmit"] {
        width: 100%;
        background: linear-gradient(135deg, #1f6feb, #4facfe);
        color: white;
        font-size: 17px;
        font-weight: 600;
        padding: 0.7em 0;
        border-radius: 12px;
        border: none;
        margin-top: 10px;
        transition: 0.25s ease;
    }
    div.stButton > button:hover, button[kind="formSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 18px rgba(31, 111, 235, 0.35);
        color: white;
    }

    /* Result card */
    .price-card {
        background: linear-gradient(135deg, #1f6feb, #4facfe);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-top: 25px;
        box-shadow: 0px 8px 25px rgba(31, 111, 235, 0.3);
    }
    .price-card .label {
        font-size: 15px;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .price-card .amount {
        font-size: 38px;
        font-weight: 700;
    }

    /* Responsive tweak for small screens */
    @media (max-width: 640px) {
        div[data-testid="stForm"] {
            padding: 20px;
        }
        .hero h1 {
            font-size: 30px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.ubj")
    return model

model = load_model()

# ============================================================
# Header
# ============================================================
st.markdown("""
    <div class="hero">
        <h1>✈️ Flight Price Predictor</h1>
        <p>Get an instant price estimate for your next flight</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# Input form — 8 fields, 4 per column
# ============================================================
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-title">🛫 Departure Details</div>', unsafe_allow_html=True)
        dep_date = st.date_input("Departure Date")
        dep_time = st.time_input("Departure Time")
        source = st.selectbox("Source City", ['Banglore', 'Delhi', 'Kolkata', 'Mumbai', 'Chennai'])
        airline = st.selectbox("Airline", [
            'Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet',
            'Vistara', 'GoAir', 'Multiple carriers Premium economy',
            'Jet Airways Business', 'Vistara Premium economy', 'Trujet', 'Air Asia'
        ])

    with col2:
        st.markdown('<div class="section-title">🛬 Arrival Details</div>', unsafe_allow_html=True)
        arr_date = st.date_input("Arrival Date")
        arr_time = st.time_input("Arrival Time")
        destination = st.selectbox("Destination City", ['Banglore', 'Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])
        stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

    st.write("")
    submitted = st.form_submit_button("Predict Price 🔍")

# ============================================================
# Predict
# ============================================================
if submitted:

    Journey_day = dep_date.day
    Journey_month = dep_date.month

    Dep_hour = dep_time.hour
    Dep_min = dep_time.minute

    Arrival_hour = arr_time.hour
    Arrival_min = arr_time.minute

    dur_hour = abs(Arrival_hour - Dep_hour)
    dur_min = abs(Arrival_min - Dep_min)

    Total_stops = stops

    # --- Airline one-hot (Air Asia = 0, not in columns) ---
    airline_cols = ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet',
                     'Vistara', 'GoAir', 'Multiple carriers Premium economy',
                     'Jet Airways Business', 'Vistara Premium economy', 'Trujet']
    airline_vals = {c: 0 for c in airline_cols}
    if airline in airline_vals:
        airline_vals[airline] = 1

    Jet_Airways = airline_vals['Jet Airways']
    IndiGo = airline_vals['IndiGo']
    Air_India = airline_vals['Air India']
    Multiple_carriers = airline_vals['Multiple carriers']
    SpiceJet = airline_vals['SpiceJet']
    Vistara = airline_vals['Vistara']
    GoAir = airline_vals['GoAir']
    Multiple_carriers_Premium_economy = airline_vals['Multiple carriers Premium economy']
    Jet_Airways_Business = airline_vals['Jet Airways Business']
    Vistara_Premium_economy = airline_vals['Vistara Premium economy']
    Trujet = airline_vals['Trujet']

    # --- Source one-hot (Banglore = 0, not in columns) ---
    source_cols = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
    source_vals = {c: 0 for c in source_cols}
    if source in source_vals:
        source_vals[source] = 1

    s_Delhi = source_vals['Delhi']
    s_Kolkata = source_vals['Kolkata']
    s_Mumbai = source_vals['Mumbai']
    s_Chennai = source_vals['Chennai']

    # --- Destination one-hot (Banglore = 0, not in columns) ---
    dest_cols = ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
    dest_vals = {c: 0 for c in dest_cols}
    if destination in dest_vals:
        dest_vals[destination] = 1

    d_Cochin = dest_vals['Cochin']
    d_Delhi = dest_vals['Delhi']
    d_New_Delhi = dest_vals['New Delhi']
    d_Hyderabad = dest_vals['Hyderabad']
    d_Kolkata = dest_vals['Kolkata']

    input_data = [[
        Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
        Arrival_hour, Arrival_min, dur_hour, dur_min,
        Air_India, GoAir, IndiGo, Jet_Airways, Jet_Airways_Business,
        Multiple_carriers, Multiple_carriers_Premium_economy, SpiceJet,
        Trujet, Vistara, Vistara_Premium_economy,
        s_Chennai, s_Delhi, s_Kolkata, s_Mumbai,
        d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi
    ]]

    prediction = model.predict(input_data)
    output = round(float(prediction[0]), 2)

    st.markdown(f"""
        <div class="price-card">
            <div class="label">Estimated Flight Price</div>
            <div class="amount">₹ {output}</div>
        </div>
    """, unsafe_allow_html=True)