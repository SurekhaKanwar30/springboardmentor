import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# =========================
# 🎨 Advanced UI Styling
# =========================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #f9f9f9;
    font-family: 'Segoe UI', sans-serif;
}

label {
    color: #dddddd !important;
}

.stSelectbox, .stNumberInput {
    background-color: #1f2937 !important;
    border-radius: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    padding: 0.6em;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #dd2476, #ff512f);
}

.result-card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(255, 82, 82, 0.4);
    margin-top: 20px;
}

hr {
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Teams & Cities
# =========================
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# =========================
# Load Model
# =========================
pipe = pickle.load(open('pickel.pkl', 'rb'))

# =========================
# App UI
# =========================
st.title("🏏 IPL Win Predictor")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select Bowling Team', sorted(teams))

city = st.selectbox('Match City', sorted(cities))

target = st.number_input('Target Score', min_value=0, step=1)

st.markdown("### 📊 Match Progress")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Prediction
# =========================
if st.button('🚀 Predict Winning Probability'):

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    currentrunrate = score / overs if overs > 0 else 0
    requiredrunrate = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [battingteam], 
        'bowling_team': [bowlingteam], 
        'city': [city], 
        'runs_left': [runs_left], 
        'balls_left': [balls_left], 
        'wickets': [wickets], 
        'total_runs_x': [target], 
        'cur_run_rate': [currentrunrate], 
        'req_run_rate': [requiredrunrate]
    })

    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.markdown("---")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🏆 Win Probability")

    res1, res2 = st.columns(2)

    with res1:
        st.markdown(f"### {battingteam}")
        st.title(f"{round(winprob * 100)}%")
        st.progress(winprob)

    with res2:
        st.markdown(f"### {bowlingteam}")
        st.title(f"{round(lossprob * 100)}%")
        st.progress(lossprob)

    st.markdown("### 📈 Run Rate Analysis")

    rates = pd.DataFrame({
        'Metric': ['Current Run Rate', 'Required Run Rate'],
        'Value': [round(currentrunrate, 2), round(requiredrunrate, 2)]
    })

    st.bar_chart(rates, x='Metric', y='Value')

    st.markdown('</div>', unsafe_allow_html=True)
