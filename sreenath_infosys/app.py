import streamlit as st
import pandas as pd
import pickle

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="IPL WIN PROBABILITY PREDICTOR",
    page_icon="🏏",
    layout="centered"
)


# 🎨 UI Styling


st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(120deg, #f0f4f8, #d9e2ec);
}

/* Main container */
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.08);
}

/* Titles */
h1 {
    color: #102a43;
    font-weight: 800;
}

h2, h3 {
    color: #243b53;
}

/* Labels */
label {
    color: #334e68 !important;
    font-weight: 600;
}

/* Inputs */
.stSelectbox, .stNumberInput {
    background-color: #f8fafc !important;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    border-radius: 10px;
    padding: 0.7em;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Result Card */
.result-box {
    background-color: #f1f5f9;
    padding: 25px;
    border-radius: 16px;
    margin-top: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

hr {
    border: 1px solid #cbd5e1;
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
pipe = pickle.load(open('pipe.pkl', 'rb'))

# =========================
# App UI
# =========================
st.title("🏏IPL WIN PROBABILITY PREDICTOR")
st.caption("Educational tool to analyze match situations in IPL")
st.markdown("---")

# Team Selection
col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowlingteam = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("Match Venue", sorted(cities))

target = st.number_input("Target Score", min_value=0, step=1)

st.markdown("### Match Progress")

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Current Score", min_value=0, step=1)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Prediction
# =========================
if st.button("Analyze Match Situation"):

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets

    current_run_rate = score / overs if overs > 0 else 0
    required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [battingteam],
        'bowling_team': [bowlingteam],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'cur_run_rate': [current_run_rate],
        'req_run_rate': [required_run_rate]
    })

    result = pipe.predict_proba(input_df)

    loss_prob = result[0][0]
    win_prob = result[0][1]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("Match Winning Probability")

    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"#### {battingteam}")
        st.metric("Win Probability", f"{round(win_prob * 100)}%")

    with colB:
        st.markdown(f"#### {bowlingteam}")
        st.metric("Win Probability", f"{round(loss_prob * 100)}%")

    st.markdown("### Run Rate Comparison")

    rate_df = pd.DataFrame({
        "Type": ["Current Run Rate", "Required Run Rate"],
        "Runs": [round(current_run_rate, 2), round(required_run_rate, 2)]
    })

    st.bar_chart(rate_df, x="Type", y="Runs")

    st.markdown('</div>', unsafe_allow_html=True)
