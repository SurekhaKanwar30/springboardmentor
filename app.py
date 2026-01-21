import streamlit as st
import pandas as pd
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

label {
    color: #e5e7eb !important;
    font-weight: 600;
}

.card {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: bold;
    background: linear-gradient(90deg, #22c55e, #3b82f6);
    color: white;
    border: none;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="card">
    <h1>🏏 IPL Win Probability Predictor</h1>
    <p>Predict the winning chances of teams based on current match situation</p>
</div>
""", unsafe_allow_html=True)

# -------------------- MODEL DATA --------------------
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

pipe = pickle.load(open('pipe.pkl', 'rb'))

# -------------------- INPUT SECTION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📥 Match Details")

col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowlingteam = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("Match City", sorted(cities))
target = st.number_input("Target Score", min_value=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Current Score", min_value=0)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0)
with col5:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if st.button("Predict Win Probability"):

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    currentrunrate = score / overs if overs != 0 else 0
    requiredrunrate = (runs_left * 6) / balls_left if balls_left != 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [battingteam],
        'bowling_team': [bowlingteam],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'cur_run_rate': [currentrunrate],
        'req_run_rate': [requiredrunrate]
    })

    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    # -------------------- RESULT DISPLAY --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Match Prediction Result")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target", target)
    m2.metric("Score", score)
    m3.metric("Runs Left", runs_left)
    m4.metric("Wickets Left", wickets_left)

    st.subheader("🏆 Winning Probability")

    left, right = st.columns(2)
    with left:
        st.metric(battingteam, f"{round(winprob * 100)}%")
        st.progress(winprob)

    with right:
        st.metric(bowlingteam, f"{round(lossprob * 100)}%")
        st.progress(lossprob)

    st.markdown('</div>', unsafe_allow_html=True)
