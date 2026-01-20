import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
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
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox('Select the city where the match is being played', sorted(cities))
target = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs Completed', min_value=0.1, max_value=20.0)
with col5:
    wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets_fallen
    current_rr = score / overs
    required_rr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [battingteam],
        'bowling_team': [bowlingteam],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'cur_run_rate': [current_rr],
        'req_run_rate': [required_rr]
    })


    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.header(battingteam + " - " + str(round(winprob * 100)) + "%")
    st.header(bowlingteam + " - " + str(round(lossprob * 100)) + "%")

    overs_axis = np.arange(int(np.floor(overs)), 22, 2)

    projected_runs = []
    run_total = score

    for ov in overs_axis:
        if ov <= 6:
            rr = current_rr * 0.9
        elif ov <= 14:
            rr = current_rr * 1.0
        else:
            rr = current_rr * 1.25

        run_total += rr * 2
        projected_runs.append(run_total)

    fig, ax = plt.subplots()
    ax.plot(overs_axis, projected_runs)
    ax.set_title("Expected Runs vs Overs")
    ax.set_xlabel("Overs")
    ax.set_ylabel("Runs")
    ax.set_xticks(np.arange(0, 22, 2))
    ax.set_yticks(np.arange(0, max(projected_runs) + 40, 20))

    st.pyplot(fig)


