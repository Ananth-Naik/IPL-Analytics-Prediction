from flask import Flask, render_template, request
import pandas as pd
import pickle

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(
    open('model/ipl_model.pkl', 'rb')
)

# Load encoders
encoders = pickle.load(
    open('model/encoders.pkl', 'rb')
)

# IPL Teams (ONLY 10 TEAMS)
teams = [

    "Chennai Super Kings",

    "Delhi Capitals",

    "Gujarat Titans",

    "Kolkata Knight Riders",

    "Lucknow Super Giants",

    "Mumbai Indians",

    "Punjab Kings",

    "Rajasthan Royals",

    "Royal Challengers Bengaluru",

    "Sunrisers Hyderabad"
]

# IPL Venues
matches = pd.read_csv('dataset/matches.csv')

venues = sorted(
    matches['venue'].dropna().unique()
)

# Home Page
@app.route('/')

def home():

    return render_template(
        'index.html',
        teams=teams,
        venues=venues
    )

# Prediction Route
@app.route('/predict', methods=['POST'])

def predict():

    # Get form values
    team1 = request.form['team1']
    team2 = request.form['team2']
    toss_winner = request.form['toss_winner']
    toss_decision = request.form['toss_decision']
    venue = request.form['venue']

    # Validation
    if team1 == team2:

        return render_template(
            'index.html',
            prediction_text="❌ Team 1 and Team 2 cannot be same!",
            teams=teams,
            venues=venues
        )

    # Convert Bengaluru name for model compatibility
    def convert_team(team):

        if team == "Royal Challengers Bengaluru":
            return "Royal Challengers Bangalore"

        return team

    model_team1 = convert_team(team1)
    model_team2 = convert_team(team2)
    model_toss = convert_team(toss_winner)

    # Encode inputs
    input_data = pd.DataFrame(
        [[
            encoders['team1'].transform([model_team1])[0],
            encoders['team2'].transform([model_team2])[0],
            encoders['toss_winner'].transform([model_toss])[0],
            encoders['toss_decision'].transform([toss_decision])[0],
            encoders['venue'].transform([venue])[0]
        ]],
        columns=[
            'team1',
            'team2',
            'toss_winner',
            'toss_decision',
            'venue'
        ]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Final winner logic
    if prediction == 1:
        winner = team1
    else:
        winner = team2

    # Return result
    return render_template(
        'index.html',
        prediction_text=f'🏆 Predicted Winner: {winner}',
        teams=teams,
        venues=venues
    )

# Run App
if __name__ == '__main__':

    app.run(debug=True)