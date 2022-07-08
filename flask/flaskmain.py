import numpy as np
from flask import Flask,render_template , request

import pickle

app = Flask(__name__)


model = pickle.load(open("score_predictor.pkl",'rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict' , methods=['POST'])
def predict():
    present_teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', #0,1,2
                     'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', #3,4,5
                     'Royal Challengers Bangalore', 'Sunrisers Hyderabad'] #6,7

    grounds =['Barabati Stadium', 'Brabourne Stadium', 'Buffalo Park',
              'De Beers Diamond Oval', 'Dr DY Patil Sports Academy', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
              'Dubai International Cricket Stadium', 'Eden Gardens', 'Feroz Shah Kotla',
              'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium',
              'JSCA International Stadium Complex', 'Kingsmead', 'M Chinnaswamy Stadium',
              'MA Chidambaram Stadium, Chepauk', 'Maharashtra Cricket Association Stadium', 'New Wanderers Stadium',
              'Newlands', 'OUTsurance Oval', 'Punjab Cricket Association IS Bindra Stadium, Mohali',
              'Punjab Cricket Association Stadium, Mohali', 'Rajiv Gandhi International Stadium, Uppal',
              'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium',
              'Shaheed Veer Narayan Singh International Stadium', 'Sharjah Cricket Stadium',
              'Sheikh Zayed Stadium', "St George's Park", 'Subrata Roy Sahara Stadium',
              'SuperSport Park', 'Wankhede Stadium']

    batting_team_ar = np.asarray(8)
    bowling_team_ar = np.asarray(8)

    venue_ar = np.asarray(30)

    if request.method == 'POST':

        batting_team = request.form["batting_team"]
        if batting_team == present_teams[0]:
            batting_team_ar[0]=1
        elif batting_team == present_teams[1]:
            batting_team_ar[1]=1
        elif batting_team == present_teams[2]:
            batting_team_ar[2]=1
        elif batting_team == present_teams[3]:
            batting_team_ar[3]=1
        elif batting_team == present_teams[4]:
            batting_team_ar[4]=1
        elif batting_team == present_teams[5]:
            batting_team_ar[5]=1
        elif batting_team == present_teams[6]:
            batting_team_ar[6]=1
        elif batting_team == present_teams[7]:
            batting_team_ar[7]=1

        bowling_team = request.form['bowling_team']
        if bowling_team ==present_teams[0]:
            bowling_team_ar[0]=1
        elif bowling_team ==present_teams[1]:
            bowling_team_ar[1]=1
        elif bowling_team ==present_teams[2]:
            bowling_team_ar[2]=1
        elif bowling_team ==present_teams[3]:
            bowling_team_ar[3]=1
        elif bowling_team ==present_teams[4]:
            bowling_team_ar[4]=1
        elif bowling_team ==present_teams[5]:
            bowling_team_ar[5]=1
        elif bowling_team ==present_teams[6]:
            bowling_team_ar[6]=1
        elif bowling_team ==present_teams[7]:
            bowling_team_ar[7]=1

        venue = request.form['venue']
        index = grounds.index(venue)

        venue_ar[index]=1

        runs = np.array([int(request.form['runs'])])
        wickets = np.array([int(request.form['wickets'])])

        """ ['season', 'toss_winner', 'toss_decision', 'win_by_runs',
       'win_by_wickets', 'venue', 'batting_team', 'bowling_team'],"""
        data = np.concatenate([2022,venue_ar,])













if __name__ == '__main__':
    app.run(debug=True)