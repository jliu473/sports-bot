from flask import Flask, render_template, request, redirect
import pandas as pd

app = Flask(__name__)

games_dict = {"logistic-regression": pd.read_csv('bets/logistic_regression_bets.csv'),
             "neural-network": pd.read_csv('bets/neural_network_bets.csv'),
             "random-forest": pd.read_csv('bets/random_forest_bets.csv'),
             "gradient-boosting-machine": pd.read_csv('bets/gradient_boosting_machine_bets.csv')}
model_dict = {"logistic-regression": "Logistic Regression",
             "neural-network": "Neural Network",
             "random-forest": "Random Forest",
             "gradient-boosting-machine": "Gradient Boosting Machine"}


@app.route("/")
def index():
    games = games_dict['logistic-regression']
    total_profit = "${:.2f}".format(games.iloc[0]['TotalProfit'])
    recent_games = games[(games['Bet'] == 'Home') | (games['Bet'] == 'Away')].tail(10)
    recent_games = recent_games[::-1]

    recent_bets = []
    for index, row in recent_games.iterrows():
        if row['Profit'] < 0:
            profit = "-$" + str(abs(round(row['Profit'])))
        else:
            profit = "+$" + str(round(row['Profit']))
        date = row['Day'][:10]
        if row['Bet'] == 'Home':
            recent_bets.append({"Team": row['HomeTeam'], "Profit": profit, "Date": date})
        elif row['Bet'] == 'Away':
            recent_bets.append({"Team": row['AwayTeam'], "Profit": profit, "Date": date})
    
    recent_games['HomeEV'] = recent_games['HomeEV'].map('{:.2f}'.format)
    recent_games['AwayEV'] = recent_games['AwayEV'].map('{:.2f}'.format)
    recent_games['HomeWinProb'] = (recent_games['HomeWinProb'] * 100).map('{:.2f}'.format)
    recent_games['AwayWinProb'] = (recent_games['AwayWinProb'] * 100).map('{:.2f}'.format)

    return render_template('index.html', total_profit=total_profit, recent_bets=recent_bets, recent_games=recent_games, model='logistic-regression', model_dict=model_dict)

@app.route("/<model>")
def index_with_model(model):
    games = games_dict[model]
    total_profit = "${:.2f}".format(games.iloc[0]['TotalProfit'])
    recent_games = games[(games['Bet'] == 'Home') | (games['Bet'] == 'Away')].tail(10)
    recent_games = recent_games[::-1]

    recent_bets = []
    for index, row in recent_games.iterrows():
        if row['Profit'] < 0:
            profit = "-$" + str(abs(round(row['Profit'])))
        else:
            profit = "+$" + str(round(row['Profit']))
        date = row['Day'][:10]
        if row['Bet'] == 'Home':
            recent_bets.append({"Team": row['HomeTeam'], "Profit": profit, "Date": date})
        elif row['Bet'] == 'Away':
            recent_bets.append({"Team": row['AwayTeam'], "Profit": profit, "Date": date})
    
    recent_games['HomeEV'] = recent_games['HomeEV'].map('{:.2f}'.format)
    recent_games['AwayEV'] = recent_games['AwayEV'].map('{:.2f}'.format)
    recent_games['HomeWinProb'] = (recent_games['HomeWinProb'] * 100).map('{:.2f}'.format)
    recent_games['AwayWinProb'] = (recent_games['AwayWinProb'] * 100).map('{:.2f}'.format)

    return render_template('index.html', total_profit=total_profit, recent_bets=recent_bets, recent_games=recent_games, model=model, model_dict=model_dict)

@app.route("/<model>/<date>")
def date(model, date):
    games = games_dict[model]
    games_date = games[games['Day'] == date + "T00:00:00"]
    games_date = games_date[['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore', 'HomeWin', 'HomeMoneyLine', 'AwayMoneyLine', 'HomeEV', 'AwayEV', 'HomeWinProb', 'AwayWinProb', 'Bet', 'Profit']]
    games_date['HomeEV'] = games_date['HomeEV'].map('{:.2f}'.format)
    games_date['AwayEV'] = games_date['AwayEV'].map('{:.2f}'.format)
    games_date['HomeWinProb'] = (games_date['HomeWinProb'] * 100).map('{:.2f}'.format)
    games_date['AwayWinProb'] = (games_date['AwayWinProb'] * 100).map('{:.2f}'.format)

    bets = []
    profit = 0
    for index, row in games_date.iterrows():
        if row['Bet'] == 'Home' or row['Bet'] == 'Away':
            profit += round(row['Profit'])
            if row['Profit'] < 0:
                profit_str = "-$" + str(abs(round(row['Profit'])))
            else:
                profit_str = "+$" + str(round(row['Profit']))
            if row['Bet'] == 'Home':
                bets.append({"Team": row['HomeTeam'], "Profit": profit_str})
            elif row['Bet'] == 'Away':
                bets.append({"Team": row['AwayTeam'], "Profit": profit_str})
    if profit < 0:
        profit = "-$" + str(abs(profit))
    else:
        profit = "+$" + str(profit)
    
    total_profit = "${:.2f}".format(games.iloc[0]['TotalProfit'])

    return render_template('date.html', date=date, table=games_date, bets=bets, profit=profit, total_profit=total_profit, model=model, model_dict=model_dict)


@app.route('/set_date/<model>', methods=['POST'])
def set_date(model):
    date = request.form['date']
    return redirect(f'/{model}/{date}')

@app.route('/set_model', methods=['POST'])
def set_model():
    model = request.form['model']
    return redirect(f'{model}')
