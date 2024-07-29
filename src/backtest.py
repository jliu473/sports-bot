import pandas as pd
import matplotlib.pyplot as plt
import pickle

def load_data():
    predict_games = pd.read_csv('../data/2024_games_with_odds.csv')
    with open('../data/features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('../data/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('../data/predict_features.pkl', 'rb') as f:
        predict_features = pickle.load(f)
    with open('../data/predict_labels.pkl', 'rb') as f:
        predict_labels = pickle.load(f)
    return predict_games, features, labels, predict_features, predict_labels


def convert_odds(odds, bet_size=1):
  # assume bet size of $100
  if odds < 0:
    return bet_size * 100 / abs(odds)
  else:
    return bet_size * odds / 100


def backtest_model(get_probs, predict_features, predict_games, min_ev=0):
  probs = get_probs(predict_features)

  profit = 0
  favorite_win, favorite_loss, underdog_win, underdog_loss = 0, 0, 0, 0
  bets = []

  for index, row in predict_games.iterrows():
    home_win_prob = probs[index]
    away_win_prob = 1 - home_win_prob
    
    # bet $100 on every game
    home_profit, away_profit = convert_odds(row['HomeMoneyLine'], bet_size=100), convert_odds(row['AwayMoneyLine'], bet_size=100)
    home_ev = home_win_prob * home_profit + away_win_prob * -100
    away_ev = away_win_prob * away_profit + home_win_prob * -100

    bet_row = row.copy()
    bet_row['HomeEV'] = home_ev
    bet_row['AwayEV'] = away_ev
    bet_row['HomeWinProb'] = home_win_prob
    bet_row['AwayWinProb'] = away_win_prob

    if home_ev > min_ev:
      bet_row['Bet'] = 'Home'
      
      if row['HomeWin'] == 1:
        profit += home_profit
        bet_row['Profit'] = home_profit
        if row['HomeMoneyLine'] < 0: 
          favorite_win += 1
        else:
          underdog_win += 1

      else:
        profit -= 100
        bet_row['Profit'] = -100
        if row['HomeMoneyLine'] < 0: 
          favorite_loss += 1
        else:
          underdog_loss += 1

      bets.append(bet_row)

    elif away_ev > min_ev:
      bet_row = row.copy()
      bet_row['Bet'] = 'Away'

      if row['HomeWin'] == 0:
        profit += away_profit
        bet_row['Profit'] = away_profit
        if row['AwayMoneyLine'] < 0: 
          favorite_win += 1
        else:
          underdog_win += 1

      else:
        profit -= 100
        bet_row['Profit'] = -100
        if row['AwayMoneyLine'] < 0: 
          favorite_loss += 1
        else:
          underdog_loss += 1
    else:
      bet_row['Bet'] = 'None'
      bet_row['Profit'] = 0
      
      bets.append(bet_row)

  percent_profit = profit / ((favorite_win + favorite_loss + underdog_win + underdog_loss) * 100)
  bets = pd.DataFrame(bets).drop(columns=['Unnamed: 0']).reset_index(drop=True)

  bets['TotalProfit'] = profit
  bets['PercentProfit'] = percent_profit
  bets['FavoriteWin'] = favorite_win
  bets['FavoriteLoss'] = favorite_loss
  bets['UnderdogWin'] = underdog_win
  bets['UnderdogLoss'] = underdog_loss

  return profit, percent_profit, favorite_win, favorite_loss, underdog_win, underdog_loss, bets


def backtest_plots(get_probs, predict_features, predict_games, low=-20, high=100):
  min_evs = list(range(low, high+1))
  y = [backtest_model(get_probs, predict_features, predict_games, min_ev=min_ev) for min_ev in min_evs]
  profits = [i[0] for i in y]
  percent_profits = [i[1] for i in y]

  plt.plot(min_evs, profits)
  plt.title('Profit by Min EV')
  plt.xlabel('Minimum EV')
  plt.ylabel('Profit ($)')
  plt.show()

  plt.plot(min_evs, percent_profits)
  plt.title('Percent Profit by Min EV')
  plt.xlabel('Minimum EV')
  plt.ylabel('Percent Profit (%)')
  plt.show()

  return profits, percent_profits