import pandas as pd
import numpy as np
import pickle

def preprocess_data(games, season_features):
  features, labels = [], []
  ignored_cols = ['SeasonType', 'Season', 'Team', 'Wins', 'Losses']
  true_cols = ['FieldGoalsPercentage', 'EffectiveFieldGoalsPercentage', 'TwoPointersPercentage',
               'ThreePointersPercentage', 'FreeThrowsPercentage', 'TrueShootingPercentage', 'WinPercentage']
  for index, row in games.iterrows():
    season = row['Season']
    season_type = row['SeasonType']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    feature = []

    home_stats = season_features.loc[(season_features['Season'] == season) & (season_features['SeasonType'] == season_type) & (season_features['Team'] == home_team)]
    home_games = home_stats['Wins'].to_numpy()[0] + home_stats['Losses'].to_numpy()[0]
    for col in home_stats.columns:
      if col in ignored_cols:
        continue
      
      if col in true_cols:
        feature.append(home_stats[col].to_numpy()[0])
      else:
        feature.append(home_stats[col].to_numpy()[0] / home_games)

    away_stats = season_features.loc[(season_features['Season'] == season) & (season_features['SeasonType'] == season_type) & (season_features['Team'] == away_team)]
    away_games = away_stats['Wins'].to_numpy()[0] + away_stats['Losses'].to_numpy()[0]
    for col in away_stats.columns:
      if col in ignored_cols:
        continue
      if col in true_cols:
        feature.append(away_stats[col].to_numpy()[0])
      else:
        feature.append(away_stats[col].to_numpy()[0] / away_games)

    features.append(feature)
    labels.append(row['HomeWin'])
    
  return np.array(features), np.array(labels)

def get_time(day):
  m = day[5:7]
  if m == '10' or m == '11':
    return 'm1'
  if m == '12':
    return 'm2'
  if m == '01':
    return 'm3'
  if m == '02':
    return 'm4'
  if m == '03':
    return 'm5'
  if m == '04':
    return 'm6'
  else:
    return 'm7'

def get_prev_time(m):
  return 'm' + str(int(m[1])-1)

def get_predict_features(predict_season_features, predict_games):
  features, labels = [], []
  ignored_cols = ['Team', 'm', 'SeasonType', 'Wins', 'Losses']
  for index, row in predict_games.iterrows():
    season_type = row['SeasonType']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    feature = []
    
    date = row['Day']
    m = get_prev_time(get_time(date))
    
    home_stats = predict_season_features.loc[(predict_season_features['Team'] == home_team) & (predict_season_features['m'] == m)]

    for col in home_stats.columns:
      if col in ignored_cols:
        continue
      else:
        feature.append(home_stats[col].to_numpy()[0])

    away_stats = predict_season_features.loc[(predict_season_features['Team'] == away_team) & (predict_season_features['m'] == m)]

    for col in away_stats.columns:
      if col in ignored_cols:
        continue
      else:
        feature.append(away_stats[col].to_numpy()[0])
    
    features.append(feature)
    labels.append(row['HomeWin'])
    
  return np.array(features), np.array(labels)


season_features = pd.read_csv('season_features_big.csv')
games = pd.read_csv('games_big.csv')
features, labels = preprocess_data(games, season_features)

predict_season_features = pd.read_csv('2024_stats.csv')
predict_games = pd.read_csv('2024_games_with_odds.csv')
predict_features, predict_labels = get_predict_features(predict_season_features, predict_games)

with open('features.pkl', 'wb') as f:
  pickle.dump(features, f)
with open('labels.pkl', 'wb') as f:
  pickle.dump(labels, f)
with open('predict_features.pkl', 'wb') as f:
  pickle.dump(predict_features, f)
with open('predict_labels.pkl', 'wb') as f:
  pickle.dump(predict_labels, f)