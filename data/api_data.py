import requests
import pandas as pd


## TRAINING DATA ## 

def get_season_stats(date):
  base_url = 'https://api.sportsdata.io/v3/nba/scores/json/TeamSeasonStats/'
  API_KEY = '<INSERT API KEY>'
  url = f'{base_url}{date}?key={API_KEY}'
  response = requests.get(url)
  df = pd.DataFrame(response.json())
  return df

dates = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
dates = dates + [date + 'POST' for date in dates]

season_data_raw = pd.concat([get_season_stats(date) for date in dates], ignore_index=True)
season_data_raw.to_csv('season_data_big_raw.csv', index=False)

features = ['SeasonType', 'Season', 'Team', 'Wins', 'Losses', 'Possessions', 'FantasyPoints', 'FieldGoalsPercentage', 'EffectiveFieldGoalsPercentage', 
            'TwoPointersPercentage', 'ThreePointersPercentage', 'FreeThrowsPercentage', 'OffensiveRebounds', 'DefensiveRebounds', 'Rebounds',
            'Assists', 'Steals', 'BlockedShots', 'Turnovers', 'PersonalFouls', 'Points', 'TrueShootingPercentage', 'PlusMinus', 'DoubleDoubles', 
            'TripleDoubles']

season_features = season_data_raw[features]
season_features['WinPercentage'] = season_features['Wins'] / (season_features['Wins'] + season_features['Losses'])
season_features.to_csv('season_features_big.csv', index=False)

def get_games(date):
  base_url = 'https://api.sportsdata.io/v3/nba/scores/json/SchedulesBasic/'
  API_KEY = '<INSERT API KEY>'
  url = f'{base_url}{date}?key={API_KEY}'
  response = requests.get(url)
  df = pd.DataFrame(response.json())
  return df

games_raw = pd.concat([get_games(date) for date in dates], ignore_index=True)
games_raw.to_csv('games_big_raw.csv', index=False)

games = games_raw.loc[games_raw['Status'].isin(['Final', 'F/OT'])]
games = games[['Season', 'SeasonType', 'Day', 'HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore']]

games['HomeWin'] = games['HomeTeamScore'] > games['AwayTeamScore']
games['HomeWin'] = games['HomeWin'].astype(int)

games.to_csv('games_big.csv', index=False)


## PREDICTION DATA ## 

def get_predict_stats(date, team_id):
  base_url = 'https://api.sportsdata.io/v3/nba/scores/json/TeamGameStatsBySeason/'
  API_KEY = '<INSERT API KEY>'
  url = f'{base_url}{date}/{team_id}/all?key={API_KEY}'
  response = requests.get(url)
  df = pd.DataFrame(response.json())
  return df

PREDICT_YEAR = 2024
predict_stats_raw = pd.concat([get_predict_stats(PREDICT_YEAR, team_id) for team_id in range(1, 31)] 
                              + [get_predict_stats(str(PREDICT_YEAR) + 'POST', team_id) for team_id in range(1, 31)], ignore_index=True)
predict_stats_raw.to_csv('2024_stats_raw.csv', index=False)

predict_stats = predict_stats_raw[features + ['Day']]

def month(day):
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

predict_stats['m'] = predict_stats['Day'].apply(month)
predict_stats.drop(columns=['Season', 'Day'], inplace=True)

predict_stats_m0 = season_features[season_features['Season'] == PREDICT_YEAR - 1]
predict_stats_m0['m'] = 'm0'

scale = ['Possessions', 'FantasyPoints', 'OffensiveRebounds', 'DefensiveRebounds', 'Rebounds',
         'Assists', 'Steals', 'BlockedShots', 'Turnovers', 'PersonalFouls', 'Points', 'PlusMinus', 'DoubleDoubles', 
         'TripleDoubles']
for col in scale:
  predict_stats_m0[col] = predict_stats_m0[col] / (predict_stats_m0['Wins'] + predict_stats_m0['Losses'])

predict_stats_m0.drop(columns=['Season', 'SeasonType', 'WinPercentage'], inplace=True)

predict_stats = pd.concat([predict_stats_m0, predict_stats], ignore_index=True)
predict_stats = predict_stats.groupby(['Team', 'm']).mean()
predict_stats['WinPercentage'] = predict_stats['Wins'] / (predict_stats['Wins'] + predict_stats['Losses'])
predict_stats.to_csv('2024_stats.csv')

predict_games = games[games['Season'] == PREDICT_YEAR].reset_index(drop=True)
predict_games.to_csv('2024_games.csv', index=False)

## ODDS DATA ##

def get_odds(date):
  base_url = 'https://api.sportsdata.io/v3/nba/odds/json/GameOddsByDate/'
  API_KEY = '<INSERT API KEY>'
  url = f'{base_url}{date}?key={API_KEY}'
  response = requests.get(url)
  df = pd.DataFrame(response.json())
  return df

all_dates = predict_games['Day'].apply(lambda x: x[0:10]).unique()
odds_raw = pd.concat([get_odds(date) for date in all_dates], ignore_index=True)
odds_raw.to_csv('2024_odds_raw.csv', index=False)

def odds_by_game(df, row):
  game = df.loc[(df['Day'] == row['Day']) & (df['HomeTeamName'] == row['HomeTeam']) & (df['AwayTeamName'] == row['AwayTeam'])]
  odds = game['PregameOdds'].iloc[0]
  home, away = None, None
  for odd in odds:
    if odd['SportsbookId'] == 7:
      home = odd['HomeMoneyLine']
      away = odd['AwayMoneyLine']
      break
  if home is None or away is None:
    raise Exception
  return home, away

def home_odds_by_game(row):
  return odds_by_game(odds, row)[0]

def away_odds_by_game(row):
  return odds_by_game(odds, row)[1]

odds = odds_raw.loc[odds_raw['Status'].isin(['Final', 'F/OT'])]
predict_games_with_odds = predict_games

predict_games_with_odds['HomeMoneyLine'] = predict_games_with_odds.apply(home_odds_by_game, axis=1)
predict_games_with_odds['AwayMoneyLine'] = predict_games_with_odds.apply(away_odds_by_game, axis=1)

predict_games_with_odds.to_csv('2024_games_with_odds.csv')