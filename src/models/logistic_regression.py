from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from backtest import load_data, backtest_model

def train_model(features, labels):
    X_train, y_train = features, labels
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler

def test_model(model, scaler, predict_features, predict_labels):
    X_test, y_test = predict_features, predict_labels
    X_test = scaler.transform(X_test)
    
    accuracy = model.score(X_test, y_test)
    return accuracy

predict_games, features, labels, predict_features, predict_labels = load_data()
model, scaler = train_model(features, labels)

def get_probs_lr(predict_features):
  X_test = scaler.transform(predict_features)
  probs = model.predict_proba(X_test)
  return probs[:,1]

profit, percent_profit, favorite_win, favorite_loss, underdog_win, underdog_loss, bets = backtest_model(get_probs_lr, predict_features, predict_games, min_ev=25)

bets.to_csv('bets/logistic_regression_bets.csv')