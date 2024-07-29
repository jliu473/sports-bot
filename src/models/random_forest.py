from sklearn.ensemble import RandomForestClassifier
from backtest import load_data, backtest_model
def train_model(features, labels):
    X_train, y_train = features, labels

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model

def test_model(model, predict_features, predict_labels):
    X_test, y_test = predict_features, predict_labels

    accuracy = model.score(X_test, y_test)
    return accuracy

predict_games, features, labels, predict_features, predict_labels = load_data()
model = train_model(features, labels)

def get_probs_rf(predict_features):
  probs = model.predict_proba(predict_features)
  return probs[:,1]

profit, percent_profit, favorite_win, favorite_loss, underdog_win, underdog_loss, bets = backtest_model(get_probs_rf, predict_features, predict_games, min_ev=25)

bets.to_csv('bets/random_forest_bets.csv')