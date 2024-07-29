from sklearn.ensemble import GradientBoostingClassifier
from backtest import load_data, backtest_model

def train_model(features, labels):
    X_train, y_train = features, labels

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    return model

def test_model(model, predict_features, predict_labels):
    X_test, y_test = predict_features, predict_labels

    accuracy = model.score(X_test, y_test)
    return accuracy

predict_games, features, labels, predict_features, predict_labels = load_data()
model = train_model(features, labels)

def get_probs_gbm(predict_features):
  probs = model.predict_proba(predict_features)
  return probs[:,1]

profit, percent_profit, favorite_win, favorite_loss, underdog_win, underdog_loss, bets = backtest_model(get_probs_gbm, predict_features, predict_games, min_ev=25)

bets.to_csv('bets/gradient_boosting_machine_bets.csv')