import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from backtest import load_data, backtest_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predict_games, features, labels, predict_features, predict_labels = load_data()

normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)
normalized_predict_features = (predict_features - predict_features.mean(axis=0)) / predict_features.std(axis=0)

features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
predict_features_tensor = torch.tensor(predict_features, dtype=torch.float32).to(device)
predict_labels_tensor = torch.tensor(predict_labels, dtype=torch.float32).to(device)

train_dataset = TensorDataset(features_tensor, labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(predict_features_tensor, predict_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = NeuralNetwork().to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

def get_probs_nn(predict_features):
  model.eval()
  probs = []
  with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        probs.append(outputs)
  
  probs = torch.cat(probs).cpu().numpy()
  return probs

profit, percent_profit, favorite_win, favorite_loss, underdog_win, underdog_loss, bets = backtest_model(get_probs_nn, predict_features, predict_games, min_ev=25)

bets.to_csv('bets/neural_network_bets.csv')