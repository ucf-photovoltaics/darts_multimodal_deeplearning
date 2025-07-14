import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
file_path = r"C:\Users\light\Downloads\clustered_iv_defects.csv"
data = pd.read_csv(file_path)
data.dropna(inplace=True)

# Convert date and time into datetime
data['Measurement_Date'] = data['Measurement_Date'].astype(str)
data['Measurement_Time'] = data['Measurement_Time'].astype(str).str.zfill(6)
data['Measurement_DateTime'] = pd.to_datetime(data['Measurement_Date'] + data['Measurement_Time'], format='%Y%m%d%H%M%S')

# Define features and targets
features = ['Crack', 'Contact', 'Interconnect', 'Corrosion']
targets = ['Voc_(V)', 'Isc_(A)', 'Pmp_(W)', 'Rs_(Ohm)']

# Add time offset feature globally
data = data.sort_values(by='Measurement_DateTime')
data['time_offset'] = (data['Measurement_DateTime'] - data['Measurement_DateTime'].min()).dt.total_seconds()
features += ['time_offset']

# Scale features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

data[features] = scaler_X.fit_transform(data[features])
data[targets] = scaler_y.fit_transform(data[targets])

# Build sequences globally
sequence_length = 10
X_seqs, y_seqs = [], []

X = data[features].values
y = data[targets].values

for i in range(len(data) - sequence_length):
    X_seqs.append(X[i:i + sequence_length])
    y_seqs.append(y[i + sequence_length])

if len(X_seqs) == 0:
    raise ValueError("No sequences created, try reducing sequence_length")

X_tensor = torch.tensor(np.array(X_seqs), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_seqs), dtype=torch.float32)

# Train test split
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = len(features)
hidden_size = 64
num_layers = 1
output_size = len(targets)

model = LSTM(input_size, hidden_size, num_layers, output_size)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Plot training history
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training and Validation Loss')
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_true = y_test.numpy()

# Inverse scale predictions and true values
y_pred_original = scaler_y.inverse_transform(y_pred)
y_true_original = scaler_y.inverse_transform(y_true)

# Metrics
r2 = r2_score(y_true_original, y_pred_original)
mae = mean_absolute_error(y_true_original, y_pred_original)
print(f"R² score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Sample predictions
print("\nPredicted values (first 5):")
print(y_pred_original[:5])
print("\nTrue values (first 5):")
print(y_true_original[:5])

# Plot predicted vs true values for each target
plt.figure(figsize=(12,8))
for i, target in enumerate(targets):
    plt.subplot(2,2,i+1)
    plt.scatter(y_true_original[:, i], y_pred_original[:, i], alpha=0.5)
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(target)
    min_val = min(y_true_original[:, i].min(), y_pred_original[:, i].min())
    max_val = max(y_true_original[:, i].max(), y_pred_original[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.tight_layout()
plt.show()

# Plot predictions and true values over samples
plt.figure(figsize=(14,8))
for i, target in enumerate(targets):
    plt.subplot(len(targets), 1, i+1)
    plt.plot(y_true_original[:, i], label='True')
    plt.plot(y_pred_original[:, i], label='Predicted', alpha=0.7)
    plt.title(f'{target} over samples')
    plt.xlabel('Sample index')
    plt.ylabel(target)
    plt.legend()
plt.tight_layout()
plt.show()


