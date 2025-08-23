import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Read in data
file_path = r"C:\Users\light\Downloads\clustered_iv_defects.csv"
data = pd.read_csv(file_path)

# Drop columns with no values
data.dropna(inplace=True)

# Define the defect and performacne metrics
defects = ['Crack', 'Contact', 'Interconnect', 'Corrosion']
performance_metrics = ['Voc_(V)', 'Isc_(A)', 'Pmp_(W)', 'Rs_(Ohm)']

# Define x and y
X = data[defects]
y = data[performance_metrics]

# Scale X and Y
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Define the MLP Neural Network
class MLPNN(nn.Module):
    def __init__(self):
        super(MLPNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.25)  # Dropout to prevent overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# Initialize the model
model = MLPNN()

# Set criterion as mean squared error
criterion = nn.MSELoss()

# Using Adam optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# Train the model
epochs = 1000
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    # Set the model to training mode
    model.train()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)

    # Save training and validation losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the test loss
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Compare predicted and true values
y_pred_np = y_pred.numpy()
y_test_np = y_test.numpy()

# Reverse scaling for predictions and true values
y_pred_original = scaler_y.inverse_transform(y_pred_np)
y_test_original = scaler_y.inverse_transform(y_test_np)

# Display the first 5 predicted values and true values for each performance metric
print("Predicted values (first 5 rows):")
print(y_pred_original[:5])
print("\nTrue values (first 5 rows):")
print(y_test_original[:5])

# Calculate R² score
r2 = r2_score(y_test_original, y_pred_original)
print(f"R² score: {r2:.4f}")

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test_original, y_pred_original)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Plot Predicted vs. True Values for Each Performance Metric
plt.figure(figsize=(10, 6))
for i, metric in enumerate(performance_metrics):
    plt.subplot(3, 3, i + 1)
    plt.scatter(y_test_original[:, i], y_pred_original[:, i], label=f"Predicted vs True {metric}")
    plt.xlabel(f"True {metric}")
    plt.ylabel(f"Predicted {metric}")
    plt.title(f"{metric}")
    plt.legend()
plt.tight_layout()
plt.show()
