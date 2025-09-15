import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path


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


def run_first_mlp(file_path: str, epochs: int = 1000, output_dir: str | None = None):
    """
    Runs the MLP model training and evaluation pipeline.

    Args:
        file_path (str): Path to clustered_iv_defects.csv
        epochs (int): Number of training epochs
        output_dir (str | None): Folder to save plots; defaults to Foundational-Package/output  # CHANGED: documented

    Returns:
        dict: metrics including test_loss, r2, mae
    """

    # Ensure output directory exists
    if output_dir is None:
        # power_loss_model/first_mlp.py -> parents[1] == Foundational-Package
        output_path = Path(__file__).resolve().parents[1] / "output"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    # Define features and targets
    defects = ['Crack', 'Contact', 'Interconnect', 'Corrosion']
    performance_metrics = ['Voc_(V)', 'Isc_(A)', 'Pmp_(W)', 'Rs_(Ohm)']

    X = data[defects]
    y = data[performance_metrics]

    # Scale
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Initialize model, criterion, optimizer
    model = MLPNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    # Training loop
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    # Plot training/validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path / "first_mlp_loss.png", bbox_inches="tight", dpi=300)
    plt.show()

    # Evaluate the test loss
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)

    # Convert to numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()

    # Reverse scaling
    y_pred_original = scaler_y.inverse_transform(y_pred_np)
    y_test_original = scaler_y.inverse_transform(y_test_np)

    # Calculate metrics
    r2 = r2_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)

    print("\n========== Model Evaluation ==========")
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    print(f"R² Score       : {r2:.4f}")
    print(f"MAE            : {mae:.4f}")

    # Display first 5 rows in table format
    results_df = pd.DataFrame(
        {
            f"Predicted {m}": y_pred_original[:5, i]
            for i, m in enumerate(performance_metrics)
        }
    )
    for i, m in enumerate(performance_metrics):
        results_df[f"True {m}"] = y_test_original[:5, i]

    print("\n========== Sample Predictions (first 5) ==========")
    print(results_df.to_string(index=False))

    # Plot Predicted vs True
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(performance_metrics):
        plt.subplot(2, 2, i + 1)
        plt.scatter(y_test_original[:, i], y_pred_original[:, i], label=f"{metric}")
        plt.xlabel(f"True {metric}")
        plt.ylabel(f"Predicted {metric}")
        plt.title(metric)
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "first_mlp_pred_vs_true.png", bbox_inches="tight", dpi=300)
    plt.show()

    return {
        "test_loss": test_loss,
        "r2_score": r2,
        "mae": mae,
        "predicted": y_pred_original[:5],
        "true": y_test_original[:5]
    }
