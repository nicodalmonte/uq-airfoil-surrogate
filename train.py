import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import AirfoilMLP

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "data/airfoil_self_noise.dat"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 64


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load tab-separated airfoil data. Columns 0-4 = features, 5 = target SPL (dB)."""
    data = np.loadtxt(path, delimiter="\t")
    return data[:, :5], data[:, 5]


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Split: 70 / 15 / 15 ─────────────────────────────────────────────────
    X, y = load_data()

    # Carve out 15 % test first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    # From remaining 85 %, take 15/85 ≈ 17.65 % → gives exactly 15 % of total for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )
    print(f"Splits  — train: {len(X_train)}, cal: {len(X_cal)}, test: {len(X_test)}")

    # ── Scale (fit ONLY on training set) ─────────────────────────────────────
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = X_scaler.fit_transform(X_train)
    X_cal_s   = X_scaler.transform(X_cal)
    X_test_s  = X_scaler.transform(X_test)

    # StandardScaler expects 2-D input for the target
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
    y_cal_s   = y_scaler.transform(y_cal.reshape(-1, 1)).squeeze()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).squeeze()

    # ── Tensors ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_tr = to_tensor(X_train_s).to(device)
    y_tr = to_tensor(y_train_s).to(device)
    X_cv = to_tensor(X_cal_s).to(device)
    y_cv = to_tensor(y_cal_s).to(device)
    X_te = to_tensor(X_test_s).to(device)

    # ── Model / optimiser / loss ──────────────────────────────────────────────
    model     = AirfoilMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_model_state: dict | None = None
    n_train          = len(X_tr)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Shuffle indices for mini-batch SGD
        perm        = torch.randperm(n_train, device=device)
        epoch_loss  = 0.0
        n_batches   = 0

        for start in range(0, n_train, BATCH_SIZE):
            idx     = perm[start : start + BATCH_SIZE]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]

            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # Validation on calibration set (no dropout)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_cv).squeeze(), y_cv).item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            # Clone to CPU so the best state is independent of further training
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={epoch_loss/n_batches:.4f}  "
                f"val_loss={val_loss:.4f}"
            )

    # ── Evaluate best checkpoint on test set ──────────────────────────────────
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        y_pred_s = model(X_te).squeeze().cpu().numpy()

    # Inverse-transform to original dB scale for interpretable metrics
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).squeeze()

    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    r2   = float(r2_score(y_test, y_pred))
    print(f"\n{'-'*50}")
    print(f"Test RMSE : {rmse:.4f} dB")
    print(f"Test R2   : {r2:.4f}")
    print(f"{'-'*50}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "X_scaler":         X_scaler,
            "y_scaler":         y_scaler,
            # Raw (unscaled) splits — downstream scripts load these directly
            "X_cal":  X_cal,
            "y_cal":  y_cal,
            "X_test": X_test,
            "y_test": y_test,
        },
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
