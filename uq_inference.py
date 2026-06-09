"""MC Dropout inference: predictive mean, total std, epistemic and aleatoric uncertainty."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from model import AirfoilMLP

torch.manual_seed(42)
np.random.seed(42)

CHECKPOINT_PATH = "checkpoints/best_model.pt"


def load_model_and_scalers(
    checkpoint_path: str = CHECKPOINT_PATH,
) -> tuple[AirfoilMLP, StandardScaler, StandardScaler]:
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AirfoilMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["X_scaler"], ckpt["y_scaler"]


def mc_predict(
    model: AirfoilMLP,
    X: torch.Tensor,
    n_samples: int = 200,
    y_scaler: StandardScaler | None = None,
) -> dict[str, np.ndarray]:
    """Monte Carlo Dropout predictive inference.

    Runs `n_samples` stochastic forward passes (Dropout active) and decomposes
    the resulting ensemble into epistemic and aleatoric uncertainty estimates.

    Uncertainty decomposition
    ─────────────────────────
    epistemic  : variance of per-point predictions across MC samples.
                 Captures *model / weight uncertainty* — large when the MC
                 ensemble disagrees about a specific input point.

    aleatoric  : mean of per-MC-run variances computed across the full dataset.
                 Approximates *irreducible / data noise* under the homoscedastic
                 assumption of a single-output MLP (no explicit noise head).
                 This gives a dataset-level constant broadcast to (N,).

    Both quantities are returned in original target scale (dB²  for variances).

    Args:
        model     : AirfoilMLP instance.
        X         : Scaled input tensor, shape (N, 5).
        n_samples : Number of MC forward passes (default 200).
        y_scaler  : Fitted StandardScaler for the target; required to
                    inverse-transform from scaled space to original dB scale.

    Returns:
        dict with numpy arrays:
            "mean"      – predictive mean,            shape (N,)
            "std"       – total predictive std,        shape (N,)
            "epistemic" – variance of MC means,        shape (N,)
            "aleatoric" – mean of per-run variances,   shape (N,)  [constant]
    """
    # Activate MC Dropout: model in eval (frozen BN), only Dropout layers in train
    model.enable_mc_dropout()

    # Collect T stochastic predictions —  shape will be (n_samples, N)
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X).squeeze(-1).cpu().numpy()  # (N,)
            preds_list.append(out)

    preds = np.stack(preds_list, axis=0)  # (n_samples, N)

    # Inverse-transform: scaled space → original dB space
    # StandardScaler is affine: y_orig = y_scaled * σ + μ
    if y_scaler is not None:
        sigma  = float(y_scaler.scale_[0])
        mu     = float(y_scaler.mean_[0])
        preds  = preds * sigma + mu  # still shape (n_samples, N)

    # ── Predictive mean (N,) ──────────────────────────────────────────────────
    mean = preds.mean(axis=0)

    # ── Total predictive std (N,) ─────────────────────────────────────────────
    std = preds.std(axis=0)

    # ── Epistemic uncertainty (N,) ────────────────────────────────────────────
    # For each input point i, measure how much the T MC samples disagree.
    # High epistemic → the model is uncertain about this region of input space.
    epistemic = preds.var(axis=0)  # (N,)

    # ── Aleatoric uncertainty (N,) ────────────────────────────────────────────
    # "Mean of per-sample (per-MC-run) variances across the dataset."
    #   Step 1: for each MC run t, compute variance of that run's predictions
    #           across all N data points  →  shape (n_samples,)
    #   Step 2: average across runs  →  scalar approximating data-level noise
    #   Step 3: broadcast to (N,) so all downstream code is shape-consistent.
    per_run_variance = preds.var(axis=1)          # (n_samples,)
    aleatoric        = np.full(mean.shape, per_run_variance.mean())  # (N,)

    return {
        "mean":      mean,
        "std":       std,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
    }


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model, X_scaler, y_scaler = load_model_and_scalers()

    ckpt   = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    X_test = ckpt["X_test"]
    y_test = ckpt["y_test"]

    X_test_s = X_scaler.transform(X_test)
    X_tensor = torch.tensor(X_test_s, dtype=torch.float32)

    results = mc_predict(model, X_tensor, n_samples=200, y_scaler=y_scaler)

    rmse = float(np.sqrt(np.mean((results["mean"] - y_test) ** 2)))
    print(f"MC Dropout — Test RMSE     : {rmse:.4f} dB")
    print(f"Mean epistemic uncertainty : {results['epistemic'].mean():.4f} dB^2")
    print(f"Mean aleatoric uncertainty : {results['aleatoric'].mean():.4f} dB^2")
    print(f"Mean total std             : {results['std'].mean():.4f} dB")
