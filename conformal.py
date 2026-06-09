"""Split Conformal Prediction for regression on the airfoil surrogate model."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from model import AirfoilMLP

torch.manual_seed(42)
np.random.seed(42)

CHECKPOINT_PATH = "checkpoints/best_model.pt"


class ConformalPredictor:
    """Split Conformal Prediction for regression.

    Uses a held-out calibration set to compute a nonconformity score quantile
    q_hat.  For any new test point x, the prediction interval is:

        [f(x) − q_hat,  f(x) + q_hat]

    where f(x) is the model's point prediction.

    This guarantees marginal coverage:  P(y ∈ interval) ≥ 1 − α
    with finite-sample validity (Theorem 1, Angelopoulos & Bates 2021).

    Reference: "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification", Angelopoulos & Bates (2021).
    """

    def __init__(self) -> None:
        self.q_hat: float | None = None
        self.alpha: float | None = None

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        model: AirfoilMLP,
        X_cal: torch.Tensor,
        y_cal: np.ndarray,
        alpha: float = 0.1,
        y_scaler: StandardScaler | None = None,
    ) -> float:
        """Compute and store q_hat for (1 − alpha) coverage.

        Algorithm
        ─────────
        1. Run the model (eval mode) on the calibration set.
        2. Inverse-transform predictions to original dB scale.
        3. Compute nonconformity scores:  s_i = |y_i − ŷ_i|.
        4. q_hat = quantile of s at level  ⌈(n+1)(1−α)⌉ / n,
           which guarantees marginal coverage ≥ 1 − α.

        Args:
            model    : Trained AirfoilMLP.
            X_cal    : Scaled calibration features, torch.Tensor (N_cal, 5).
            y_cal    : Calibration targets in **original** scale, numpy (N_cal,).
            alpha    : Miscoverage rate; targets (1−alpha) coverage.
            y_scaler : StandardScaler for inverse-transforming predictions.

        Returns:
            q_hat (float) in original dB scale.
        """
        self.alpha = alpha

        model.eval()
        with torch.no_grad():
            preds_s = model(X_cal).squeeze(-1).cpu().numpy()  # scaled predictions

        # Inverse-transform predictions to original dB scale
        if y_scaler is not None:
            preds = preds_s * float(y_scaler.scale_[0]) + float(y_scaler.mean_[0])
        else:
            preds = preds_s

        # Nonconformity score: absolute residual in original scale
        scores = np.abs(y_cal - preds)  # (N_cal,)

        n = len(scores)
        # Finite-sample quantile level — ensures coverage ≥ 1 − α
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        self.q_hat = float(np.quantile(scores, q_level))

        print(
            f"Conformal calibration  alpha={alpha:.2f}  "
            f"q_hat={self.q_hat:.4f} dB  "
            f"(target coverage {1-alpha:.0%})"
        )
        return self.q_hat

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_interval(
        self,
        model: AirfoilMLP,
        X: torch.Tensor,
        q_hat: float | None = None,
        y_scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetric prediction intervals: ŷ ± q_hat.

        Args:
            model    : Trained AirfoilMLP.
            X        : Scaled features, torch.Tensor (N, 5).
            q_hat    : Override threshold; uses stored value if None.
            y_scaler : StandardScaler for inverse-transforming predictions.

        Returns:
            (lower, upper) — numpy arrays of shape (N,) in original dB scale.
        """
        q = q_hat if q_hat is not None else self.q_hat
        if q is None:
            raise RuntimeError("Call calibrate() before predict_interval().")

        model.eval()
        with torch.no_grad():
            preds_s = model(X).squeeze(-1).cpu().numpy()

        if y_scaler is not None:
            preds = preds_s * float(y_scaler.scale_[0]) + float(y_scaler.mean_[0])
        else:
            preds = preds_s

        return preds - q, preds + q

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_coverage(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Print and return empirical coverage and mean interval width.

        Args:
            lower, upper : Prediction interval bounds, shape (N,).
            y_true       : True targets in original scale, shape (N,).

        Returns:
            dict with keys "coverage" and "mean_width".
        """
        covered    = (y_true >= lower) & (y_true <= upper)
        coverage   = float(covered.mean())
        mean_width = float((upper - lower).mean())

        target = 1.0 - (self.alpha or 0.0)
        print(
            f"Empirical coverage : {coverage:.4f}  "
            f"(theoretical target: {target:.2f})"
        )
        print(f"Mean interval width: {mean_width:.4f} dB")

        return {"coverage": coverage, "mean_width": mean_width}


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt    = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model   = AirfoilMLP()
    model.load_state_dict(ckpt["model_state_dict"])

    X_scaler = ckpt["X_scaler"]
    y_scaler = ckpt["y_scaler"]
    X_cal    = ckpt["X_cal"]
    y_cal    = ckpt["y_cal"]
    X_test   = ckpt["X_test"]
    y_test   = ckpt["y_test"]

    X_cal_s  = torch.tensor(X_scaler.transform(X_cal),  dtype=torch.float32)
    X_test_s = torch.tensor(X_scaler.transform(X_test), dtype=torch.float32)

    cp = ConformalPredictor()
    cp.calibrate(model, X_cal_s, y_cal, alpha=0.1, y_scaler=y_scaler)
    lower, upper = cp.predict_interval(model, X_test_s, y_scaler=y_scaler)
    cp.evaluate_coverage(lower, upper, y_test)
