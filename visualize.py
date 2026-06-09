"""Generate all UQ result figures and save them to results/."""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

from conformal import ConformalPredictor
from model import AirfoilMLP
from uq_inference import mc_predict

torch.manual_seed(42)
np.random.seed(42)

CHECKPOINT_PATH = "checkpoints/best_model.pt"
RESULTS_DIR = "results"

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_everything():
    """Load model, scalers, and pre-split datasets from the training checkpoint."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = AirfoilMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        model,
        ckpt["X_scaler"],
        ckpt["y_scaler"],
        ckpt["X_cal"],
        ckpt["y_cal"],
        ckpt["X_test"],
        ckpt["y_test"],
    )


def run_inference(model, X_scaler, y_scaler, X_cal, y_cal, X_test, y_test):
    """Run MC Dropout and conformal prediction; return all quantities needed for plots."""
    X_cal_s  = torch.tensor(X_scaler.transform(X_cal),  dtype=torch.float32)
    X_test_s = torch.tensor(X_scaler.transform(X_test), dtype=torch.float32)

    # MC Dropout (200 stochastic forward passes)
    mc = mc_predict(model, X_test_s, n_samples=200, y_scaler=y_scaler)

    # Conformal prediction calibrated on the calibration set at alpha=0.10
    cp = ConformalPredictor()
    cp.calibrate(model, X_cal_s, y_cal, alpha=0.1, y_scaler=y_scaler)
    lower, upper = cp.predict_interval(model, X_test_s, y_scaler=y_scaler)

    return mc, lower, upper


# ── Plot 01 ───────────────────────────────────────────────────────────────────

def plot_01(mc: dict, y_test: np.ndarray, save_dir: str) -> None:
    """Scatter of predicted mean vs. true SPL, coloured by epistemic uncertainty."""
    mean      = mc["mean"]
    epistemic = mc["epistemic"]

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(y_test, mean, c=epistemic, cmap="viridis", alpha=0.7, s=18, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Epistemic uncertainty (dB²)")

    # Perfect-prediction diagonal
    lo = min(y_test.min(), mean.min()) - 1
    hi = max(y_test.max(), mean.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction (y = x)")

    ax.set_xlabel("True Sound Pressure Level (dB)")
    ax.set_ylabel("Predicted Mean (dB)")
    ax.set_title("Predictions vs. True Values")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(save_dir, "01_predictions_vs_true.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 02 ───────────────────────────────────────────────────────────────────

def plot_02(
    mc: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> None:
    """First 100 test samples (sorted by true value) with both uncertainty bands."""
    n = min(100, len(y_test))

    # Sort by true value so the line is monotone and bands are readable
    sort_idx      = np.argsort(y_test[:n])
    x_axis        = np.arange(n)
    y_sorted      = y_test[:n][sort_idx]
    mean_sorted   = mc["mean"][:n][sort_idx]
    std_sorted    = mc["std"][:n][sort_idx]
    lower_sorted  = lower[:n][sort_idx]
    upper_sorted  = upper[:n][sort_idx]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Conformal interval (light orange) — drawn first so it appears behind
    ax.fill_between(
        x_axis, lower_sorted, upper_sorted,
        alpha=0.35, color="orange", label="Conformal interval (90%)",
    )
    # MC Dropout band: mean ± 2σ (light blue)
    ax.fill_between(
        x_axis,
        mean_sorted - 2 * std_sorted,
        mean_sorted + 2 * std_sorted,
        alpha=0.40, color="steelblue", label="MC Dropout ±2σ",
    )
    # True values
    ax.plot(x_axis, y_sorted,    color="black",     lw=1.8, label="True value")
    # MC mean
    ax.plot(x_axis, mean_sorted, color="steelblue", lw=1.0, linestyle="--", label="MC mean")

    ax.set_xlabel("Sample index (sorted by true SPL)")
    ax.set_ylabel("Sound Pressure Level (dB)")
    ax.set_title("MC Dropout and Conformal Prediction Uncertainty Bands (first 100 samples)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(save_dir, "02_uncertainty_bands.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 03 ───────────────────────────────────────────────────────────────────

def plot_03(mc: dict, y_test: np.ndarray, save_dir: str) -> None:
    """Scatter of epistemic uncertainty vs. absolute error, with regression trendline."""
    epistemic = mc["epistemic"]
    abs_error = np.abs(mc["mean"] - y_test)

    # Linear fit (y = ax + b)
    coeffs  = np.polyfit(epistemic, abs_error, 1)
    trend_x = np.linspace(epistemic.min(), epistemic.max(), 300)
    trend_y = np.polyval(coeffs, trend_x)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(epistemic, abs_error, alpha=0.45, s=18, color="steelblue", zorder=3)
    ax.plot(
        trend_x, trend_y, "r-", lw=2,
        label=f"Linear fit  (slope = {coeffs[0]:.2f})",
    )

    ax.set_xlabel("Epistemic uncertainty (dB²)")
    ax.set_ylabel("Absolute error (dB)")
    ax.set_title("Higher Epistemic Uncertainty → Higher Error")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(save_dir, "03_epistemic_vs_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 04 ───────────────────────────────────────────────────────────────────

def plot_04(
    model,
    X_scaler,
    y_scaler,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> None:
    """Bar chart comparing empirical vs. theoretical coverage for α ∈ {0.05, 0.10, 0.20}."""
    alphas = [0.05, 0.10, 0.20]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    X_cal_s  = torch.tensor(X_scaler.transform(X_cal),  dtype=torch.float32)
    X_test_s = torch.tensor(X_scaler.transform(X_test), dtype=torch.float32)

    empirical: list[float] = []
    for alpha in alphas:
        cp = ConformalPredictor()
        cp.calibrate(model, X_cal_s, y_cal, alpha=alpha, y_scaler=y_scaler)
        lo, hi   = cp.predict_interval(model, X_test_s, y_scaler=y_scaler)
        covered  = float(((y_test >= lo) & (y_test <= hi)).mean())
        empirical.append(covered)

    theoretical = [1.0 - a for a in alphas]
    x_pos       = np.arange(len(alphas))
    bar_width   = 0.45

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x_pos, empirical, bar_width, color=colors, alpha=0.82, zorder=3)

    # Annotate each bar with its empirical value
    for bar, emp in zip(bars, empirical):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{emp:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Horizontal dashed lines at each theoretical target, coloured to match bars
    for theo, color in zip(theoretical, colors):
        ax.axhline(theo, color=color, linestyle="--", linewidth=1.8, alpha=0.9,
                   label=f"Target {theo:.0%}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α = {a}" for a in alphas])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Coverage")
    ax.set_title("Conformal Prediction: Empirical vs. Theoretical Coverage")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    path = os.path.join(save_dir, "04_coverage_calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading checkpoint …")
    model, X_scaler, y_scaler, X_cal, y_cal, X_test, y_test = load_everything()

    print("Running MC Dropout and conformal inference …")
    mc, lower, upper = run_inference(model, X_scaler, y_scaler, X_cal, y_cal, X_test, y_test)

    print("\nGenerating plots …")
    plot_01(mc, y_test, RESULTS_DIR)
    plot_02(mc, lower, upper, y_test, RESULTS_DIR)
    plot_03(mc, y_test, RESULTS_DIR)
    plot_04(model, X_scaler, y_scaler, X_cal, y_cal, X_test, y_test, RESULTS_DIR)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
