import os
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"

# Save next to this script, i.e. data/airfoil_self_noise.dat
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airfoil_self_noise.dat")


def download(url: str = URL, save_path: str = SAVE_PATH) -> None:
    """Download the NASA Airfoil Self-Noise dataset from the UCI ML Repository."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Dataset already present: {save_path}")
        return

    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(save_path, "wb") as fh:
        fh.write(response.content)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"Saved to {save_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    download()
