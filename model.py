import torch
import torch.nn as nn

torch.manual_seed(42)


class AirfoilMLP(nn.Module):
    """MLP surrogate for the NASA Airfoil Self-Noise dataset.

    Architecture: 5 → 128 → 128 → 64 → 1
    ReLU activations, Dropout(p=0.2) after every hidden layer.
    """

    def __init__(self, input_dim: int = 5, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def enable_mc_dropout(self) -> None:
        """MC Dropout mode: full eval() but re-enable every Dropout layer.

        This keeps BatchNorm (if any) in eval while making Dropout stochastic
        so repeated forward passes sample different weight masks.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


if __name__ == "__main__":
    torch.manual_seed(42)
    model = AirfoilMLP()
    print(model)

    x = torch.randn(4, 5)

    # Eval mode must be deterministic
    model.eval()
    out_a = model(x)
    out_b = model(x)
    print("Eval deterministic:", torch.allclose(out_a, out_b))

    # MC Dropout mode must be stochastic
    model.enable_mc_dropout()
    out_c = model(x)
    out_d = model(x)
    print("MC Dropout stochastic:", not torch.allclose(out_c, out_d))
    print("Output shape:", out_c.shape)  # expect (4, 1)
