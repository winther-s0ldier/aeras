import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


U_WIND = 1.0
D_COEFF = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class ToyPINN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inputs = torch.stack([x, t], dim=1)
        return self.net(inputs).squeeze(1)


def pde_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    C = model(x, t)


    dC_dx = torch.autograd.grad(C.sum(), x, create_graph=True)[0]
    dC_dt = torch.autograd.grad(C.sum(), t, create_graph=True)[0]


    d2C_dx2 = torch.autograd.grad(dC_dx.sum(), x, create_graph=True)[0]


    residual = dC_dt + U_WIND * dC_dx - D_COEFF * d2C_dx2
    return residual


def train_toy_pinn():
    model = ToyPINN().to(DEVICE)

    n_colloc = 5000
    n_bc = 500
    n_ic = 500


    adam_epochs = 15000
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=adam_epochs, eta_min=1e-5
    )

    print(f"Phase 1: Adam for {adam_epochs} epochs (PDE weight=10)...")
    start = time.time()
    history = []

    for epoch in range(adam_epochs):
        optimizer.zero_grad()


        x_c = torch.rand(n_colloc, device=DEVICE)
        t_c = torch.rand(n_colloc, device=DEVICE)
        residual = pde_residual(model, x_c, t_c)
        loss_pde = (residual ** 2).mean()


        x_ic = torch.rand(n_ic, device=DEVICE)
        t_ic = torch.zeros(n_ic, device=DEVICE)
        C_ic = model(x_ic, t_ic)
        C_ic_true = torch.sin(2 * np.pi * x_ic)
        loss_ic = ((C_ic - C_ic_true) ** 2).mean()


        t_bc = torch.rand(n_bc, device=DEVICE)
        C_left = model(torch.zeros(n_bc, device=DEVICE), t_bc)
        C_right = model(torch.ones(n_bc, device=DEVICE), t_bc)
        loss_bc = ((C_left - C_right) ** 2).mean()


        loss = 10 * loss_pde + loss_ic + loss_bc

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            history.append({
                "epoch": epoch,
                "total": loss.item(),
                "pde": loss_pde.item(),
                "ic": loss_ic.item(),
                "bc": loss_bc.item(),
            })
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:5d} | Loss: {loss.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | "
                  f"BC: {loss_bc.item():.4e} | LR: {lr:.2e}")

    adam_time = time.time() - start
    print(f"\nPhase 1 complete in {adam_time:.1f}s")


    print("\nPhase 2: L-BFGS fine-tuning (1000 steps)...")
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe"
    )


    x_c_f = torch.rand(n_colloc, device=DEVICE)
    t_c_f = torch.rand(n_colloc, device=DEVICE)
    x_ic_f = torch.rand(n_ic, device=DEVICE)
    t_ic_f = torch.zeros(n_ic, device=DEVICE)
    t_bc_f = torch.rand(n_bc, device=DEVICE)

    lbfgs_steps = 0

    def closure():
        nonlocal lbfgs_steps
        lbfgs.zero_grad()
        res = pde_residual(model, x_c_f.clone(), t_c_f.clone())
        l_pde = (res ** 2).mean()
        C_ic_ = model(x_ic_f, t_ic_f)
        l_ic = ((C_ic_ - torch.sin(2 * np.pi * x_ic_f)) ** 2).mean()
        C_l = model(torch.zeros(n_bc, device=DEVICE), t_bc_f)
        C_r = model(torch.ones(n_bc, device=DEVICE), t_bc_f)
        l_bc = ((C_l - C_r) ** 2).mean()
        l = 10 * l_pde + l_ic + l_bc
        l.backward()
        lbfgs_steps += 1
        if lbfgs_steps % 100 == 0:
            print(f"  L-BFGS step {lbfgs_steps:4d} | PDE: {l_pde.item():.4e} | "
                  f"IC: {l_ic.item():.4e} | BC: {l_bc.item():.4e}")
        return l

    for _ in range(50):
        lbfgs.step(closure)


    with torch.enable_grad():
        res = pde_residual(model, x_c_f.clone(), t_c_f.clone())
        final_pde = (res ** 2).mean().item()


    with torch.no_grad():
        C_ic_ = model(x_ic_f, t_ic_f)
        final_ic = ((C_ic_ - torch.sin(2 * np.pi * x_ic_f)) ** 2).mean().item()

    history.append({"epoch": adam_epochs, "pde": final_pde, "ic": final_ic,
                    "bc": 0.0, "total": 0.0})

    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed:.1f}s")
    return model, history


def visualize_results(model):
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    times = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]

    x_test = torch.linspace(0, 1, 200, device=DEVICE)

    for idx, t_val in enumerate(times):
        ax = axes[idx // 3][idx % 3]
        t_test = torch.full_like(x_test, t_val)

        with torch.no_grad():
            C_pred = model(x_test, t_test).cpu().numpy()


        x_np = x_test.cpu().numpy()
        C_true = (np.exp(-D_COEFF * (2 * np.pi) ** 2 * t_val) *
                  np.sin(2 * np.pi * (x_np - U_WIND * t_val)))

        ax.plot(x_np, C_true, "k-", linewidth=2, label="Analytical")
        ax.plot(x_np, C_pred, "r--", linewidth=2, label="PINN")
        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("x")
        ax.set_ylabel("C(x, t)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("1D Advection-Diffusion: PINN vs Analytical Solution", fontsize=14)
    plt.tight_layout()

    save_path = Path(__file__).parent.parent / "experiments" / "toy_pinn_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {save_path}")
    plt.show()


if __name__ == "__main__":

    print("=" * 40)
    print("  aeras Toy PINN: 1D Advection-Diffusion")
    print("=" * 40)
    print("  If this doesn't converge, don't try Delhi NCR.")

    model, history = train_toy_pinn()


    final_pde = history[-1]["pde"]
    final_ic = history[-1]["ic"]

    if final_pde < 1e-4 and final_ic < 1e-4:
        print("\n>>> PASS: Toy PINN converged! PDE and IC losses < 1e-4")
        print(">>> Safe to proceed with Delhi NCR model.")
    else:
        print(f"\n>>> WARN: Losses still high (PDE={final_pde:.4e}, IC={final_ic:.4e})")
        print(">>> Try more epochs or adjust learning rate before proceeding.")

    visualize_results(model)
