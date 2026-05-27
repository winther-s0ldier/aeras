import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, USE_AMP, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    CHECKPOINT_EVERY, CHECKPOINTS_DIR, NUM_COLLOCATION,
    CURRICULUM_DATA_ONLY_EPOCHS, CURRICULUM_PDE_RAMPUP_EPOCHS,
)
from src.models.pinn import AerasPINN, compute_pde_residual, compute_boundary_residual
from src.models.source_net import SourceNet, SourceGrid
from src.models.loss import AerasLoss
from src.data.collocation import latin_hypercube, residual_adaptive_resample


class AerasTrainer:
    def __init__(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        inverse_mode: bool = False,
        source_type: str = "net",
        use_wandb: bool = True,
    ):
        self.device = DEVICE
        self.inverse_mode = inverse_mode


        self.train_data = {k: v.to(self.device) for k, v in train_data.items()}
        self.val_data = {k: v.to(self.device) for k, v in val_data.items()}


        self.model = AerasPINN().to(self.device)
        self.source_model = None
        if inverse_mode:
            if source_type == "grid":
                self.source_model = SourceGrid().to(self.device)
            else:
                self.source_model = SourceNet().to(self.device)


        self.loss_fn = AerasLoss(adaptive=True).to(self.device)


        params = list(self.model.parameters())
        if self.source_model is not None:
            params += list(self.source_model.parameters())
        self.optimizer = Adam(params, lr=LEARNING_RATE)


        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2, eta_min=1e-6
        )


        self.use_amp = USE_AMP and self.device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None


        colloc_np = latin_hypercube(NUM_COLLOCATION, dim=3)
        self.collocation_points = torch.tensor(colloc_np, dtype=torch.float32).to(self.device)


        t_train = self.train_data["inputs"][:, 2]
        self._sorted_train_times, self._sort_order = t_train.sort()


        self.use_wandb = use_wandb
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "pde_residual": []}

        if use_wandb:
            try:
                import wandb
                from src.config import WANDB_PROJECT
                wandb.init(
                    project=WANDB_PROJECT,
                    config={
                        "hidden_layers": self.model.network.__len__() // 2,
                        "hidden_dim": 128,
                        "lr": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                        "inverse_mode": inverse_mode,
                        "source_type": source_type,
                    }
                )
            except Exception as e:
                print(f"[WARN] WandB init failed: {e}. Logging locally only.")
                self.use_wandb = False

        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.source_model:
            print(f"[TRAINER] Source params: {sum(p.numel() for p in self.source_model.parameters()):,}")
        print(f"[TRAINER] Collocation points: {len(self.collocation_points):,}")
        print(f"[TRAINER] Mixed precision: {self.use_amp}")

    def _sample_collocation_batch(self, batch_size: int) -> torch.Tensor:
        assert self.collocation_points.shape[1] == 3, (
            "collocation_points must be [N, 3]. "
            "If RAR added wind columns, that's a bug — see RAR code."
        )

        idx = torch.randint(0, len(self.collocation_points), (batch_size,))
        points = self.collocation_points[idx]
        t_colloc = points[:, 2]


        pos = torch.searchsorted(
            self._sorted_train_times.contiguous(), t_colloc.contiguous()
        ).clamp(0, len(self._sorted_train_times) - 1)

        left_pos = (pos - 1).clamp(0)
        left_dist  = (t_colloc - self._sorted_train_times[left_pos]).abs()
        right_dist = (t_colloc - self._sorted_train_times[pos]).abs()


        nearest_sorted_pos = torch.where(left_dist < right_dist, left_pos, pos)
        wind_idx = self._sort_order[nearest_sorted_pos]

        extra_features = self.train_data["inputs"][wind_idx, 3:]

        return torch.cat([points, extra_features], dim=1)

    def _sample_boundary_points(self, n_points: int = 256):
        n_each = n_points // 2

        x_pts = []
        y_pts = []
        for _ in range(n_each // 2):

            x_pts.append([0.0, np.random.uniform(), np.random.uniform()])
            x_pts.append([1.0, np.random.uniform(), np.random.uniform()])

            y_pts.append([np.random.uniform(), 0.0, np.random.uniform()])
            y_pts.append([np.random.uniform(), 1.0, np.random.uniform()])

        x_xyt = torch.tensor(x_pts, dtype=torch.float32, device=self.device)
        y_xyt = torch.tensor(y_pts, dtype=torch.float32, device=self.device)


        def add_features(xyt):
            wind_idx = torch.randint(0, len(self.train_data["inputs"]), (len(xyt),))
            extra_features = self.train_data["inputs"][wind_idx, 3:]
            return torch.cat([xyt, extra_features], dim=1)

        return add_features(x_xyt), add_features(y_xyt)

    def train_step(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.source_model:
            self.source_model.train()


        n_data = len(self.train_data["inputs"])
        data_idx = torch.randint(0, n_data, (min(BATCH_SIZE, n_data),))
        data_inputs = self.train_data["inputs"][data_idx]
        data_targets = self.train_data["targets"][data_idx]
        data_mask = self.train_data["mask"][data_idx] if "mask" in self.train_data else None


        colloc_inputs = self._sample_collocation_batch(BATCH_SIZE)


        bc_x_inputs, bc_y_inputs = self._sample_boundary_points(256)

        self.optimizer.zero_grad()

        amp_device = "cuda" if self.use_amp else "cpu"


        with autocast(amp_device, enabled=self.use_amp):
            C_pred = self.model(data_inputs)
            L_data = self.loss_fn.data_loss(C_pred, data_targets, data_mask)

            if "ic_inputs" in self.train_data:
                C_t0 = self.model(self.train_data["ic_inputs"])
                L_ic = self.loss_fn.ic_loss(C_t0, self.train_data["ic_targets"])
            else:
                L_ic = torch.tensor(0.0, device=self.device)


        with torch.enable_grad():


            source_term = None
            if self.source_model:
                source_term = self.source_model(colloc_inputs[:, :3].float())


            residual = compute_pde_residual(
                self.model, colloc_inputs.float(), source_term
            )
            L_pde = self.loss_fn.pde_loss(residual)


            bc_residual_x = compute_boundary_residual(
                self.model, bc_x_inputs.float(), normal_direction=0
            )
            bc_residual_y = compute_boundary_residual(
                self.model, bc_y_inputs.float(), normal_direction=1
            )
            L_bc = self.loss_fn.bc_loss(bc_residual_x) + self.loss_fn.bc_loss(bc_residual_y)


        L_sparsity = torch.tensor(0.0, device=self.device)
        if self.source_model and source_term is not None:
            L_sparsity = self.loss_fn.sparsity_loss(source_term)


        loss_dict = self.loss_fn(
            {"data": L_data, "pde": L_pde, "bc": L_bc, "ic": L_ic, "sparsity": L_sparsity},
            epoch=epoch,
            curriculum_data_epochs=CURRICULUM_DATA_ONLY_EPOCHS,
            curriculum_rampup_epochs=CURRICULUM_PDE_RAMPUP_EPOCHS,
        )


        if self.use_amp:
            self.scaler.scale(loss_dict["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()

        return loss_dict

    def validate(self) -> Dict[str, float]:
        self.model.eval()


        with torch.no_grad():
            C_pred = self.model(self.val_data["inputs"])
            val_loss = ((C_pred - self.val_data["targets"]) ** 2).mean().item()
            val_mae = (C_pred - self.val_data["targets"]).abs().mean().item()


        with torch.enable_grad():
            colloc = self._sample_collocation_batch(1024)
            residual = compute_pde_residual(self.model, colloc.float())
            physics_residual = residual.abs().mean().item()

        return {
            "val_mse": val_loss,
            "val_mae": val_mae,
            "physics_residual": physics_residual,
        }

    def train(self, epochs: int = EPOCHS, resume_from: Optional[str] = None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"[TRAIN] Resumed from epoch {start_epoch}")

        print(f"[TRAIN] Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(start_epoch, epochs):
            loss_dict = self.train_step(epoch)


            if epoch % 100 == 0:
                val_metrics = self.validate()
                elapsed = time.time() - start_time
                eps = (epoch - start_epoch + 1) / elapsed

                print(
                    f"Epoch {epoch:6d} | "
                    f"Loss: {loss_dict['total'].item():.4e} | "
                    f"Data: {loss_dict['data']:.4e} | "
                    f"PDE: {loss_dict['pde']:.4e} | "
                    f"Val MAE: {val_metrics['val_mae']:.4e} | "
                    f"Phys: {val_metrics['physics_residual']:.4e} | "
                    f"PDE_w: {loss_dict['pde_scale']:.2f} | "
                    f"{eps:.1f} ep/s"
                )

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(loss_dict["total"].item())
                self.history["val_loss"].append(val_metrics["val_mse"])
                self.history["pde_residual"].append(val_metrics["physics_residual"])

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/total_loss": loss_dict["total"].item(),
                        "train/data_loss": loss_dict["data"],
                        "train/pde_loss": loss_dict["pde"],
                        "train/bc_loss": loss_dict["bc"],
                        "train/pde_scale": loss_dict["pde_scale"],
                        "val/mse": val_metrics["val_mse"],
                        "val/mae": val_metrics["val_mae"],
                        "val/physics_residual": val_metrics["physics_residual"],
                        "physics/Dx": self.model.Dx.item(),
                        "physics/Dy": self.model.Dy.item(),
                        "physics/lambda_dep": self.model.lambda_dep.item(),
                        "physics/log_lambda_dep": self.model.log_lambda_dep.item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })


            if epoch > 0 and epoch % CHECKPOINT_EVERY == 0:
                self.save_checkpoint(f"epoch_{epoch}")


            if epoch > 0 and epoch % 5000 == 0 and epoch > CURRICULUM_DATA_ONLY_EPOCHS:
                print(f"[TRAIN] Resampling collocation points (RAR)...")
                new_points = residual_adaptive_resample(
                    self.model,
                    self.collocation_points[:, :3],
                    lambda m, p: compute_pde_residual(
                        m, torch.cat([p, torch.zeros(len(p), self.train_data["inputs"].shape[1] - 3, device=self.device)], dim=1)
                    ),
                    n_new=NUM_COLLOCATION // 4,
                    device=self.device,
                )


                keep_idx = torch.randperm(len(self.collocation_points))[:int(NUM_COLLOCATION * 0.75)]
                old_points = self.collocation_points[keep_idx]
                new_pts = new_points.to(self.device)
                self.collocation_points = torch.cat([old_points, new_pts], dim=0)
                print(f"[TRAIN] Collocation points updated: {len(self.collocation_points):,}")


        self.save_checkpoint("final")
        self._save_history()

        total_time = time.time() - start_time
        print(f"\n[TRAIN] Done! {epochs} epochs in {total_time/3600:.1f} hours")

    def save_checkpoint(self, name: str):
        from src.config import CHECKPOINT_PREFIX
        path = CHECKPOINTS_DIR / f"{CHECKPOINT_PREFIX}_{name}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
        }
        if self.source_model:
            state["source_model"] = self.source_model.state_dict()
        if self.scaler:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, path)
        print(f"[CKPT] Saved: {path}")

    def load_checkpoint(self, name: str) -> int:
        from src.config import CHECKPOINT_PREFIX
        path = CHECKPOINTS_DIR / f"{CHECKPOINT_PREFIX}_{name}.pt"
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.history = state.get("history", self.history)

        if self.source_model and "source_model" in state:
            self.source_model.load_state_dict(state["source_model"])
        if self.scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        last_epoch = self.history["epoch"][-1] if self.history["epoch"] else 0
        print(f"[CKPT] Loaded: {path} (epoch {last_epoch})")
        return last_epoch

    def _save_history(self):
        path = CHECKPOINTS_DIR / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[HISTORY] Saved to {path}")

# Backwards compatibility — remove after Phase 5
VayuPINNTrainer = AerasTrainer
