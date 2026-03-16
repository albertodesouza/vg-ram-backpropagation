#!/usr/bin/env python3
"""
Train and evaluate a multi-layer VG-RAM WNN on MNIST.

Usage:
    python train.py --config configs/mnist_tiny.yaml
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
import yaml

from vgram.annealing import TemperatureScheduler
from vgram.data import get_mnist_loaders
from vgram.network import VGRAMNetwork


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _layer_dict(lc: dict, cfg: dict) -> dict:
    """Convert a single YAML layer section into kwargs for VGRAMNetwork."""
    return {
        "num_neurons": lc["num_neurons"],
        "num_synapses": lc["num_synapses"],
        "num_entries": lc["num_entries"],
        "output_dim": lc["output_dim"],
        "tau_b": cfg["training"].get("tau_b_start", 1.0),
        "tau_a": cfg["training"].get("tau_a_start", 1.0),
        "beta": cfg["training"].get("beta_start", 1.0),
        "neuron_chunk_size": lc.get("neuron_chunk_size", 0),
        "use_grad_checkpoint": lc.get("use_grad_checkpoint", False),
    }


def build_network(cfg: dict, device: torch.device) -> VGRAMNetwork:
    net_cfg = cfg["network"]

    if "layers" in net_cfg:
        # Legacy format: explicit list of layers
        layer_cfgs = [_layer_dict(lc, cfg) for lc in net_cfg["layers"]]
    else:
        # New format: first_layer + intermediate_layers + output_layer
        layer_cfgs = [_layer_dict(net_cfg["first_layer"], cfg)]

        inter = net_cfg.get("intermediate_layers", {})
        count = inter.get("count", 0)
        for _ in range(count):
            layer_cfgs.append(_layer_dict(inter, cfg))

        layer_cfgs.append(_layer_dict(net_cfg["output_layer"], cfg))

    num_layers = len(layer_cfgs)
    print(f"Network: {num_layers} layer(s)")

    net = VGRAMNetwork(
        layer_configs=layer_cfgs,
        input_size=net_cfg.get("input_size", 784),
        num_classes=net_cfg.get("num_classes", 10),
    )
    return net.to(device)


@torch.no_grad()
def evaluate(
    net: VGRAMNetwork,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy) on *loader*."""
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def _init_wandb(cfg: dict, config_path: str) -> bool:
    """Initialise W&B if enabled in config. Returns True if active."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False

    import wandb

    wandb.init(
        project=wandb_cfg.get("project", "vgram-wnn"),
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes"),
        config=cfg,
    )
    return True


def _log_wandb(metrics: dict, step: int) -> None:
    import wandb
    wandb.log(metrics, step=step)


def train(cfg: dict, config_path: str) -> None:
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)

    device_name = cfg["training"].get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device}")

    use_wandb = _init_wandb(cfg, config_path)

    # --- Data ---
    train_loader, val_loader, test_loader = get_mnist_loaders(
        cfg["data"], data_dir="./data",
    )
    print(
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader) if val_loader else 0}, "
        f"Test batches: {len(test_loader)}"
    )

    # --- Network ---
    net = build_network(cfg, device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Trainable parameters: {num_params:,}")

    if use_wandb:
        import wandb
        wandb.config.update({"num_params": num_params, "device": str(device)})

    # --- Optimiser ---
    train_cfg = cfg["training"]
    lr = train_cfg.get("lr", 0.01)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # --- Temperature scheduler ---
    epochs = train_cfg.get("epochs", 50)
    temp_scheduler = TemperatureScheduler(
        network=net,
        total_epochs=epochs,
        tau_b_start=train_cfg.get("tau_b_start", 1.0),
        tau_b_end=train_cfg.get("tau_b_end", 0.1),
        tau_a_start=train_cfg.get("tau_a_start", 1.0),
        tau_a_end=train_cfg.get("tau_a_end", 0.1),
        beta_start=train_cfg.get("beta_start", 1.0),
        beta_end=train_cfg.get("beta_end", 10.0),
    )

    lambda_bin_mem = train_cfg.get("lambda_bin_mem", 0.001)
    lambda_bin_out = train_cfg.get("lambda_bin_out", 0.001)

    # --- Training loop ---
    print(f"\n{'Epoch':>5} {'Loss':>10} {'RegLoss':>10} {'TrainAcc':>10} "
          f"{'ValAcc':>10} {'tau_b':>8} {'tau_a':>8} {'beta':>8} {'Time':>7}")
    print("-" * 90)

    for epoch in range(epochs):
        temps = temp_scheduler.step(epoch)
        net.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = net(images)
            ce_loss = F.cross_entropy(logits, labels)
            reg_loss = net.regularization_loss(lambda_bin_mem, lambda_bin_out)
            loss = ce_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += ce_loss.item() * labels.size(0)
            epoch_correct += (logits.detach().argmax(dim=-1) == labels).sum().item()
            epoch_total += labels.size(0)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        reg_val = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss

        val_acc = None
        val_loss = None
        val_acc_str = "   —"
        if val_loader is not None:
            val_loss, val_acc = evaluate(net, val_loader, device)
            val_acc_str = f"{val_acc:10.4f}"

        print(
            f"{epoch + 1:5d} {train_loss:10.4f} {reg_val:10.6f} "
            f"{train_acc:10.4f} {val_acc_str:>10} "
            f"{temps['tau_b']:8.4f} {temps['tau_a']:8.4f} "
            f"{temps['beta']:8.4f} {elapsed:6.1f}s"
        )

        if use_wandb:
            wb_metrics = {
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/reg_loss": reg_val,
                "schedule/tau_b": temps["tau_b"],
                "schedule/tau_a": temps["tau_a"],
                "schedule/beta": temps["beta"],
                "epoch_time_s": elapsed,
            }
            if val_acc is not None:
                wb_metrics["val/loss"] = val_loss
                wb_metrics["val/accuracy"] = val_acc
            _log_wandb(wb_metrics, step=epoch + 1)

    # --- Final evaluation ---
    print("\n" + "=" * 50)
    test_loss, test_acc = evaluate(net, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    train_loss_final, train_acc_final = evaluate(net, train_loader, device)
    print(f"Train Loss: {train_loss_final:.4f}  |  Train Accuracy: {train_acc_final:.4f}")
    print("=" * 50)

    if use_wandb:
        import wandb
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "final_train/loss": train_loss_final,
            "final_train/accuracy": train_acc_final,
        })
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VG-RAM WNN on MNIST")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
