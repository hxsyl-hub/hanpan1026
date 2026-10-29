"""Train baseline models (no pretrained weights) on the solder-joint dataset.

Usage:
    python train.py                      # train all 5 models sequentially
    python train.py --model resnet34     # train one model
    python train.py --device cuda        # force device
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).parent
MODEL_NAMES = ["vgg16", "googlenet", "resnet34", "resnet50", "mobilenet_v2"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SolderDataset(Dataset):
    def __init__(self, list_file: Path, data_dir: Path, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                rel, idx = line.strip().split("\t")
                self.samples.append((rel, int(idx)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, idx = self.samples[i]
        img = Image.open(self.data_dir / rel).convert("RGB")
        return self.transform(img), idx


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "googlenet":
        m = models.googlenet(weights=None, init_weights=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


def forward_logits(model, images):
    out = model(images)
    if not torch.is_tensor(out):  # googlenet in train mode returns aux outputs
        out = out.logits
    return out


def make_loaders(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    splits_dir = ROOT / "splits"
    data_dir = Path(args.data_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = SolderDataset(splits_dir / f"{split}.txt", data_dir, transform)
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers, pin_memory=True)
    with open(splits_dir / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    return loaders, classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = forward_logits(model, images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_model(name, args, loaders, classes, device):
    out_dir = ROOT / args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(classes)

    torch.manual_seed(args.seed)
    model = build_model(name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = {"model": name,
               "config": {"batch_size": args.batch_size, "lr": args.lr,
                          "epochs": args.epochs, "optimizer": "AdamW",
                          "seed": args.seed, "img_size": 224},
               "classes": classes, "epochs": []}

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
             "test_loss", "test_acc", "seconds"])

    best_val_acc = -1.0
    best_epoch = 0
    print(f"\n===== {name}  params={count_params(model):.2f}M =====")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
        dt = time.time() - t0

        history["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "test_loss": test_loss, "test_acc": test_acc, "seconds": dt})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                 f"{val_loss:.6f}", f"{val_acc:.6f}",
                 f"{test_loss:.6f}", f"{test_acc:.6f}", f"{dt:.1f}"])

        star = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "model_name": name, "classes": classes,
                        "best_val_acc": val_acc, "best_epoch": epoch},
                       out_dir / "best.pth")
            star = "  *best*"
        print(f"[{name}] epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}  "
              f"test_acc={test_acc:.4f}  {dt:.1f}s{star}")
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[{name}] done. best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return {"model": name, "params_M": round(count_params(model), 2),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "final_test_acc": history["epochs"][-1]["test_acc"],
            "total_seconds": round(sum(e["seconds"] for e in history["epochs"]), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--models", default=",".join(MODEL_NAMES),
                   help="comma-separated model names")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    loaders, classes = make_loaders(args)
    print("classes:", list(classes))

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        results.append(train_model(name, args, loaders, classes, device))

    summary_path = ROOT / args.out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsummary saved to {summary_path}")
    for r in results:
        print(f"  {r['model']:14s} best_val_acc={r['best_val_acc']:.4f} "
              f"final_test_acc={r['final_test_acc']:.4f} ({r['total_seconds']}s)")


if __name__ == "__main__":
    main()
