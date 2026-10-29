"""Smoke test: run this on the training machine (e.g. AutoDL) BEFORE real training.

    python smoke_test.py

Checks (all on CPU unless --device given):
  1. dataset integrity (8 class dirs, images readable)
  2. split files exist, no train/val/test leakage, labels valid
  3. all 5 models build + forward with correct output shape
  4. one real optimizer step (forward + backward) per model
  5. one mini train epoch + evaluation via train.py functions
  6. plot_curves produces all 6 figures from fake history
Exits non-zero if any check fails.
"""
import json
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

import train
from split_dataset import DATA_DIR, IMG_EXTS
from plot_curves import plot_all

ROOT = Path(__file__).parent
RESULTS = []


def check(name, fn):
    try:
        detail = fn() or ""
        RESULTS.append((name, True, detail))
        print(f"[PASS] {name}  {detail}")
    except Exception:
        RESULTS.append((name, False, traceback.format_exc(limit=3)))
        print(f"[FAIL] {name}")
        traceback.print_exc()


def check_dataset():
    classes = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    assert len(classes) == 8, f"expected 8 classes, got {len(classes)}: {classes}"
    total = 0
    for cls in classes:
        files = [p for p in (DATA_DIR / cls).iterdir() if p.suffix.lower() in IMG_EXTS]
        assert files, f"no images in {cls}"
        for p in files[:2]:
            img = Image.open(p)
            img.load()
        total += len(files)
    return f"8 classes, {total} images total"


def check_splits():
    splits_dir = ROOT / "splits"
    sets = {}
    for split in ["train", "val", "test"]:
        rows = (splits_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        assert rows, f"{split}.txt is empty"
        rels, labels = [], []
        for line in rows:
            rel, idx = line.split("\t")
            assert (DATA_DIR / rel).exists(), f"missing file: {rel}"
            rels.append(rel)
            labels.append(int(idx))
        sets[split] = set(rels)
        assert set(labels) <= set(range(8)), f"bad labels in {split}.txt"
        print(f"    {split}: {len(rows)} imgs, label dist={dict(sorted(Counter(labels).items()))}")
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"
    total = sum(len(s) for s in sets.values())
    assert total == 3436, f"split total {total} != 3436"
    return f"no leakage, total={total} (7:2:1)"


def get_mini_loaders():
    transform = train.transforms.Compose([
        train.transforms.Resize((224, 224)),
        train.transforms.ToTensor(),
    ])
    ds = train.SolderDataset(ROOT / "splits" / "train.txt", DATA_DIR, transform)
    tr = DataLoader(Subset(ds, range(16)), batch_size=4, shuffle=True)
    ev = DataLoader(Subset(ds, range(16, 24)), batch_size=4)
    return tr, ev


def make_step_fn(name, tr_loader):
    def fn():
        model = train.build_model(name, num_classes=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        images, targets = next(iter(tr_loader))
        model.train()
        optimizer.zero_grad()
        logits = train.forward_logits(model, images)
        assert logits.shape == (targets.size(0), 8), f"bad shape {logits.shape}"
        loss = criterion(logits, targets)
        assert torch.isfinite(loss), f"non-finite loss {loss}"
        loss.backward()
        optimizer.step()
        return f"forward+backward ok, loss={loss.item():.3f}"
    return fn


def make_mini_train_fn(name, tr_loader, ev_loader, device):
    def fn():
        torch.manual_seed(0)
        model = train.build_model(name, 8).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tl, ta = train.train_one_epoch(model, tr_loader, criterion, optimizer, device)
        vl, va = train.evaluate(model, ev_loader, criterion, device)
        assert 0 <= ta <= 1 and 0 <= va <= 1
        return f"1 mini epoch ok (train_acc={ta:.2f}, val_acc={va:.2f})"
    return fn


def check_plots():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_curves_"))
    try:
        fake = {"model": "dummy", "epochs": [
            {"epoch": e, "train_loss": 1.0 / e, "train_acc": 0.5 * e / 10,
             "val_loss": 1.1 / e, "val_acc": 0.4 * e / 10,
             "test_loss": 1.2 / e, "test_acc": 0.45 * e / 10,
             "seconds": 1.0} for e in range(1, 11)]}
        (tmp / "dummy").mkdir()
        (tmp / "dummy" / "history.json").write_text(
            json.dumps(fake), encoding="utf-8")
        plot_all(tmp)
        expected = {f"{m}_{s}.png" for m in ["acc", "loss"]
                    for s in ["train", "val", "test"]}
        got = {p.name for p in (tmp / "curves").glob("*.png")}
        assert got == expected, f"missing: {expected - got}"
        return f"{len(got)} figures ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    device = torch.device("cpu")
    check("dataset integrity", check_dataset)
    check("splits valid", check_splits)
    check("plot curves", check_plots)

    tr_loader, ev_loader = get_mini_loaders()
    for name in train.MODEL_NAMES:
        check(f"{name}: build+forward+backward", make_step_fn(name, tr_loader))
    for name in train.MODEL_NAMES:
        check(f"{name}: mini train epoch",
              make_mini_train_fn(name, tr_loader, ev_loader, device))

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n===== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed =====")
    if failed:
        sys.exit(1)
    print("All good. Ready to run: python train.py")


if __name__ == "__main__":
    main()
