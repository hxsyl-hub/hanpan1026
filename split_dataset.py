"""Stratified 7:2:1 train/val/test split. Pure stdlib, run locally."""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "splits"
RATIOS = (0.7, 0.2, 0.1)
IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def main():
    classes = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "classes.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(SEED)
    by_split = {"train": [], "val": [], "test": []}

    for cls in classes:
        files = sorted(p for p in (DATA_DIR / cls).iterdir()
                       if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        n = len(files)
        n_train = round(n * RATIOS[0])
        n_val = round(n * RATIOS[1])
        n_test = n - n_train - n_val
        splits = [("train", files[:n_train]),
                  ("val", files[n_train:n_train + n_val]),
                  ("test", files[n_train + n_val:])]
        print(f"{cls:22s} total={n:5d}  train={n_train} val={n_val} test={n_test}")
        for split, items in splits:
            for p in items:
                by_split[split].append((p.relative_to(DATA_DIR).as_posix(),
                                        class_to_idx[cls]))

    for split, rows in by_split.items():
        rows.sort()
        with open(OUT_DIR / f"{split}.txt", "w", encoding="utf-8") as f:
            for rel, idx in rows:
                f.write(f"{rel}\t{idx}\n")
        print(f"{split}: {len(rows)} images -> splits/{split}.txt")

    total = sum(len(v) for v in by_split.values())
    assert total == 3436, f"expected 3436, got {total}"
    print("Done.")


if __name__ == "__main__":
    main()
