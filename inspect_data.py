"""Step 1: Inspect the Waterbirds dataset BEFORE training.

Run this first in a Kaggle notebook cell. It will:
    1. Confirm WILDS can find the dataset at your DATA_ROOT.
    2. Print the size of each split.
    3. Print the count of each of the 4 subgroups per split
       (this is the bias distribution we expect to be highly skewed).
    4. Show one sample image filename from each subgroup so you can
       visually verify the labels.
    5. Verify the dataset returns (image, label, metadata) triples.

Expected output (numbers approximate):
    TRAIN: landbird-on-land ~3491, landbird-on-water ~184,
           waterbird-on-land ~56,  waterbird-on-water ~1064
    TEST:  landbird-on-land ~467,  landbird-on-water ~642,
           waterbird-on-land ~634, waterbird-on-water ~3555
    The minority groups in TRAIN (~240 images out of ~4800) are the
    cause of the spurious-correlation failure.
"""
from __future__ import annotations

from collections import Counter

from erm.config import DATA_ROOT, GROUP_NAMES, NUM_GROUPS
from erm.data import group_id_from_metadata, load_waterbirds


def main() -> None:
    print(f"[config] DATA_ROOT = {DATA_ROOT}")
    print(f"[config] expects to find: {DATA_ROOT / 'waterbirds_v1.0'}")

    # ---------- 1) Sanity check on directory structure ----------
    expected = DATA_ROOT / "waterbirds_v1.0"
    if not expected.exists():
        raise FileNotFoundError(
            f"\n\nCould not find {expected}.\n"
            f"Either move/extract your dataset so that\n"
            f"  {DATA_ROOT}/waterbirds_v1.0/metadata.csv\n"
            f"exists, OR set the WILDS_DATA_DIR env var to the parent\n"
            f"directory that contains waterbirds_v1.0/.\n"
        )
    metadata_csv = expected / "metadata.csv"
    print(f"[ok] found {expected}")
    print(f"[ok] metadata.csv exists: {metadata_csv.exists()}")

    # ---------- 2) Per-split subgroup counts ----------
    for split in ("train", "val", "test"):
        subset = load_waterbirds(split)
        counts = Counter()
        for i in range(len(subset)):
            # Use the underlying dataset (no transform) to read raw metadata.
            _, _, metadata = subset.dataset[i]
            gid = group_id_from_metadata(metadata)
            counts[GROUP_NAMES[gid]] += 1

        total = sum(counts.values())
        print(f"\n=== {split.upper():>5s}  (n = {total}) ===")
        for gname in GROUP_NAMES:
            cnt = counts[gname]
            pct = 100 * cnt / total if total else 0
            tag = "MAJORITY" if cnt / total > 0.30 else "MINORITY"
            print(f"  {gname:<22s}  {cnt:5d}  ({pct:5.1f}%)  [{tag}]")

    # ---------- 3) Verify sample structure ----------
    print("\n=== Sample triple (transformed) ===")
    subset = load_waterbirds("train")
    x, y, metadata = subset[0]
    print(f"  image type      : {type(x).__name__}, shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"  label           : {y}  (0=landbird, 1=waterbird)")
    print(f"  metadata shape  : {tuple(metadata.shape)}")
    print(f"  metadata values : {metadata.tolist()}")
    gid = group_id_from_metadata(metadata.numpy() if hasattr(metadata, "numpy") else metadata)
    print(f"  -> belongs to group {gid} = {GROUP_NAMES[gid]}")


if __name__ == "__main__":
    main()
