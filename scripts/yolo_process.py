import argparse
import os
import shutil

import pandas as pd


def copy_split(csv_path, raw_root, out_root, split_name):
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        src = os.path.join(raw_root, os.path.relpath(r.Filename, raw_root))
        dst_dir = os.path.join(out_root, split_name, str(r.ClassId))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-root", default="data/raw", help="where Train/ Test/ folders live"
    )
    p.add_argument(
        "--csv-dir", default="data/processed", help="where train.csv & test.csv live"
    )
    p.add_argument("--out-root", default="data/traffic_signs", help="YOLO data dir")
    args = p.parse_args()

    # train & val (if you have a val split) & test
    copy_split(
        os.path.join(args.csv_dir, "train.csv"), args.raw_root, args.out_root, "train"
    )
    copy_split(
        os.path.join(args.csv_dir, "test.csv"), args.raw_root, args.out_root, "val"
    )
    print("→ Done! your structure is:")
    print(f"{args.out_root}/train/<class_id>/*.png")
    print(f"{args.out_root}/val  /<class_id>/*.png")
