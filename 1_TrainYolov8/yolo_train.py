"""Train YOLOv8n from the command line with the notebook's settings.

Mirrors the training cell in yolo_train.ipynb: COCO yolov8n.pt on
dataset/data.yaml at 640 px, batch 32, seed 0, for 100 epochs. Only batch and
workers are written down in the notebook; epochs, imgsz and seed come from the
Ultralytics defaults, so this script states them explicitly instead. The
notebook sets workers=0, this script defaults to four data-loader workers.
Both clear runs/train before training and save the new results there, without
backing up the previous weights or logs.

--optimizer stays "auto" by default, which lets Ultralytics choose both the
optimizer and the learning rate (AdamW with lr0 = 0.001667 on this 2-class
dataset, decaying to 1.7e-5). Naming an optimizer requires --lr0 as well,
because the Ultralytics default of 0.01 is an SGD-scale value.
"""

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent


def clear_training_output():
    """Delete only this script's runs/train directory when starting training."""
    train_dir = SCRIPT_DIR / "runs" / "train"
    # Reject links/junctions that redirect cleanup outside the intended directory.
    if train_dir.resolve() != train_dir:
        raise ValueError(f"Refusing to clear a redirected training directory: {train_dir}")
    if train_dir.exists():
        if not train_dir.is_dir():
            raise NotADirectoryError(f"Training output is not a directory: {train_dir}")
        print(f"Clearing previous training results: {train_dir}")
        shutil.rmtree(train_dir)
    return train_dir


def main():
    ap = argparse.ArgumentParser(
        description="Train YOLOv8n; clear runs/train and save the new run there (no backup)."
    )
    ap.add_argument("--model", default=str(SCRIPT_DIR / "yolov8n.pt"),
                    help="starting weights (yolov8n.pt = COCO, or the previous best.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32,
                    help="Training images per batch (default: 32)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="auto",
                    help="'auto' lets Ultralytics pick the optimizer and the learning "
                         "rate; anything else (AdamW, SGD, ...) requires --lr0")
    ap.add_argument("--lr0", type=float, default=None,
                    help="initial learning rate; ignored by Ultralytics when the "
                         "optimizer is 'auto', so it is only accepted alongside --optimizer")
    args = ap.parse_args()

    if args.optimizer == "auto" and args.lr0 is not None:
        ap.error("--lr0 is discarded when --optimizer=auto; pass e.g. "
                 "--optimizer AdamW --lr0 0.001 to actually set the learning rate")
    if args.optimizer != "auto" and args.lr0 is None:
        ap.error(f"--optimizer {args.optimizer} needs an explicit --lr0 "
                 "(0.001 for Adam/AdamW, 0.01 for SGD); the Ultralytics default of "
                 "0.01 is an SGD-scale value")

    train_args = {"optimizer": args.optimizer}
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0

    model = YOLO(args.model)
    train_dir = clear_training_output()
    model.train(
        data=str(SCRIPT_DIR / "dataset" / "data.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        workers=args.workers,
        project=str(train_dir.parent),
        name=train_dir.name,
        exist_ok=True,
        **train_args,
    )
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    # Keep the platform's default process start method; Windows uses spawn.
    main()
