"""Generate reference data, fit a method and export evaluation metrics."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "generate", "train", "evaluate"))
    parser.add_argument("--config", type=Path, help="JSON settings file")
    parser.add_argument("--method", help="MCINP, Global-Q, MCP, FNO, DeepONet or POD-linear")
    parser.add_argument("--root", type=Path, default=Path("runs/example"), help="output directory")
    parser.add_argument("--dimension", type=int, choices=(1, 2), help="override the configuration dimension")
    parser.add_argument("--seed", type=int, help="override the training seed")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="override the compute device")
    args = parser.parse_args()

    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    from mcinp.config import DEFAULTS_PATH, load_config
    from mcinp.runtime import pipeline
    from mcinp.runtime.common import write
    import torch

    try:
        config = load_config(args.config or DEFAULTS_PATH, args.method,
                             args.dimension, args.seed, args.device)
    except ValueError as error:
        parser.error(str(error))
    if args.stage != "generate" and config["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Set device to cpu or use a CUDA-enabled PyTorch environment.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage_args = SimpleNamespace(stop_after_case=None, stop_after_split=None,
                                 stop_after_epoch=None, split=None)
    if args.stage in ("run", "generate"):
        pipeline.generate(stage_args, config, args.root)
    if args.stage in ("run", "train"):
        pipeline.train(stage_args, config, args.root)
        write(args.root / "slots" / pipeline.slot(config) / "settings.json", config)
    if args.stage in ("run", "evaluate"):
        pipeline.evaluate(stage_args, config, args.root)
        print(pipeline.export_metrics(config, args.root))


if __name__ == "__main__":
    main()
