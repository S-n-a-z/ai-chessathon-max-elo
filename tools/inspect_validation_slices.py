"""Compare original networks on the same validation rows, including near-equal positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def predict(
    net: dict[str, np.ndarray], data: dict[str, np.ndarray], rows: np.ndarray
) -> np.ndarray:
    accumulators = []
    for color in ("white", "black"):
        features = data[f"{color}_features"][rows]
        active = np.arange(32)[None, :] < data[f"{color}_lengths"][rows, None]
        embedded = net["feature_weights"][features].astype(np.int32)
        embedded *= active[:, :, None]
        summed = embedded.sum(axis=1, dtype=np.int32) + net["feature_bias"]
        accumulators.append(np.clip(summed, 0, net["activation_clip"]))
    head = net["output_weights"].astype(np.float32) * net["output_scale"]
    difference = (accumulators[0] - accumulators[1]).astype(np.float32) * net["feature_scale"]
    return (difference @ head + net["tempo"] * data["side_to_move"][rows]) * net["target_scale_cp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    networks = []
    for path in args.weights:
        with np.load(path, allow_pickle=False) as archive:
            networks.append({key: archive[key] for key in archive.files if key != "metadata"})
    counts = np.zeros(16, dtype=np.int64)
    errors = np.zeros((len(networks) + 1, 16), dtype=np.float64)
    for shard in manifest["shards"]:
        with np.load(args.cache / shard["file"], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        all_rows = np.flatnonzero(data["split"] == 1)
        for offset in range(0, len(all_rows), 256):
            rows = all_rows[offset : offset + 256]
            baseline = data["baseline_cp"][rows]
            teacher = data["teacher_cp"][rows]
            bucket = data["bucket"][rows]
            forecasts = [baseline] + [baseline + predict(net, data, rows) for net in networks]
            for column in range(16):
                mask = np.ones(len(rows), dtype=bool) if column == 15 else bucket == column
                counts[column] += np.count_nonzero(mask)
                for model, forecast in enumerate(forecasts):
                    errors[model, column] += np.abs(forecast[mask] - teacher[mask]).sum()
    report = {"cache": str(args.cache), "split": "validation only", "models": []}
    for model, label in enumerate(["classical"] + [str(path) for path in args.weights]):
        report["models"].append(
            {
                "model": label,
                "all_mae_cp": errors[model, 15] / counts[15],
                "by_bucket": [
                    {
                        "bucket": i,
                        "positions": int(counts[i]),
                        "mae_cp": errors[model, i] / max(1, counts[i]),
                    }
                    for i in range(15)
                ],
            }
        )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for model in report["models"]:
        print(f"{model['model']}: {model['all_mae_cp']:.2f} cp", flush=True)


if __name__ == "__main__":
    main()
