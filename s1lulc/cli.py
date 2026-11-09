"""Command-line interface.

    s1lulc export        --project MY-EE-PROJECT [--regions a,b] [--dry-run]
    s1lulc build-dataset --src DRIVE_FOLDER --out DATASET_DIR
    s1lulc train         --data DATASET_DIR/data [--variant all|v1_r34_scratch,...]
    s1lulc evaluate      --bundle outputs/train/<variant>/bundle --data DATASET_DIR/data
                         --out outputs/eval/<variant>
    s1lulc predict       --lat 48.14 --lon 11.42 --start 2025-10 --bundle BUNDLE
                         --project MY-EE-PROJECT --out outputs/munich
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config


def _cmd_export(args: argparse.Namespace) -> int:
    import ee

    from .gee.export import export_regions

    cfg = load_config(args.config_dir)
    regions = cfg.regions
    if args.regions:
        wanted = set(args.regions.split(","))
        unknown = wanted - {r.name for r in regions}
        if unknown:
            print(f"unknown regions: {sorted(unknown)}", file=sys.stderr)
            return 2
        regions = [r for r in regions if r.name in wanted]
    ee.Initialize(project=args.project)
    recs = export_regions(cfg, folder=args.folder, regions=regions,
                          manifest_path=Path(args.manifest), dry_run=args.dry_run,
                          force=args.force)
    print(f"{'planned' if args.dry_run else 'started'} {len(recs)} export(s); "
          f"manifest: {args.manifest}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from .data.build import build_dataset

    meta = build_dataset(Path(args.src), Path(args.out), load_config(args.config_dir),
                         seed=args.seed)
    bad = [q for q in meta["qc"] if not q["ok"]]
    print(f"{len(meta['regions'])} regions written to {Path(args.out) / 'data'}; "
          f"{len(bad)} excluded")
    for q in bad:
        print(f"  excluded {q['region']}: {'; '.join(q['errors'])}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import load_train_config, train_variant

    cfg = load_train_config(args.train_config)
    variants = list(cfg["variants"]) if args.variant == "all" else args.variant.split(",")
    for v in variants:
        if v not in cfg["variants"]:
            print(f"unknown variant {v}; choose from {list(cfg['variants'])}", file=sys.stderr)
            return 2
        path = train_variant(args.data, args.out, v, cfg, device=args.device,
                             resume=not args.no_resume, epochs=args.epochs)
        print(f"best bundle: {path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_bundle

    res = evaluate_bundle(args.bundle, args.data, args.out, device=args.device)
    for s, d in res["splits"].items():
        print(f"{s:9s} OA {d['raw']['overall_accuracy']:.3f}  mIoU {d['raw']['mean_iou']:.3f}"
              f"  (smoothed mIoU {d['smoothed']['mean_iou']:.3f})")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .prototype.run import run_prototype
    from .prototype.window import InputError

    try:
        res = run_prototype(args.lat, args.lon, args.start, args.bundle, args.out,
                            project=args.project, cache_dir=args.cache,
                            dynamic_world=not args.no_dynamic_world,
                            config_dir=args.config_dir, device=args.device,
                            smoothing_window=args.smoothing_window)
    except InputError as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"file not found: {e} (check --bundle: it must contain model.pt and bundle.json)",
              file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - last-resort message for Earth Engine/auth problems
        if type(e).__module__.startswith("ee"):
            print(f"Earth Engine error: {e}\nCheck the project ID, that the project is "
                  "registered for Earth Engine, and that you are authenticated "
                  "(earthengine authenticate / ee.Authenticate()).", file=sys.stderr)
            return 3
        raise
    raw = res["series"]["raw"]
    print(f"\n{'class':16s} {'first %':>8s} {'last %':>8s} {'change pp':>10s}  significant")
    for c, n in enumerate(res["class_names"]):
        sig = raw.get("significant_total")
        print(f"{n:16s} {raw['shares'][raw['first_month'], c]:8.2f} "
              f"{raw['shares'][raw['last_month'], c]:8.2f} {raw['total'][c]:+10.2f}  "
              f"{'' if sig is None else ('yes' if sig[c] else 'no')}")
    for w in res["warnings"]:
        print(f"note: {w['kind']}: {w['text']}")
    print(f"\nreport: {Path(args.out) / 'report.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="s1lulc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", default=None, help="directory with the YAML configs")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"s1lulc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="export training regions from Earth Engine to Drive")
    e.add_argument("--project", required=True, help="Earth Engine Cloud project ID")
    e.add_argument("--folder", default="lulc_s1_data", help="Google Drive folder")
    e.add_argument("--regions", default="", help="comma-separated subset (default: all)")
    e.add_argument("--manifest", default="export_manifest.json")
    e.add_argument("--dry-run", action="store_true", help="compute grids, start nothing")
    e.add_argument("--force", action="store_true", help="export even if a task exists")
    e.set_defaults(func=_cmd_export)

    b = sub.add_parser("build-dataset", help="exported GeoTIFFs -> training dataset")
    b.add_argument("--src", required=True, help="folder with the exported region GeoTIFFs")
    b.add_argument("--out", required=True, help="output folder (data/ and reports/)")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=_cmd_build)

    t = sub.add_parser("train", help="train one, several (a,b) or all variants")
    t.add_argument("--data", required=True, help="dataset data/ folder")
    t.add_argument("--out", default="outputs/train")
    t.add_argument("--variant", default="all")
    t.add_argument("--train-config", default=None)
    t.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    t.add_argument("--device", default=None)
    t.add_argument("--no-resume", action="store_true")
    t.set_defaults(func=_cmd_train)

    v = sub.add_parser("evaluate", help="evaluate a model bundle on val/test/OOD regions")
    v.add_argument("--bundle", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--device", default=None)
    v.set_defaults(func=_cmd_evaluate)

    pr = sub.add_parser("predict", help="prototype: 12 monthly maps + change report")
    pr.add_argument("--lat", type=float, required=True)
    pr.add_argument("--lon", type=float, required=True)
    pr.add_argument("--start", required=True, help="first month, YYYY-MM")
    pr.add_argument("--bundle", required=True, help="trained model bundle folder")
    pr.add_argument("--project", required=True, help="Earth Engine Cloud project ID")
    pr.add_argument("--out", default="outputs/prediction")
    pr.add_argument("--cache", default="cache", help="download cache folder ('' = off)")
    pr.add_argument("--no-dynamic-world", action="store_true")
    pr.add_argument("--smoothing-window", type=int, default=3)
    pr.add_argument("--device", default=None)
    pr.set_defaults(func=_cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
