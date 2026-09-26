import argparse
import importlib
import inspect
import sys
from pathlib import Path


def _kwargs_for(fn, args):
    """Retourne les kwargs supplémentaires supportés par fn."""
    sig    = inspect.signature(fn)
    params = sig.parameters
    extra  = {}
    if "mode" in params and args.mode is not None:
        extra["mode"] = args.mode
    if "resize" in params and args.resize is not None:
        extra["resize"] = args.resize
    return extra


def main():
    parser = argparse.ArgumentParser(description="cam-effects — apply visual effects to video")
    parser.add_argument("--effect",  required=True, help="Effect name (e.g. blob_art)")
    parser.add_argument("--input",   help="Input video file (omit for webcam)")
    parser.add_argument("--output",  help="Output file (file mode) or /dev/videoX (virtual cam mode)")
    parser.add_argument("--cam",     type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--virtual", action="store_true", help="Output to v4l2loopback virtual camera")
    parser.add_argument("--mode",    help="Visual mode (blob_art: default | loupe | vitrail | voronoi)")
    parser.add_argument("--resize",  type=int, help="Resize height before processing (ex: 720). Speeds up file rendering.")
    args = parser.parse_args()

    try:
        module = importlib.import_module(f"effects.{args.effect}")
    except ModuleNotFoundError:
        print(f"Error: effect '{args.effect}' not found in effects/")
        sys.exit(1)

    if args.input:
        # mode fichier
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file '{args.input}' not found")
            sys.exit(1)
        output_path = Path(args.output) if args.output else Path("output") / f"{args.effect}_{input_path.name}"
        extra = _kwargs_for(module.run, args)
        mode_tag = f" [{args.mode}]" if args.mode else ""
        print(f"[file] {args.effect}{mode_tag} : {input_path} → {output_path}")
        module.run(str(input_path), str(output_path), **extra)
        print("Done.")
    else:
        # mode live webcam
        virtual_dev = args.output if args.virtual else None
        extra = _kwargs_for(module.run_live, args)
        print(f"[live] {args.effect} — webcam {args.cam} {'→ ' + virtual_dev if virtual_dev else '(fenêtre)'}")
        print("Appuie sur Q pour quitter.")
        module.run_live(args.cam, virtual_dev, **extra)


if __name__ == "__main__":
    main()
