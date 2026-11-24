"""Turn ddddocr's pretrained captcha model into a model folder the server can load.

ddddocr (https://github.com/sml2h3/ddddocr, MIT licence) ships general captcha OCR models
trained on many captcha styles, with a character set of ~8,200 entries (Chinese, Latin
letters, digits, symbols). Serving it gives you an answer for captcha styles you have not
trained a model for yet. Measure it on each customer's samples (training/evaluate.py)
before relying on it.

Needs the ddddocr package on the machine that runs this script (not on the server):

  pip install "ddddocr>=1.6"
  python -m training.import_ddddocr --out models/ddddocr-v1
  python -m training.import_ddddocr --variant beta --allowed letters+digits --out models/ddddocr-beta-v1

--allowed limits answers to a set of characters. That helps a lot when you know what a
captcha can contain: the model can't confuse "o" with a Chinese character it will never see.
Shortcuts: letters (a-z A-Z), lower (a-z), upper (A-Z), digits (0-9), joined with "+";
anything else is taken as the literal characters. --lowercase lowercases answers, for sites
that ignore case.

Pixels are scaled to -1..1 as ddddocr 1.4 did. ddddocr 1.6 scales to 0..1; on our test sets
-1..1 was as good or better (default model, letters only: 50.3% vs 45.1% on captcha-library
captchas, 92.7% vs 92.8% on simple ones). --no-normalize matches ddddocr 1.6 exactly.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path

SHORTCUTS = {
    "letters": string.ascii_letters,
    "lower": string.ascii_lowercase,
    "upper": string.ascii_uppercase,
    "digits": string.digits,
}

VARIANTS = {
    # file in the ddddocr package, CharsetManager method for its classes
    "old": ("common_old.onnx", "_get_old_charset"),   # ddddocr's default model
    "beta": ("common.onnx", "_get_beta_charset"),     # DdddOcr(beta=True)
}


def parse_allowed(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split("+")
    if all(p in SHORTCUTS for p in parts):
        return "".join(dict.fromkeys("".join(SHORTCUTS[p] for p in parts)))
    return "".join(dict.fromkeys(value))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="model folder to create, e.g. models/ddddocr-v1")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="old")
    p.add_argument("--name", help="model name shown by /v1/health (default: the folder name)")
    p.add_argument("--allowed", help="characters answers may contain, e.g. letters+digits")
    p.add_argument("--lowercase", action="store_true", help="lowercase every answer")
    p.add_argument("--no-normalize", action="store_true",
                   help="feed pixels as 0..1 like ddddocr 1.6 does, instead of -1..1 like ddddocr 1.4")
    args = p.parse_args(argv)

    try:
        import ddddocr
        from ddddocr.models.charset_manager import CharsetManager
    except ImportError:
        raise SystemExit('ddddocr is not installed. Run: pip install "ddddocr>=1.6"') from None

    package_dir = Path(ddddocr.__file__).parent
    onnx_file, charset_method = VARIANTS[args.variant]
    classes = getattr(CharsetManager(), charset_method)()
    if classes[0] != "":
        raise SystemExit(f"Unexpected ddddocr charset: class 0 is {classes[0]!r}, expected the blank ''")

    dist = importlib.metadata.distribution("ddddocr")
    allowed = parse_allowed(args.allowed)
    if allowed:
        missing = set(allowed) - set(classes)
        if missing:
            raise SystemExit(f"ddddocr's model can't output: {''.join(sorted(missing))}")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(package_dir / onnx_file, args.out / "model.onnx")

    spec = {
        "name": args.name or args.out.name,
        "input": {
            "width": None, "height": 64, "channels": 1, "layout": "NCHW", "resample": "lanczos",
            **({} if args.no_normalize else {"mean": 0.5, "std": 0.5}),
        },
        "decoder": "ctc",
        "output_layout": "TNC",
        "charset": classes[1:],  # class 0 is the CTC blank
        "allowed": allowed,
        "lowercase": args.lowercase,
        "source": {
            "project": "ddddocr",
            "url": "https://github.com/sml2h3/ddddocr",
            "version": dist.version,
            "file": onnx_file,
            "licence": "MIT, see LICENSE-ddddocr.txt",
        },
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "model.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")

    licence = dist.metadata.get("License") or "MIT License"
    (args.out / "LICENSE-ddddocr.txt").write_text(
        f"model.onnx and the charset in model.json come from ddddocr {dist.version}\n"
        f"https://github.com/sml2h3/ddddocr\n\n{licence}\n"
    )
    limit = f"{len(allowed)} allowed characters" if allowed else f"all {len(classes) - 1} characters"
    print(f"Wrote {args.out} (ddddocr {dist.version}, {args.variant} model, {limit})")


if __name__ == "__main__":
    main()
