"""Generate synthetic text captchas with labels.

Use it to bootstrap a model before you have real samples, or to pad out a small set of
real customer captchas. Output is a folder of PNGs plus labels.txt in the same
"<name> <text>" format as words_captcha/spec_train_val.txt.

  python -m training.synth --out datasets/synth-train --count 30000
  python -m training.synth --out datasets/synth-test --count 2000 --seed 2
  python -m training.synth --out datasets/digits --charset 0123456789 --min-len 4 --max-len 6
  python -m training.synth --out datasets/simple-test --style simple --count 1000 --seed 3
  python -m training.synth --out datasets/math-train --style math --width 150 --count 25000

The math style draws sums like "4 + 3 - 5" in plain sans-serif fonts on light grey with thin
lines and specks; the label is the expression without spaces ("4+3-5"). --ops picks the
operators (default "+-x": mostly plus and minus, sometimes multiply, labelled "×").
"""

from __future__ import annotations

import argparse
import random
import re
import string
from multiprocessing import Pool
from pathlib import Path

# Plain sans-serif fonts for the math style; whichever exist on this machine are used.
MATH_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def math_expression(rng: random.Random, ops: str) -> str:
    """Two or three numbers (mostly one digit) joined by operators, e.g. "4+3-5"."""
    count = rng.choice([2, 3, 3, 3])
    nums = [str(rng.randint(0, 9)) if rng.random() < 0.75 else str(rng.randint(10, 99)) for _ in range(count)]
    labels = {"+": "+", "-": "-", "x": "×", "×": "×", "*": "×"}
    choices = [labels[o] for o in ops]
    weights = [1 if c == "×" and len(choices) > 1 else 4 for c in choices]
    out = nums[0]
    for n in nums[1:]:
        out += rng.choices(choices, weights)[0] + n
    return out


def _math(text: str, width: int, height: int):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    fonts = [f for f in MATH_FONTS if Path(f).is_file()]
    bg = random.randint(226, 248)
    img = Image.new("L", (width, height), bg)
    draw = ImageDraw.Draw(img)

    glyphs = {"-": random.choice(["-", "-", "−"]), "×": random.choice(["×", "x", "*"])}
    tokens = re.findall(r"\d+|[+\-×]", text)
    shown = random.choice([" ", " ", "  ", ""]).join(glyphs.get(t, t) for t in tokens)

    def make_font(size):
        return ImageFont.truetype(random.choice(fonts), size) if fonts else ImageFont.load_default(size=size)

    size = int(height * random.uniform(0.44, 0.6))
    font = make_font(size)
    box = draw.textbbox((0, 0), shown, font=font)
    while box[2] - box[0] > width * 0.92 and size > 10:
        size -= 2
        font = make_font(size)
        box = draw.textbbox((0, 0), shown, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = max(2, (width - tw) / 2 + random.uniform(-0.08, 0.08) * width) - box[0]
    y = (height - th) / 2 + random.uniform(-0.1, 0.1) * height - box[1]

    def lines():
        for _ in range(random.randint(1, 3)):
            x1, x2 = random.uniform(-0.1, 0.5) * width, random.uniform(0.5, 1.1) * width
            draw.line([(x1, random.uniform(0, height)), (x2, random.uniform(0, height))],
                      fill=random.randint(120, 195), width=random.choice([1, 1, 2]))

    behind = random.random() < 0.5
    if behind:
        lines()
    draw.text((x, y), shown, font=font, fill=random.randint(0, 70))
    if not behind:
        lines()
    for _ in range(random.randint(30, 160)):
        draw.point((random.randrange(width), random.randrange(height)), fill=random.randint(140, 225))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.6)))
    return img.convert("RGB")


def _simple(text: str, width: int, height: int):
    """A plainer style: tilted letters, a few lines and dots, light background."""
    from PIL import Image, ImageDraw, ImageFont

    bg = tuple(random.randint(215, 255) for _ in range(3))
    img = Image.new("RGB", (width, height), bg)
    font = ImageFont.load_default(size=int(height * random.uniform(0.5, 0.65)))
    x = random.uniform(0.06, 0.14) * width
    step = (width * 0.8) / max(len(text), 1)
    for ch in text:
        ink = tuple(random.randint(0, 110) for _ in range(3))
        tile = Image.new("RGBA", (height, height), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((height * 0.2, height * 0.1), ch, font=font, fill=ink + (255,))
        tile = tile.rotate(random.uniform(-25, 25), resample=Image.BICUBIC)
        img.paste(tile, (int(x), int(random.uniform(-0.08, 0.12) * height)), tile)
        x += step * random.uniform(0.8, 1.05)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 3)):
        pts = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(2)]
        draw.line(pts, fill=tuple(random.randint(60, 180) for _ in range(3)), width=random.randint(1, 2))
    for _ in range(random.randint(20, 60)):
        px, py = random.uniform(0, width), random.uniform(0, height)
        draw.point((px, py), fill=tuple(random.randint(0, 200) for _ in range(3)))
    return img


def _make(job: tuple[int, str, int, int, str, int, str]) -> str:
    index, text, width, height, out, seed, style = job
    random.seed(seed * 1_000_003 + index)  # both styles draw their noise from `random`
    name = f"s{index:06d}"
    path = Path(out) / f"{name}.png"
    if style == "simple":
        _simple(text, width, height).save(path)
    elif style == "math":
        _math(text, width, height).save(path)
    else:
        from captcha.image import ImageCaptcha

        ImageCaptcha(width=width, height=height).write(text, str(path), format="png")
    return f"{name} {text}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=30000)
    p.add_argument("--charset", default=string.ascii_lowercase)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=5)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--style", choices=["captcha", "simple", "math"], default="captcha",
                   help="captcha: the captcha library's warped style; simple: tilted letters, lines, dots; "
                        "math: sums like 4 + 3 - 5")
    p.add_argument("--ops", default="+-x", help="math style: operators to use, from + - x")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.count):
        if args.style == "math":
            text = math_expression(rng, args.ops)
        else:
            n = rng.randint(args.min_len, args.max_len)
            text = "".join(rng.choice(args.charset) for _ in range(n))
        jobs.append((i, text, args.width, args.height, str(args.out), args.seed, args.style))

    with Pool(args.workers) as pool:
        lines = pool.map(_make, jobs, chunksize=256)
    (args.out / "labels.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} captchas to {args.out}")


if __name__ == "__main__":
    main()
