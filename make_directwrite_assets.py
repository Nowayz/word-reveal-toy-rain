from pathlib import Path
import argparse
import json
import re
import subprocess
import tempfile

from PIL import Image

from make_assets import OBJECTS, slugify, sprite_outline


ASSETS = Path("assets")
MANIFEST = ASSETS / "manifest.json"
LARGE_DIR = ASSETS / "large"
SPRITE_DIR = ASSETS / "sprites"
RENDERER = Path("tools") / "render_emoji_dw.exe"


def load_symbols():
    symbols = {word: symbol for word, symbol in OBJECTS}
    next_words = Path("next_words.txt")
    if next_words.exists():
        for line in next_words.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([a-z]+(?: [a-z]+)*)(.*)$", line.strip())
            if match:
                symbols[match.group(1).strip()] = match.group(2).strip()
    return symbols


def select_items(manifest, symbols, targets):
    if not targets:
        return [item for item in manifest if item["word"] in symbols]
    wanted = {target.strip().lower() for target in targets}
    wanted |= {slugify(target) for target in wanted}
    return [
        item
        for item in manifest
        if item["word"].lower() in wanted or slugify(item["word"]) in wanted
    ]


def fit_to_canvas(source, size, padding):
    bbox = source.getbbox()
    if not bbox:
        raise RuntimeError("Rendered emoji was empty.")
    cropped = source.crop(bbox)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    target = size - padding * 2
    scale = min(target / cropped.width, target / cropped.height)
    fitted_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    fitted = cropped.resize(fitted_size, Image.Resampling.LANCZOS)
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    image.alpha_composite(fitted, (x, y))
    return image


def render_symbol(symbol, output):
    command = [str(RENDERER), symbol, str(output), "1024", "720"]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate emoji images using the DirectWrite COLR paint-tree renderer."
    )
    parser.add_argument("targets", nargs="*", help="Optional words or slugs to regenerate.")
    args = parser.parse_args()

    if not RENDERER.exists():
        raise RuntimeError(f"Missing renderer: {RENDERER}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    symbols = load_symbols()
    items = select_items(manifest, symbols, args.targets)
    if not items:
        raise RuntimeError("No matching manifest entries to render.")

    LARGE_DIR.mkdir(parents=True, exist_ok=True)
    SPRITE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dw-emoji-") as temp_name:
        temp = Path(temp_name)
        for index, item in enumerate(items, start=1):
            word = item["word"]
            symbol = symbols[word]
            slug = slugify(word)
            raw = temp / f"{slug}.png"
            large = LARGE_DIR / f"{slug}.png"
            sprite = SPRITE_DIR / f"{slug}.png"

            print(f"[{index}/{len(items)}] {word}")
            render_symbol(symbol, raw)
            rendered = Image.open(raw).convert("RGBA")
            fit_to_canvas(rendered, 512, 58).save(large)
            fit_to_canvas(rendered, 48, 4).save(sprite)

            item["image"] = str(large).replace("\\", "/")
            item["sprite"] = str(sprite).replace("\\", "/")
            item["hitbox"] = sprite_outline(sprite)
            item["hitboxSource"] = "directwrite-colr-paint-tree"

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {len(items)} DirectWrite emoji asset pairs.")


if __name__ == "__main__":
    main()
