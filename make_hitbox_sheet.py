from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json


def load_font(size):
    for path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    manifest = json.loads(Path("assets/manifest.json").read_text(encoding="utf-8"))
    cols = 10
    cell = 128
    label_h = 24
    rows = (len(manifest) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * cell), "#fffaf0")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)

    for i, item in enumerate(manifest):
        col = i % cols
        row = i // cols
        ox = col * cell
        oy = row * cell

        draw.rectangle((ox, oy, ox + cell - 1, oy + cell - 1), outline="#d9d0bd")
        sprite = Image.open(item["sprite"]).convert("RGBA")
        scale = 1.8
        size = int(48 * scale)
        x = ox + (cell - size) // 2
        y = oy + 12
        enlarged = sprite.resize((size, size), Image.Resampling.NEAREST)
        sheet.alpha_composite(enlarged, (x, y))

        cx = x + size / 2
        cy = y + size / 2
        points = [(cx + p["x"] * size, cy + p["y"] * size) for p in item["hitbox"]]
        draw.line(points + [points[0]], fill="#ff1744", width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#00b2ca", outline="#21313a")

        label = item["word"][:17]
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = ox + (cell - (bbox[2] - bbox[0])) / 2
        draw.text((tx, oy + cell - label_h + 5), label, fill="#21313a", font=font)

    out = Path("sprite-hitbox-debug-sheet.png")
    sheet.convert("RGB").save(out, quality=95)
    print(out.resolve())


if __name__ == "__main__":
    main()
