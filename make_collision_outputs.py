from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json


ROOT = Path("assets")
SPRITES = ROOT / "sprites"
POLYS = ROOT / "sprite-pointvis-10"
OUT_JSON = POLYS / "collision-polygons.json"
OUT_SHEET = POLYS / "collision-polygons-sheet.png"


def load_font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def source_sprites():
    skip = ("overlay", "pointvis", "converge", "autopoints", "sdfsafe", "50starts", "points-")
    return [
        p for p in sorted(SPRITES.glob("*.png"))
        if not any(token in p.stem for token in skip)
    ]


def main():
    items = []
    for sprite_path in source_sprites():
        json_path = POLYS / f"{sprite_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        verts = [{"x": float(x), "y": float(y)} for x, y in data["vertices"]]
        xs = [p["x"] for p in verts]
        ys = [p["y"] for p in verts]
        items.append({
            "name": sprite_path.stem,
            "sprite": str(sprite_path).replace("\\", "/"),
            "overlay": str((POLYS / f"{sprite_path.stem}.png")).replace("\\", "/"),
            "area": float(data["area"]),
            "vertices": verts,
            "bounds": {
                "x": min(xs),
                "y": min(ys),
                "w": max(xs) - min(xs),
                "h": max(ys) - min(ys),
            },
        })

    OUT_JSON.write_text(json.dumps(items, indent=2), encoding="utf-8")

    cols = 10
    cell = 156
    sprite_size = 96
    label_h = 28
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * cell), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)

    for i, item in enumerate(items):
        col = i % cols
        row = i // cols
        ox = col * cell
        oy = row * cell
        draw.rectangle((ox, oy, ox + cell - 1, oy + cell - 1), outline="#d0d0d0")

        sprite = Image.open(item["sprite"]).convert("RGBA")
        enlarged = sprite.resize((sprite_size, sprite_size), Image.Resampling.NEAREST)
        x0 = ox + (cell - sprite_size) // 2
        y0 = oy + 12
        sheet.alpha_composite(enlarged, (x0, y0))
        scale = sprite_size / sprite.width

        pts = [(x0 + p["x"] * scale, y0 + p["y"] * scale) for p in item["vertices"]]
        bx = item["bounds"]["x"] * scale + x0
        by = item["bounds"]["y"] * scale + y0
        bw = item["bounds"]["w"] * scale
        bh = item["bounds"]["h"] * scale
        draw.rectangle((bx, by, bx + bw, by + bh), outline="#0077ff", width=2)
        draw.line(pts + [pts[0]], fill="#ff1744", width=2)
        for px, py in pts:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#00b2ca")

        label = f"{item['name']} ({len(item['vertices'])})"
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = ox + (cell - (bbox[2] - bbox[0])) / 2
        draw.text((tx, oy + cell - label_h + 7), label, fill="#202020", font=font)

    sheet.convert("RGB").save(OUT_SHEET, quality=95)
    print(OUT_JSON.resolve())
    print(OUT_SHEET.resolve())


if __name__ == "__main__":
    main()
