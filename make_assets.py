from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import re


OBJECTS = [
    ("ball", "⚽"), ("teddy bear", "🧸"), ("toy car", "🚗"), ("train", "🚂"),
    ("airplane", "✈️"), ("rocket", "🚀"), ("boat", "⛵"), ("bicycle", "🚲"),
    ("doll", "🪆"), ("blocks", "🧱"), ("puzzle", "🧩"), ("kite", "🪁"),
    ("drum", "🥁"), ("guitar", "🎸"), ("piano", "🎹"), ("microphone", "🎤"),
    ("book", "📘"), ("crayon", "🖍️"), ("paintbrush", "🖌️"), ("scissors", "✂️"),
    ("backpack", "🎒"), ("lunchbox", "🍱"), ("apple", "🍎"), ("banana", "🍌"),
    ("strawberry", "🍓"), ("grapes", "🍇"), ("watermelon", "🍉"), ("cookie", "🍪"),
    ("cupcake", "🧁"), ("ice cream", "🍦"), ("pizza", "🍕"), ("sandwich", "🥪"),
    ("milk", "🥛"), ("spoon", "🥄"), ("toothbrush", "🪥"), ("soap", "🧼"),
    ("shoe", "👟"), ("hat", "🧢"), ("shirt", "👕"), ("socks", "🧦"),
    ("umbrella", "☂️"), ("sun", "☀️"), ("moon", "🌙"), ("star", "⭐"),
    ("cloud", "☁️"), ("rainbow", "🌈"), ("snowflake", "❄️"), ("flower", "🌸"),
    ("tree", "🌳"), ("leaf", "🍃"), ("dog", "🐶"), ("cat", "🐱"),
    ("rabbit", "🐰"), ("mouse", "🐭"), ("monkey", "🐵"), ("lion", "🦁"),
    ("tiger", "🐯"), ("bear", "🐻"), ("panda", "🐼"), ("frog", "🐸"),
    ("duck", "🦆"), ("chicken", "🐔"), ("penguin", "🐧"), ("fish", "🐟"),
    ("turtle", "🐢"), ("butterfly", "🦋"), ("bee", "🐝"), ("ladybug", "🐞"),
    ("house", "🏠"), ("bed", "🛏️"), ("chair", "🪑"), ("clock", "🕒"),
    ("lamp", "💡"), ("key", "🔑"), ("phone", "📱"), ("camera", "📷"),
    ("television", "📺"), ("computer", "💻"), ("robot", "🤖"), ("crown", "👑"),
    ("balloon", "🎈"), ("gift", "🎁"), ("party hat", "🥳"), ("magic wand", "🪄"),
    ("basketball", "🏀"), ("baseball", "⚾"), ("football", "🏈"), ("tennis ball", "🎾"),
    ("medal", "🏅"), ("flag", "🚩"), ("fire truck", "🚒"), ("police car", "🚓"),
    ("bus", "🚌"), ("tractor", "🚜"), ("ambulance", "🚑"), ("traffic light", "🚦"),
    ("stop sign", "🛑"), ("castle", "🏰"), ("beach ball", "🏐"), ("rubber duck", "🦆"),
]


COLORS = [
    "#f7b267", "#f79d65", "#f4845f", "#7dcfb6", "#00b2ca",
    "#1d4e89", "#9b5de5", "#f15bb5", "#fee440", "#00bbf9",
]


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def font(size):
    candidates = [
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/SegoeUIEmoji.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered_text(draw, box, text, face, fill):
    bbox = draw.textbbox((0, 0), text, font=face, embedded_color=True)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2 - bbox[0]
    y = box[1] + (box[3] - box[1] - height) / 2 - bbox[1]
    try:
        draw.text((x, y), text, font=face, embedded_color=True)
    except TypeError:
        draw.text((x, y), text, font=face, fill=fill)


def paste_symbol_fit(base, box, symbol, size):
    temp_size = size * 4
    temp = Image.new("RGBA", (temp_size, temp_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    face = font(size)
    bbox = draw.textbbox((0, 0), symbol, font=face, embedded_color=True)
    x = (temp_size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (temp_size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    try:
        draw.text((x, y), symbol, font=face, embedded_color=True)
    except TypeError:
        draw.text((x, y), symbol, font=face, fill="#263238")

    alpha_bbox = temp.getbbox()
    if not alpha_bbox:
        return

    cropped = temp.crop(alpha_bbox)
    target_w = box[2] - box[0]
    target_h = box[3] - box[1]
    scale = min(target_w / cropped.width, target_h / cropped.height)
    new_size = (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale)))
    resample = Image.Resampling.LANCZOS if new_size[0] > 60 else Image.Resampling.NEAREST
    fitted = cropped.resize(new_size, resample)
    px = int(box[0] + (target_w - fitted.width) / 2)
    py = int(box[1] + (target_h - fitted.height) / 2)
    base.alpha_composite(fitted, (px, py))


def make_large(word, symbol, color, path):
    img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    paste_symbol_fit(img, (58, 58, 454, 454), symbol, 290)
    img.save(path)


def make_sprite(symbol, color, path):
    hi = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    paste_symbol_fit(hi, (4, 4, 92, 92), symbol, 66)
    sprite = hi.resize((48, 48), Image.Resampling.LANCZOS)
    sprite.save(path)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(points):
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1] -
        points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2


def triangle_area(a, b, c):
    return abs(cross(a, b, c)) / 2


def simplify_polygon(points, target=7):
    points = list(points)
    while len(points) > target:
        scores = []
        for i, point in enumerate(points):
            prev_point = points[i - 1]
            next_point = points[(i + 1) % len(points)]
            scores.append((triangle_area(prev_point, point, next_point), i))
        _, remove_index = min(scores)
        points.pop(remove_index)
    return points


def sprite_outline(path):
    im = Image.open(path).convert("RGBA")
    alpha = im.getchannel("A")
    points = []
    width, height = im.size
    pixels = alpha.load()
    for y in range(height):
        for x in range(width):
            if pixels[x, y] <= 16:
                continue
            edge = (
                x == 0 or y == 0 or x == width - 1 or y == height - 1 or
                pixels[x - 1, y] <= 16 or pixels[min(x + 1, width - 1), y] <= 16 or
                pixels[x, y - 1] <= 16 or pixels[x, min(y + 1, height - 1)] <= 16
            )
            if edge:
                points.append((x, y))

    if len(points) < 3:
        return [{"x": -0.5, "y": -0.5}, {"x": 0.5, "y": -0.5}, {"x": 0.5, "y": 0.5}, {"x": -0.5, "y": 0.5}]

    hull = convex_hull(points)
    outline = simplify_polygon(hull, 7 if len(hull) >= 7 else len(hull))
    if len(outline) < 6 and len(hull) >= 6:
        outline = simplify_polygon(hull, 6)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    normalized = [{"x": round((x - cx) / span, 4), "y": round((y - cy) / span, 4)} for x, y in outline]
    return normalized


def main():
    assets = Path("assets")
    large_dir = assets / "large"
    sprite_dir = assets / "sprites"
    audio_dir = assets / "audio"
    large_dir.mkdir(parents=True, exist_ok=True)
    sprite_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, (word, symbol) in enumerate(OBJECTS):
        slug = slugify(word)
        color = COLORS[i % len(COLORS)]
        large = large_dir / f"{slug}.png"
        sprite = sprite_dir / f"{slug}.png"
        make_large(word, symbol, color, large)
        make_sprite(symbol, color, sprite)
        manifest.append({
            "word": word,
            "image": str(large).replace("\\", "/"),
            "sprite": str(sprite).replace("\\", "/"),
            "audio": str((audio_dir / f"{slug}.opus")).replace("\\", "/"),
            "hitbox": sprite_outline(sprite),
        })

    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} large images and {len(manifest)} sprites.")


if __name__ == "__main__":
    main()
