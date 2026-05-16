from pathlib import Path
import json


MANIFEST = Path("assets/manifest.json")
POLYGONS = Path("assets/sprite-pointvis-10/collision-polygons.json")
SPRITE_SIZE = 48.0


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    polygons = json.loads(POLYGONS.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in polygons}

    missing = []
    for item in manifest:
        sprite_name = Path(item["sprite"]).stem
        poly = by_name.get(sprite_name)
        if not poly:
            missing.append(sprite_name)
            continue
        item["hitbox"] = [
            {
                "x": round((point["x"] - SPRITE_SIZE / 2.0) / SPRITE_SIZE, 6),
                "y": round((point["y"] - SPRITE_SIZE / 2.0) / SPRITE_SIZE, 6),
            }
            for point in poly["vertices"]
        ]
        item["hitboxSource"] = "assets/sprite-pointvis-10/collision-polygons.json"

    if missing:
        raise RuntimeError(f"Missing polygons for: {', '.join(missing)}")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {len(manifest)} manifest hitboxes from {POLYGONS}")


if __name__ == "__main__":
    main()
