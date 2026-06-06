from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, map_coordinates


ROOT = Path("assets")
SPRITES = ROOT / "sprites"
POLYS = ROOT / "sprite-pointvis-10"
OUT_JSON = POLYS / "collision-polygons.json"
OUT_SHEET = POLYS / "collision-polygons-sheet.png"

MIN_SIDES = 4
MAX_SIDES = 32
ALPHA_THRESHOLD = 128
EPS_MIN = 0.5
EPS_MAX = 8.0
EPS_STEPS = 120
EDGE_TOLERANCE = 0.01
SEARCH_EDGE_SAMPLES_PER_PIXEL = 8
FINAL_EDGE_SAMPLES_PER_PIXEL = 20
MAX_POLYGON_EXTENT_PADDING = 4.0
SUPPORT_MARGINS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
RDP_EPSILON_FRACTIONS = np.linspace(0.001, 0.08, 160)
RDP_EXPAND_SCALES = [1.0, 1.005, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3]
DILATION_KERNELS = (3, 5, 7, 9, 11, 13, 15)
DILATION_ITERATIONS = (1, 2, 3)


def load_font(size: int):
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


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class CollisionSolver:
    def __init__(self, sprite_path: Path):
        self.sprite_path = sprite_path
        self.sprite = Image.open(sprite_path).convert("RGBA")
        rgba = np.array(self.sprite)
        self.height, self.width = rgba.shape[:2]
        self.mask = rgba[..., 3] >= ALPHA_THRESHOLD
        self.sprite_area_px2 = int(self.mask.sum())
        self.sdf = distance_transform_edt(self.mask) - distance_transform_edt(~self.mask)
        self.contours, _ = cv2.findContours(
            (self.mask.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not self.contours:
            raise RuntimeError(f"No external contour found for {sprite_path}")
        self.main_contour = max(self.contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)

    def raster_poly(self, poly: np.ndarray) -> np.ndarray:
        im = Image.new("1", (self.width, self.height), 0)
        ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in poly], fill=1)
        return np.array(im, dtype=bool)

    @staticmethod
    def simple_polygon(poly: np.ndarray) -> bool:
        def orient(a, b, c) -> float:
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def proper_intersection(a, b, c, d) -> bool:
            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)
            return (o1 * o2 < -1e-7) and (o3 * o4 < -1e-7)

        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or j == (i + 1) % n or (j + 1) % n == i:
                    continue
                if i == 0 and j == n - 1:
                    continue
                if proper_intersection(a, b, poly[j], poly[(j + 1) % n]):
                    return False
        return True

    def edge_sdf_values(self, poly: np.ndarray, samples_per_pixel: int) -> list[float]:
        max_values = []
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            dist = float(np.linalg.norm(p1 - p0))
            sample_count = max(int(math.ceil(dist * samples_per_pixel)), 4)
            xs = np.linspace(p0[0], p1[0], sample_count)
            ys = np.linspace(p0[1], p1[1], sample_count)
            values = map_coordinates(self.sdf, [ys, xs], order=1, mode="nearest")
            max_values.append(float(values.max()))
        return max_values

    def validate(self, poly: np.ndarray | None, samples_per_pixel: int) -> dict | None:
        if poly is None or len(poly) < MIN_SIDES or len(poly) > MAX_SIDES:
            return None
        if not np.all(np.isfinite(poly)):
            return None
        if not self.simple_polygon(poly):
            return None
        if (
            float(poly[:, 0].min()) < -MAX_POLYGON_EXTENT_PADDING
            or float(poly[:, 1].min()) < -MAX_POLYGON_EXTENT_PADDING
            or float(poly[:, 0].max()) > self.width - 1 + MAX_POLYGON_EXTENT_PADDING
            or float(poly[:, 1].max()) > self.height - 1 + MAX_POLYGON_EXTENT_PADDING
        ):
            return None

        poly_mask = self.raster_poly(poly)
        missing = int((self.mask & ~poly_mask).sum())
        if missing != 0:
            return None

        edge_max = self.edge_sdf_values(poly, samples_per_pixel=samples_per_pixel)
        largest = max(edge_max)
        if largest > EDGE_TOLERANCE:
            return None

        return {
            "area": polygon_area(poly),
            "missing_sprite_pixels": missing,
            "edge_max_sdf": edge_max,
            "largest_edge_max_sdf": float(largest),
            "signed_area": signed_area(poly),
        }

    def arc_points(self, i: int, j: int) -> np.ndarray:
        contour = self.main_contour
        if j >= i:
            return contour[i : j + 1]
        return np.vstack([contour[i:], contour[: j + 1]])

    def support_polygon(self, indices: list[int], normal_sign: int, margin: float) -> np.ndarray | None:
        lines: list[tuple[np.ndarray, float]] = []
        for k in range(len(indices)):
            i = indices[k]
            j = indices[(k + 1) % len(indices)]
            p0 = self.main_contour[i]
            p1 = self.main_contour[j]
            direction = p1 - p0
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                return None
            left_normal = np.array([-direction[1], direction[0]], dtype=np.float64) / length
            normal = normal_sign * left_normal
            c = float((self.arc_points(i, j) @ normal).min() - margin)
            lines.append((normal, c))

        vertices = []
        for k in range(len(indices)):
            n1, c1 = lines[k - 1]
            n2, c2 = lines[k]
            a = np.vstack([n1, n2])
            if abs(float(np.linalg.det(a))) < 1e-7:
                return None
            vertices.append(np.linalg.solve(a, np.array([c1, c2], dtype=np.float64)))
        return np.array(vertices, dtype=np.float64)

    def add_rdp_candidates(self, records: list[dict], source_poly: np.ndarray, metadata: dict) -> None:
        if len(source_poly) < MIN_SIDES or len(source_poly) > MAX_SIDES:
            return

        centroid = source_poly.mean(axis=0)
        for scale in RDP_EXPAND_SCALES:
            poly = centroid + (source_poly - centroid) * scale
            result = self.validate(poly, samples_per_pixel=FINAL_EDGE_SAMPLES_PER_PIXEL)
            if result is None:
                continue
            records.append({
                **result,
                "poly": poly,
                "sides": len(poly),
                "scale": float(scale),
                **metadata,
            })
            break

    def support_candidates(self) -> list[dict]:
        records: list[dict] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for eps in np.linspace(EPS_MIN, EPS_MAX, EPS_STEPS):
            approx = cv2.approxPolyDP(self.main_contour.astype(np.float32), float(eps), True)
            approx_points = approx[:, 0, :].astype(np.float64)
            indices = [
                int(np.argmin(np.sum((self.main_contour - p) ** 2, axis=1)))
                for p in approx_points
            ]
            indices = sorted(set(indices))
            if len(indices) < MIN_SIDES or len(indices) > MAX_SIDES:
                continue
            key = (len(indices), tuple(indices))
            if key in seen:
                continue
            seen.add(key)

            for normal_sign in (-1, 1):
                for margin in SUPPORT_MARGINS:
                    poly = self.support_polygon(indices, normal_sign, margin)
                    result = self.validate(poly, samples_per_pixel=SEARCH_EDGE_SAMPLES_PER_PIXEL)
                    if result is None:
                        continue
                    final = self.validate(poly, samples_per_pixel=FINAL_EDGE_SAMPLES_PER_PIXEL)
                    if final is None:
                        continue
                    records.append({
                        **final,
                        "poly": poly,
                        "sides": len(poly),
                        "method": "support-line",
                        "eps": float(eps),
                        "support_margin": margin,
                        "normal_sign": normal_sign,
                        "scale": None,
                        "dilation_kernel": None,
                        "dilation_iterations": None,
                        "dilated_contour_count": None,
                        "dilated_contour_source": None,
                    })
        return records

    def dilated_rdp_candidates(self) -> list[dict]:
        records: list[dict] = []
        seen: set[tuple] = set()
        base = (self.mask.astype(np.uint8) * 255)
        for kernel_size in DILATION_KERNELS:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            for iterations in DILATION_ITERATIONS:
                dilated = cv2.dilate(base, kernel, iterations=iterations)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                contour_sets = [("largest", max(contours, key=cv2.contourArea))]
                if len(contours) > 1:
                    contour_sets.append(("convex-hull-of-all-dilated-contours", cv2.convexHull(np.vstack(contours))))

                for contour_source, contour in contour_sets:
                    arc_length = cv2.arcLength(contour, True)
                    if arc_length <= 0:
                        continue
                    for epsilon_fraction in RDP_EPSILON_FRACTIONS:
                        approx = cv2.approxPolyDP(contour, float(epsilon_fraction * arc_length), True)
                        poly = approx[:, 0, :].astype(np.float64)
                        if len(poly) < MIN_SIDES or len(poly) > MAX_SIDES:
                            continue
                        key = (
                            kernel_size,
                            iterations,
                            contour_source,
                            tuple((int(round(x * 1000)), int(round(y * 1000))) for x, y in poly),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        self.add_rdp_candidates(records, poly, {
                            "method": "dilated-rdp",
                            "eps": float(epsilon_fraction * arc_length),
                            "epsilon_fraction": float(epsilon_fraction),
                            "support_margin": 0.0,
                            "normal_sign": 0,
                            "dilation_kernel": kernel_size,
                            "dilation_iterations": iterations,
                            "dilated_contour_count": len(contours),
                            "dilated_contour_source": contour_source,
                        })
        return records

    def particle_wrap_candidates(self) -> list[dict]:
        records: list[dict] = []
        ys, xs = np.nonzero(self.mask)
        all_points = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(all_points)
        arc_length = cv2.arcLength(hull, True)
        if arc_length <= 0:
            return records
        seen: set[tuple[tuple[int, int], ...]] = set()
        for epsilon_fraction in RDP_EPSILON_FRACTIONS:
            approx = cv2.approxPolyDP(hull, float(epsilon_fraction * arc_length), True)
            poly = approx[:, 0, :].astype(np.float64)
            if len(poly) < MIN_SIDES or len(poly) > MAX_SIDES:
                continue
            key = tuple((int(round(x * 1000)), int(round(y * 1000))) for x, y in poly)
            if key in seen:
                continue
            seen.add(key)
            self.add_rdp_candidates(records, poly, {
                "method": "particle-wrap-rdp",
                "eps": float(epsilon_fraction * arc_length),
                "epsilon_fraction": float(epsilon_fraction),
                "support_margin": 0.0,
                "normal_sign": 0,
                "dilation_kernel": None,
                "dilation_iterations": None,
                "dilated_contour_count": None,
                "dilated_contour_source": None,
            })
        return records

    def solve(self) -> dict:
        records = []
        records.extend(self.support_candidates())
        records.extend(self.dilated_rdp_candidates())
        records.extend(self.particle_wrap_candidates())
        if not records:
            raise RuntimeError(f"No valid collision polygon found for {self.sprite_path}")
        records.sort(key=lambda r: (r["area"], r["sides"], r["method"]))
        return records[0]


def draw_overlay(sprite_path: Path, poly: np.ndarray, out_path: Path) -> None:
    sprite = Image.open(sprite_path).convert("RGBA")
    scale = 5
    pad = 18
    label_h = 26
    w, h = sprite.size
    img = Image.new("RGBA", (w * scale + pad * 2, h * scale + pad * 2 + label_h), (20, 20, 20, 255))
    img.alpha_composite(sprite.resize((w * scale, h * scale), Image.Resampling.NEAREST), (pad, pad))
    draw = ImageDraw.Draw(img)
    pts = [(pad + float(x) * scale, pad + float(y) * scale) for x, y in poly]
    fill = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(fill).polygon(pts, fill=(0, 190, 255, 45))
    img.alpha_composite(fill)
    draw = ImageDraw.Draw(img)
    draw.line(pts + [pts[0]], fill=(0, 220, 255, 255), width=3)
    for x, y in pts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 220, 255, 255), outline=(255, 255, 255, 255))
    draw.rectangle((0, h * scale + pad * 2, img.width, img.height), fill=(0, 0, 0, 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def make_sheet(items: list[dict]) -> None:
    cols = 10
    cell = 156
    sprite_size = 96
    label_h = 31
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * cell), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    small = load_font(10)

    for i, item in enumerate(items):
        col = i % cols
        row = i // cols
        ox = col * cell
        oy = row * cell
        draw.rectangle((ox, oy, ox + cell - 1, oy + cell - 1), outline="#d0d0d0")

        sprite = Image.open(item["sprite"]).convert("RGBA")
        enlarged = sprite.resize((sprite_size, sprite_size), Image.Resampling.NEAREST)
        x0 = ox + (cell - sprite_size) // 2
        y0 = oy + 10
        sheet.alpha_composite(enlarged, (x0, y0))
        scale = sprite_size / sprite.width
        pts = [(x0 + p["x"] * scale, y0 + p["y"] * scale) for p in item["vertices"]]
        draw.line(pts + [pts[0]], fill="#00bcd4", width=3)
        for px, py in pts:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#00e5ff", outline="#10323a")

        label = f"{item['name']} ({len(item['vertices'])})"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        tx = ox + (cell - (label_bbox[2] - label_bbox[0])) / 2
        draw.text((tx, oy + cell - label_h + 3), label, fill="#202020", font=font)
        method = item["method"].replace("-rdp", "")
        method_bbox = draw.textbbox((0, 0), method, font=small)
        mx = ox + (cell - (method_bbox[2] - method_bbox[0])) / 2
        draw.text((mx, oy + cell - 14), method, fill="#666666", font=small)

    sheet.convert("RGB").save(OUT_SHEET, quality=95)


def main() -> None:
    POLYS.mkdir(parents=True, exist_ok=True)
    items = []
    for sprite_path in source_sprites():
        print(sprite_path.stem, flush=True)
        solver = CollisionSolver(sprite_path)
        record = solver.solve()
        poly = record["poly"]
        verts = [{"x": round(float(x), 6), "y": round(float(y), 6)} for x, y in poly]
        xs = [p["x"] for p in verts]
        ys = [p["y"] for p in verts]

        per_shape = {
            "name": sprite_path.stem,
            "algorithm": "bounded support-line search, dilated approxPolyDP fallback, particle-wrap fallback",
            "method": record["method"],
            "area": float(record["area"]),
            "sprite_area_pixels2": solver.sprite_area_px2,
            "sides": len(verts),
            "alpha_threshold": ALPHA_THRESHOLD,
            "verification": {
                "missing_sprite_pixels": int(record["missing_sprite_pixels"]),
                "largest_edge_max_sdf": float(record["largest_edge_max_sdf"]),
            },
            "parameters": {
                "max_sides": MAX_SIDES,
                "support_margin": record["support_margin"],
                "normal_sign": record["normal_sign"],
                "rdp_epsilon": record["eps"],
                "epsilon_fraction": record.get("epsilon_fraction"),
                "scale": record.get("scale"),
                "dilation_kernel": record.get("dilation_kernel"),
                "dilation_iterations": record.get("dilation_iterations"),
                "dilated_contour_count": record.get("dilated_contour_count"),
                "dilated_contour_source": record.get("dilated_contour_source"),
            },
            "vertices": [[p["x"], p["y"]] for p in verts],
        }
        (POLYS / f"{sprite_path.stem}.json").write_text(json.dumps(per_shape, indent=2), encoding="utf-8")

        overlay_path = POLYS / f"{sprite_path.stem}.png"
        draw_overlay(sprite_path, poly, overlay_path)
        items.append({
            "name": sprite_path.stem,
            "sprite": str(sprite_path).replace("\\", "/"),
            "overlay": str(overlay_path).replace("\\", "/"),
            "area": float(record["area"]),
            "method": record["method"],
            "vertices": verts,
            "bounds": {
                "x": min(xs),
                "y": min(ys),
                "w": max(xs) - min(xs),
                "h": max(ys) - min(ys),
            },
        })

    OUT_JSON.write_text(json.dumps(items, indent=2), encoding="utf-8")
    make_sheet(items)
    print(OUT_JSON.resolve())
    print(OUT_SHEET.resolve())


if __name__ == "__main__":
    main()
