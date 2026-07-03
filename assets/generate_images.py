"""Generate simple, abstract illustrative PNG banners for each disease.

No external image downloads — everything is drawn programmatically with PIL.
"""
import math
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "images")
SIZE = (400, 300)


def _base(bg):
    img = Image.new("RGB", SIZE, bg)
    return img, ImageDraw.Draw(img)


def draw_heart():
    img, d = _base((255, 235, 238))
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    color = (211, 47, 47)
    r = 60
    d.ellipse([cx - r * 1.5, cy - r, cx, cy + r], fill=color)
    d.ellipse([cx, cy - r, cx + r * 1.5, cy + r], fill=color)
    d.polygon([(cx - r * 1.5, cy), (cx + r * 1.5, cy), (cx, cy + r * 2)], fill=color)
    # simple ECG line
    pts = []
    for x in range(0, SIZE[0], 4):
        y = cy - 40 * math.sin(x / 20) if 100 < x < 300 else cy
        pts.append((x, y))
    d.line(pts, fill=(255, 255, 255), width=3)
    img.save(os.path.join(OUT_DIR, "heart.png"))


def draw_diabetes():
    img, d = _base((227, 242, 253))
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    color = (21, 101, 192)
    r = 70
    d.ellipse([cx - r, cy - r + 20, cx + r, cy + r + 20], fill=color)
    d.polygon([(cx - 25, cy - r - 10), (cx + 25, cy - r - 10), (cx, cy - r - 70)], fill=color)
    d.ellipse([cx - 20, cy - 5, cx + 20, cy + 35], fill=(227, 242, 253))
    img.save(os.path.join(OUT_DIR, "diabetes.png"))


def draw_breast_cancer():
    img, d = _base((252, 228, 236))
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    color = (194, 24, 91)
    width = 22
    d.line([(cx - 40, cy - 70), (cx + 40, cy + 30)], fill=color, width=width)
    d.line([(cx + 40, cy - 70), (cx - 40, cy + 30)], fill=color, width=width)
    d.polygon([(cx - 15, cy + 20), (cx + 15, cy + 20), (cx, cy + 90)], fill=color)
    img.save(os.path.join(OUT_DIR, "breast_cancer.png"))


def draw_ckd():
    img, d = _base((232, 245, 233))
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    color = (46, 125, 50)
    d.ellipse([cx - 70, cy - 60, cx + 10, cy + 60], fill=color)
    d.ellipse([cx - 20, cy - 30, cx + 20, cy + 30], fill=(232, 245, 233))
    d.ellipse([cx + 10, cy - 60, cx + 90, cy + 60], fill=color)
    d.ellipse([cx + 60, cy - 30, cx + 100, cy + 30], fill=(232, 245, 233))
    img.save(os.path.join(OUT_DIR, "ckd.png"))


def draw_parkinsons():
    img, d = _base((243, 229, 245))
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    color = (106, 27, 154)
    d.ellipse([cx - 80, cy - 60, cx + 80, cy + 60], fill=color)
    for i in range(6):
        angle = i * math.pi / 3
        x1, y1 = cx + 20 * math.cos(angle), cy + 20 * math.sin(angle)
        x2, y2 = cx + 55 * math.cos(angle), cy + 55 * math.sin(angle)
        d.line([(x1, y1), (x2, y2)], fill=(243, 229, 245), width=4)
    img.save(os.path.join(OUT_DIR, "parkinsons.png"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    draw_heart()
    draw_diabetes()
    draw_breast_cancer()
    draw_ckd()
    draw_parkinsons()
    print(f"Saved 5 images to {OUT_DIR}")


if __name__ == "__main__":
    main()
