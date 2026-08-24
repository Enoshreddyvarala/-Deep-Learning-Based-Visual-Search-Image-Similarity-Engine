"""
Generates a small synthetic 'shoe catalog' of placeholder images so the app
is fully runnable out-of-the-box, without requiring the ~50k-image UT-Zappos50K
dataset to be downloaded. Mirrors the Category/SubCategory/Brand/*.jpg folder
layout used by the training notebook (visual_search_shoe_similarity.ipynb).

Run once: python generate_demo_catalog.py
Swap in the real UT-Zappos50K folder + trained model artifacts (see README)
to move from demo mode to the real trained embedding model.
"""
import os
import csv
import random
from PIL import Image, ImageDraw

random.seed(42)

CATALOG_DIR = os.path.join(os.path.dirname(__file__), "static", "catalog_demo")

STRUCTURE = {
    "Boots": {
        "Ankle": ["Timberland", "DrMartens", "Frye"],
        "Knee-High": ["Coach", "Sam Edelman"],
    },
    "Sandals": {
        "Flat": ["Birkenstock", "Teva"],
        "Heel": ["SteveMadden", "Nine West"],
    },
    "Shoes": {
        "Sneakers": ["Nike", "Adidas", "NewBalance", "Converse"],
        "Oxford": ["ColeHaan", "Clarks"],
    },
    "Slippers": {
        "Slip-On": ["Ugg", "LLBean"],
    },
}

# Distinct base colors per subcategory so the fallback (color-histogram)
# similarity engine has real, meaningfully-different visual signal to work with.
SUBCAT_COLORS = {
    "Ankle": (92, 64, 51),
    "Knee-High": (61, 43, 31),
    "Flat": (222, 184, 135),
    "Heel": (178, 34, 52),
    "Sneakers": (245, 245, 245),
    "Oxford": (40, 40, 40),
    "Slip-On": (200, 170, 130),
}


def _shoe_silhouette(draw, w, h, base_color, variant):
    """Draw a very simple stylized shoe silhouette, varied slightly per image."""
    jitter = lambda v: max(0, min(255, v + random.randint(-18, 18)))
    color = tuple(jitter(c) for c in base_color)
    accent = tuple(jitter(255 - c) for c in base_color)

    draw.rectangle([0, 0, w, h], fill=(250, 250, 250))
    sole_y = int(h * 0.78)
    draw.rounded_rectangle([w * 0.08, sole_y, w * 0.92, sole_y + h * 0.10],
                            radius=8, fill=(30, 30, 30))
    body_pts = [
        (w * 0.10, sole_y),
        (w * 0.14, h * 0.45),
        (w * 0.30, h * 0.30 + variant),
        (w * 0.55, h * 0.28),
        (w * 0.85, h * 0.40),
        (w * 0.90, sole_y),
    ]
    draw.polygon(body_pts, fill=color)
    draw.line([(w * 0.30, h * 0.45), (w * 0.60, h * 0.35)], fill=accent, width=4)
    draw.ellipse([w * 0.68, h * 0.30, w * 0.80, h * 0.42], fill=accent)
    return


def generate(images_per_brand=3, size=224):
    rows = []
    if os.path.isdir(CATALOG_DIR):
        for cat, subcats in STRUCTURE.items():
            for subcat, brands in subcats.items():
                for brand in brands:
                    d = os.path.join(CATALOG_DIR, cat, subcat, brand)
                    os.makedirs(d, exist_ok=True)
                    base_color = SUBCAT_COLORS.get(subcat, (128, 128, 128))
                    for i in range(images_per_brand):
                        img = Image.new("RGB", (size, size), (255, 255, 255))
                        draw = ImageDraw.Draw(img)
                        _shoe_silhouette(draw, size, size, base_color, variant=i * 6)
                        fname = f"{brand.lower()}_{subcat.lower()}_{i}.png".replace(" ", "")
                        fpath = os.path.join(d, fname)
                        img.save(fpath)
                        rel = os.path.relpath(fpath, os.path.dirname(__file__))
                        rows.append({
                            "path": rel.replace("\\", "/"),
                            "category": cat,
                            "subcategory": subcat,
                            "brand": brand,
                            "class_label": f"{cat}/{subcat}/{brand}",
                        })
    csv_path = os.path.join(CATALOG_DIR, "demo_catalog_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "subcategory", "brand", "class_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} demo catalog images -> {CATALOG_DIR}")
    print(f"Index written -> {csv_path}")


if __name__ == "__main__":
    generate()
