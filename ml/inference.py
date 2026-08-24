"""
Inference layer for the shoe visual-search engine.

Two modes, chosen automatically at startup:

1. TRAINED MODEL MODE (production)
   Used when all deployment artifacts produced by
   `visual_search_shoe_similarity.ipynb` are present in MODEL_ARTIFACTS_DIR:
     - shoe_embedding_model.pt   (ResNet50 + projection head, triplet-loss trained)
     - catalog_embeddings.npy    (L2-normalized 256-d embeddings, one per catalog image)
     - catalog_index.csv         (path, category, subcategory, brand per row, aligned to embeddings)
     - catalog.faiss             (FAISS IndexFlatIP over the embeddings -> cosine similarity)
   Also requires `torch`, `torchvision`, and `faiss` to be installed.

2. DEMO MODE (fallback, zero heavy dependencies)
   Used automatically when the above isn't available, so the app is runnable
   out of the box. Uses a simple RGB color-histogram + downsampled pixel-grid
   descriptor (numpy/Pillow only) over a small synthetic demo catalog
   (see generate_demo_catalog.py). This is NOT the trained deep model -
   it is a stand-in with the exact same interface (`find_similar`) so the
   Flask app, DB schema, and UI work identically once the real model is dropped in.

To go from demo -> trained model:
  1. Run visual_search_shoe_similarity.ipynb end-to-end on the UT-Zappos50K dataset.
  2. Copy the four output artifacts into MODEL_ARTIFACTS_DIR.
  3. `pip install torch torchvision faiss-cpu` and restart the app.
"""
import os
import csv
import json
import numpy as np
from PIL import Image

_TORCH_MODE_AVAILABLE = True
_TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    import faiss
except ImportError as exc:
    _TORCH_MODE_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


EMBED_DIM = 256
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


if _TORCH_MODE_AVAILABLE:

    class ShoeEmbeddingNet(nn.Module):
        """Same architecture as trained in visual_search_shoe_similarity.ipynb:
        ResNet50 backbone (ImageNet-pretrained) + projection head -> L2-normalized
        256-d embedding, trained with triplet loss."""

        def __init__(self, embed_dim=EMBED_DIM):
            super().__init__()
            backbone = models.resnet50(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.projection = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, embed_dim),
            )

        def forward(self, x):
            feats = self.backbone(x)
            emb = self.projection(feats)
            emb = F.normalize(emb, p=2, dim=1)
            return emb


class ModelService:
    """Singleton-style service the Flask app calls into for visual search."""

    def __init__(self, app_root, artifacts_dir, demo_catalog_dir):
        self.app_root = app_root
        self.artifacts_dir = artifacts_dir
        self.demo_catalog_dir = demo_catalog_dir
        self.mode = "demo"
        self.model = None
        self.index = None
        self.catalog = None  # list of dicts
        self._demo_features = None  # np.ndarray, only in demo mode

        if self._trained_artifacts_present():
            try:
                self._load_trained_model()
                self.mode = "trained_model"
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[ModelService] Failed to load trained model, falling back to demo mode: {exc}")
                self._load_demo_catalog()
        else:
            self._log_demo_mode_reason()
            self._load_demo_catalog()

    # ---------- setup ----------

    def _trained_artifacts_present(self):
        if not _TORCH_MODE_AVAILABLE:
            return False
        required = ["shoe_embedding_model.pt", "catalog_embeddings.npy",
                    "catalog_index.csv", "catalog.faiss"]
        return all(os.path.exists(os.path.join(self.artifacts_dir, f)) for f in required)

    def _log_demo_mode_reason(self):
        if not _TORCH_MODE_AVAILABLE:
            print(
                "[ModelService] Running in demo mode: torch/torchvision/faiss not installed "
                f"({_TORCH_IMPORT_ERROR}). Install them from requirements.txt and restart."
            )
            return
        required = ["shoe_embedding_model.pt", "catalog_embeddings.npy",
                    "catalog_index.csv", "catalog.faiss"]
        missing = [f for f in required if not os.path.exists(os.path.join(self.artifacts_dir, f))]
        if missing:
            print(
                "[ModelService] Running in demo mode: missing artifact(s) in "
                f"{self.artifacts_dir}: {', '.join(missing)}"
            )
        else:
            print("[ModelService] Running in demo mode: trained artifacts check failed unexpectedly.")

    def _load_trained_model(self):
        self.model = ShoeEmbeddingNet(EMBED_DIM)
        state = torch.load(
            os.path.join(self.artifacts_dir, "shoe_embedding_model.pt"),
            map_location="cpu",
        )
        self.model.load_state_dict(state)
        self.model.eval()

        self.index = faiss.read_index(os.path.join(self.artifacts_dir, "catalog.faiss"))

        catalog_rows = []
        with open(os.path.join(self.artifacts_dir, "catalog_index.csv"), newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["path"] = self._to_static_catalog_path(row["path"])
                catalog_rows.append(row)
        self.catalog = catalog_rows

        self.eval_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @staticmethod
    def _to_static_catalog_path(raw_path):
        """Map Kaggle/absolute catalog paths to Flask static-relative URLs.

        Training artifacts store paths like
        /kaggle/input/.../ut-zap50k-images-square/Sandals/Athletic/Nike/file.jpg
        Local images live at static/catalog/<Category>/... so templates can use
        url_for('static', filename=path).
        """
        p = (raw_path or "").replace("\\", "/")
        marker = "ut-zap50k-images-square/"
        idx = p.rfind(marker)
        if idx != -1:
            rel = p[idx + len(marker):]
            if rel.startswith(marker):
                rel = rel[len(marker):]
            return "catalog/" + rel.lstrip("/")
        if p.startswith("catalog/") or p.startswith("static/catalog/"):
            return p[len("static/"):] if p.startswith("static/") else p
        parts = [part for part in p.split("/") if part]
        cats = {"Boots", "Sandals", "Shoes", "Slippers"}
        for i, part in enumerate(parts):
            if part in cats:
                return "catalog/" + "/".join(parts[i:])
        return p

    def _load_demo_catalog(self):
        index_csv = os.path.join(self.demo_catalog_dir, "demo_catalog_index.csv")
        if not os.path.exists(index_csv):
            # generate on first run
            import generate_demo_catalog
            generate_demo_catalog.generate()

        rows = []
        with open(index_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self.catalog = rows
        self._demo_features = np.stack(
            [self._color_grid_descriptor(os.path.join(self.app_root, row["path"])) for row in rows]
        )

    # ---------- descriptors ----------

    @staticmethod
    def _color_grid_descriptor(path, grid=8):
        """Cheap, dependency-light visual descriptor used only in demo mode:
        average RGB color per cell of an 8x8 grid, flattened + L2-normalized.
        Real similarity search in production uses the trained CNN embedding."""
        img = Image.open(path).convert("RGB").resize((grid * 16, grid * 16))
        arr = np.asarray(img, dtype=np.float32)
        h, w, _ = arr.shape
        cell_h, cell_w = h // grid, w // grid
        feats = []
        for i in range(grid):
            for j in range(grid):
                cell = arr[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                feats.append(cell.mean(axis=(0, 1)))
        vec = np.concatenate(feats).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ---------- public API ----------

    def find_similar(self, query_image_path, top_k=8):
        """Returns (results, mode) where results is a list of dicts:
        {path, category, subcategory, brand, similarity} sorted best-first."""
        if self.mode == "trained_model":
            return self._find_similar_trained(query_image_path, top_k), self.mode
        return self._find_similar_demo(query_image_path, top_k), self.mode

    def _find_similar_trained(self, query_image_path, top_k):
        img = Image.open(query_image_path).convert("RGB")
        tensor = self.eval_transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = self.model(tensor).numpy().astype(np.float32)
        scores, idxs = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            row = self.catalog[idx]
            results.append({
                "path": row["path"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "brand": row["brand"],
                "similarity": float(score),
            })
        return results

    def _find_similar_demo(self, query_image_path, top_k):
        query_vec = self._color_grid_descriptor(query_image_path)
        sims = self._demo_features @ query_vec  # cosine sim (vectors are L2-normalized)
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            row = self.catalog[idx]
            results.append({
                "path": row["path"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "brand": row["brand"],
                "similarity": float(sims[idx]),
            })
        return results

    def info(self):
        return {
            "mode": self.mode,
            "catalog_size": len(self.catalog) if self.catalog else 0,
            "embedding_dim": EMBED_DIM if self.mode == "trained_model" else 128,
            "backbone": "ResNet50 (triplet loss)" if self.mode == "trained_model" else "Color-grid descriptor (demo fallback)",
        }
