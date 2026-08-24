# Cobbler's Eye — AI Visual Shoe Search

A Flask web application demonstrating an AI-powered visual search tool for a fashion
marketplace business scenario: a shopper uploads a photo of a shoe, and the system retrieves
visually similar catalog items — even when product names/text metadata differ.

Built around the model + dataset from `visual_search_shoe_similarity.ipynb`
(ResNet50 + triplet loss embedding model, FAISS nearest-neighbor search, UT-Zappos50K dataset).

## Features

- **Home page** — explains the business scenario, how the retrieval pipeline works
  (embed → rank → retrieve), and the model/dataset specs, pulled live from whichever
  inference mode is active.
- **Sign up / log in** — email + password auth (bcrypt-hashed), session-based via Flask-Login.
- **Visual search page** — upload a shoe photo, choose Top-K (4/8/12), see ranked results
  with similarity scores.
- **History page** — every past search (query thumbnail + results) per logged-in user, with
  delete support.
- **SQLite** — `users` and `search_history` tables via Flask-SQLAlchemy, stored at
  `instance/app.db`.

## Two inference modes

The app runs in one of two modes automatically, decided at startup in `ml/inference.py`:

| | **Demo mode** (default) | **Trained model mode** |
|---|---|---|
| Trigger | No trained artifacts found | All 4 artifact files present + `torch`/`faiss` installed |
| Descriptor | Lightweight 8×8 color-grid histogram (Pillow/NumPy only) | ResNet50 + triplet-loss projection head, 256-d embedding |
| Catalog | Small synthetic placeholder catalog (`generate_demo_catalog.py`) | Real UT-Zappos50K catalog |
| Search index | Brute-force cosine similarity in NumPy | FAISS `IndexFlatIP` |

This means the whole app — auth, upload, ranking UI, history — is runnable and testable
immediately, with **zero heavy dependencies**, and swapping in the real trained model is a
drop-in change.

### Switching to the trained model

1. Run `visual_search_shoe_similarity.ipynb` end-to-end against the UT-Zappos50K dataset
   (attach it via Kaggle "Add Input", or point `IMAGE_DIR` at a local copy).
2. Copy the 4 artifacts it produces into `model_artifacts/`:
   - `shoe_embedding_model.pt`
   - `catalog_embeddings.npy`
   - `catalog_index.csv`
   - `catalog.faiss`
3. `pip install torch torchvision faiss-cpu` (uncomment in `requirements.txt`).
4. Restart the app — it will detect the artifacts and switch to trained-model mode
   automatically. The home page's "Current mode" line and the search page's mode badge
   both reflect this.

## Setup

```bash
cd shoe_visual_search
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
python generate_demo_catalog.py                     # only needed for demo mode; auto-runs on first request too
python app.py
```

Then open http://localhost:5000, sign up, and try the search page with any shoe photo
(or any photo at all — demo mode will still return the closest color/shape matches from the
placeholder catalog).

## Project structure

```
shoe_visual_search/
├── app.py                  # Flask app factory
├── config.py                # App configuration
├── extensions.py            # db / login_manager / bcrypt singletons
├── models.py                 # User, SearchHistory (SQLAlchemy)
├── auth.py                    # /signup /login /logout
├── main.py                     # / /search /history
├── generate_demo_catalog.py     # builds the placeholder catalog for demo mode
├── ml/
│   └── inference.py               # ModelService: trained-model + demo-mode inference
├── templates/                       # Jinja templates
├── static/
│   ├── css/style.css                  # app styling
│   ├── uploads/                        # user-uploaded query images
│   └── catalog_demo/                    # generated placeholder catalog images
├── model_artifacts/                       # drop trained artifacts here (see above)
├── instance/app.db                          # SQLite database (created on first run)
└── requirements.txt
```

## Notes

- Passwords are hashed with bcrypt; never stored in plaintext.
- Uploaded images are capped at 8 MB and validated by extension (png/jpg/jpeg/webp).
- Precision@K evaluation (subcategory agreement) is implemented in the training notebook,
  not re-run inside the web app — it's a training-time quality check on held-out data.
