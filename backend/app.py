# backend/app.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)

    # ---- CORS --------------------------------------------------------------
    default_origins = [
        "http://localhost:5173",
        "https://localhost:5173",
        "https://proud-plant-085ec8d0f.2.azurestaticapps.net",  # your SWA
    ]
    extra = os.getenv("ALLOWED_ORIGINS", "")
    allowed_origins = default_origins + [o.strip() for o in extra.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

    # ---- Paths & mutable state ---------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_MODEL = BASE_DIR / "models" / "model.joblib"
    DEFAULT_FEATURES = BASE_DIR / "data" / "features.csv"
    DEFAULT_STORES = BASE_DIR / "data" / "stores.csv"

    MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
    FEATURES_PATH = Path(os.getenv("FEATURES_CSV", DEFAULT_FEATURES))
    STORES_PATH = Path(os.getenv("STORES_CSV", DEFAULT_STORES))
    # If your store column in features.csv has a specific name, set STORE_COLUMN.
    # e.g. STORE_COLUMN=Number  (case-insensitive match)
    STORE_COLUMN = os.getenv("STORE_COLUMN", "")

    state: Dict[str, object] = {
        "model": None,
        "feature_names": [],
        "stores": [],
        "last_error": None,
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "stores_path": str(STORES_PATH),
        "store_source": None,
    }

    # ---- Helpers ------------------------------------------------------------

    def _resolve_model_path(p: Path) -> Path:
        if p.exists():
            return p
        alt = p.with_suffix(".pkl")
        return alt if alt.exists() else p

    def _load_features_csv(path: Path) -> List[str]:
        """Accept either a single-column list OR a table (use headers; drop common targets)."""
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            names = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            banned = {"target", "label", "y"}
            names = [c for c in df.columns if c.lower() not in banned]
        if not names:
            raise ValueError("No feature names found in features.csv")
        return names

    def _infer_stores_from_onehot(names: List[str]) -> List[dict]:
        """Infer store ids from one-hot feature names like store_001 or store[T.002]."""
        candidates: set[str] = set()
        for n in names:
            low = n.lower()

            for pref in ("store_", "storeid_", "store_id_", "location_", "loc_", "shop_"):
                if low.startswith(pref):
                    suffix = n[len(pref) :].strip()
                    suffix = suffix.replace("[T.", "").replace("]", "").replace("__", "_")
                    if suffix:
                        candidates.add(suffix)

            m = re.match(r"^(store|store_id|location|shop)[\s_\-]+(.+)$", low)
            if m:
                suffix = n[m.end(1) :].lstrip(" _-").replace("[T.", "").replace("]", "").strip()
                if suffix:
                    candidates.add(suffix)

        if not candidates:
            return []

        def sort_key(x: str):
            return (0, int(x)) if x.isdigit() else (1, x)

        return [{"id": s, "name": f"Store {s}"} for s in sorted(candidates, key=sort_key)]

    def _infer_stores_from_features_table(path: Path) -> List[dict]:
        """
        If features.csv is a TABLE, pull the distinct values from the store column.
        Candidate column names are checked case-insensitively.
        You can force a specific column via STORE_COLUMN env var.
        """
        df = pd.read_csv(path)
        if df.empty:
            return []

        cols_lower = [c.lower().strip() for c in df.columns]

        # explicit override
        if STORE_COLUMN:
            try_names = [STORE_COLUMN.lower().strip()]
        else:
            try_names = [
                "store", "store_id", "storeid", "store number", "store_number",
                "number", "location", "shop",
            ]

        col_name = None
        for cand in try_names:
            if cand in cols_lower:
                col_name = df.columns[cols_lower.index(cand)]
                break
        if not col_name:
            return []

        values = pd.unique(df[col_name].dropna())
        ids = [str(v) for v in values]

        def sort_key(x: str):
            return (0, int(x)) if x.isdigit() else (1, x)

        return [{"id": s, "name": f"Store {s}"} for s in sorted(ids, key=sort_key)]

    def _load_stores_csv(path: Path) -> List[dict]:
        """Fallback: read stores from a separate CSV (id[,name] or store[,name])."""
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "id" in cols:
            id_col = df.columns[cols.index("id")]
        elif "store" in cols:
            id_col = df.columns[cols.index("store")]
        else:
            id_col = df.columns[0]
        name_col = df.columns[cols.index("name")] if "name" in cols else None

        out = []
        for _, r in df.iterrows():
            sid = str(r[id_col])
            name = str(r[name_col]) if name_col else f"Store {sid}"
            out.append({"id": sid, "name": name})
        return out

    def load_artifacts() -> None:
        """Load model + features (+ stores) into memory."""
        try:
            mp = _resolve_model_path(MODEL_PATH)
            model = joblib.load(mp)
            feats = _load_features_csv(FEATURES_PATH)

            stores = []
            source = None

            # 1) PREFER: pull stores from features.csv table (e.g., 'Number' column)
            if FEATURES_PATH.exists():
                tbl_stores = _infer_stores_from_features_table(FEATURES_PATH)
                if tbl_stores:
                    stores = tbl_stores
                    source = "features_csv_column"

            # 2) FALLBACK: infer from one-hot feature names if table didn’t work
            if not stores:
                oh_stores = _infer_stores_from_onehot(feats)
                if oh_stores:
                    stores = oh_stores
                    source = "onehot_features"

            # 3) LAST RESORT: separate stores.csv if present
            if not stores and STORES_PATH.exists():
                csv_stores = _load_stores_csv(STORES_PATH)
                if csv_stores:
                    stores = csv_stores
                    source = "stores_csv"

            state["model"] = model
            state["feature_names"] = feats
            state["stores"] = stores
            state["store_source"] = source
            state["model_path"] = str(mp)
            state["last_error"] = None

        except Exception as e:
            state["model"] = None
            state["feature_names"] = []
            state["stores"] = []
            state["store_source"] = None
            state["last_error"] = str(e)

    def _prepare_dataframe(payload: dict) -> Tuple[pd.DataFrame, bool]:
        """
        Accepts:
          - {"data": {feat: value, ...}}
          - {"rows": [ {feat: val}, {...} ]}
          - Or a flat dict {feat: value}
        """
        data = payload.get("data", payload.get("row", payload))
        rows = payload.get("rows")
        feats: List[str] = state["feature_names"]

        if rows and isinstance(rows, list):
            df = pd.DataFrame(rows)
            missing = [f for f in feats if f not in df.columns]
            if missing:
                raise KeyError(f"Missing features: {missing}")
            df = df[feats]
            return df, True
        else:
            if not isinstance(data, dict):
                raise TypeError("Payload should be a JSON object or {data: {...}}")
            missing = [f for f in feats if f not in data]
            if missing:
                raise KeyError(f"Missing features: {missing}")
            row = [data[f] for f in feats]
            df = pd.DataFrame([row], columns=feats)
            return df, False

    # ---- Load artifacts at startup -----------------------------------------
    load_artifacts()

    # ---- Routes -------------------------------------------------------------

    @app.get("/api/health")
    def health():
        ok = state["model"] is not None and len(state["feature_names"]) > 0
        return jsonify(
            status="ok" if ok else "init",
            service="backend",
            n_features=len(state["feature_names"]),
            n_stores=len(state["stores"]),
            model_path=state["model_path"],
            features_path=state["features_path"],
            stores_path=state["stores_path"],
            store_source=state["store_source"],
            last_error=state["last_error"],
        )

    @app.get("/api/features")
    def features():
        if not state["feature_names"]:
            return jsonify(error="features_not_loaded", message=state["last_error"]), 500
        return jsonify(features=state["feature_names"])

    @app.get("/api/stores")
    def stores():
        if state["stores"]:
            return jsonify(stores=state["stores"], source=state["store_source"] or "inferred_or_csv")
        # No stores found: give a helpful note
        return jsonify(
            stores=[],
            source="unknown",
            note=(
                "No store list found. Set STORE_COLUMN env var (e.g. STORE_COLUMN=Number), "
                "or provide backend/data/stores.csv"
            ),
        )

    @app.get("/api/hello")
    def hello():
        store = request.args.get("store")
        msg = "Hello from backend!"
        if store:
            msg += f" (store={store})"
        return jsonify(message=msg)

    @app.post("/api/forecast")
    def forecast():
        if state["model"] is None or not state["feature_names"]:
            return jsonify(error="model_not_loaded", message=state["last_error"]), 503

        payload = request.get_json(force=True, silent=True) or {}
        try:
            X, _ = _prepare_dataframe(payload)
            X = X.apply(pd.to_numeric, errors="ignore")
            yhat = state["model"].predict(X)
            preds = getattr(yhat, "tolist", lambda: yhat)()
            return jsonify(predictions=preds, rows=len(preds))
        except KeyError as e:
            return jsonify(error="missing_features", message=str(e)), 400
        except Exception as e:
            return jsonify(error="prediction_failed", message=str(e)), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
