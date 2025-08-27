# app.py
import os
import json
import threading
from io import BytesIO
from typing import Optional, List, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib

# ------------------------------------------------------------
# Optional Azure Blob dependency
# ------------------------------------------------------------
try:
    from azure.storage.blob import BlobClient  # type: ignore
    HAS_AZURE = True
except Exception:
    HAS_AZURE = False


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def _norm(s: str) -> str:
    """normalize a column name (lowercase, spaces->underscores)."""
    return str(s).strip().lower().replace(" ", "_")

def _download_blob_bytes(
    *, connection_string: str, container: str, blob_name: str
) -> bytes:
    if not HAS_AZURE:
        raise RuntimeError("azure-storage-blob is not installed")
    client = BlobClient.from_connection_string(
        conn_str=connection_string, container_name=container, blob_name=blob_name
    )
    return client.download_blob().readall()

def _resolve_store_column(df: pd.DataFrame) -> Optional[str]:
    """
    Decide which column to use for store IDs.
    Priority:
      1) explicit env var STORE_ID_COLUMN (case/space-insensitive)
      2) common names: store, store_id, store_number, store_num
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}

    override = os.getenv("STORE_ID_COLUMN", "").strip()
    if override:
        key = _norm(override)
        if key in norm_map:
            return norm_map[key]

    for candidate in ("store", "store_id", "store_number", "store_num"):
        if candidate in norm_map:
            return norm_map[candidate]

    # If nothing obvious, but we have exactly one 'store' substring column, use it
    candidates = [c for c in cols if "store" in _norm(c)]
    if len(candidates) == 1:
        return candidates[0]
    return None

def _build_store_cache(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build [{id,name}] from the chosen store column; otherwise fall back to any
    columns that contain 'store'.
    """
    if df is None or df.empty:
        return []

    store_col = _resolve_store_column(df)
    if store_col:
        vals = (
            pd.Series(df[store_col])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        vals = sorted(vals)
        return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]

    # Fallback: show any columns that *look* store-related
    candidates = [c for c in df.columns if "store" in _norm(c)]
    return [{"id": c, "name": c.replace("_", " ").title()} for c in candidates[:5000]]


# ------------------------------------------------------------
# App factory
# ------------------------------------------------------------
def create_app() -> Flask:
    """
    Behavior:
      * If local files exist -> load them.
      * Else, if Blob vars are present -> download & load.
      * Else -> start OK with no model/data; health shows ok and /api/forecast returns 503.
    """
    app = Flask(__name__)
    CORS(app)

    # Runtime state
    app.config.update(
        MODEL=None,                  # Any fitted object (joblib)
        FEATURES_DF=pd.DataFrame(),  # DataFrame or empty
        STORE_LIST_CACHE=[],         # [{"id": "...", "name": "..."}]
        STATUS="starting",           # starting | warming | ok | error
        LAST_ERROR=None,             # str | None
        WARM=dict(started=False, done=False, error=None),
    )

    # ---------- Local paths (relative to app root by default)
    local_model_path = os.getenv("MODEL_PATH", os.path.join("backend", "models", "model.pkl"))
    local_features_path = os.getenv("FEATURES_PATH", os.path.join("backend", "data", "features.csv"))

    # Also consider same-folder simple names if present
    local_model_candidates = [
        os.path.join(APP_ROOT, local_model_path),
        os.path.join(APP_ROOT, "model.pkl"),
    ]
    local_features_candidates = [
        os.path.join(APP_ROOT, local_features_path),
        os.path.join(APP_ROOT, "features.csv"),
    ]

    # ---------- Blob config
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    blob_container = os.getenv("AZURE_STORAGE_CONTAINER", "").strip()
    model_blob = os.getenv("MODEL_BLOB_NAME", "model.pkl").strip()
    features_blob = os.getenv("FEATURES_BLOB_NAME", "features.csv").strip()
    stores_blob = os.getenv("STORES_BLOB_NAME", "stores.json").strip()

    def blob_configured() -> bool:
        return bool(conn_str and blob_container)

    # ---------- Optional fallback for /api/stores while warming
    try:
        FALLBACK_STORES = json.loads(os.getenv("FALLBACK_STORES_JSON", "[]"))
        if not isinstance(FALLBACK_STORES, list):
            FALLBACK_STORES = []
    except Exception:
        FALLBACK_STORES = []

    def _try_load_stores_json_from_blob() -> List[Dict[str, Any]]:
        if not blob_configured():
            return []
        try:
            b = _download_blob_bytes(
                connection_string=conn_str,
                container=blob_container,
                blob_name=stores_blob,
            )
            data = json.loads(b.decode("utf-8"))
            if isinstance(data, list) and data:
                if isinstance(data[0], str):
                    return [{"id": s, "name": f"Store {s}"} for s in data]
                if isinstance(data[0], dict) and "id" in data[0]:
                    return data
        except Exception:
            pass
        return []

    # --------------------------------------------------------
    # Warm-up: run once in background so startup is quick
    # --------------------------------------------------------
    def warm_start():
        if app.config["WARM"]["started"]:
            return
        app.config["WARM"]["started"] = True
        app.config["STATUS"] = "warming"
        app.logger.info("Warm-up: starting (blob_container=%s)", blob_container or "<none>")

        try:
            # Seed cache quickly from stores.json in Blob (if any)
            if blob_configured():
                quick_stores = _try_load_stores_json_from_blob()
                if quick_stores:
                    app.config["STORE_LIST_CACHE"] = quick_stores
                    app.logger.info("Warm-up: primed store cache from stores.json (%d)", len(quick_stores))

            model = None
            df = None

            # 1) Prefer local artifacts (if both exist)
            lm = next((p for p in local_model_candidates if os.path.exists(p)), None)
            lf = next((p for p in local_features_candidates if os.path.exists(p)), None)

            if lm and lf:
                app.logger.info("Warm-up: loading local artifacts (model=%s, features=%s)", lm, lf)
                model = joblib.load(lm)
                df = pd.read_csv(lf, low_memory=False)

            # 2) Else try Blob (if configured)
            elif blob_configured():
                if not HAS_AZURE:
                    raise RuntimeError("Blob config provided but 'azure-storage-blob' is not installed")
                app.logger.info("Warm-up: downloading artifacts from Blob container '%s'", blob_container)
                model_bytes = _download_blob_bytes(
                    connection_string=conn_str, container=blob_container, blob_name=model_blob
                )
                feat_bytes = _download_blob_bytes(
                    connection_string=conn_str, container=blob_container, blob_name=features_blob
                )
                model = joblib.load(BytesIO(model_bytes))
                df = pd.read_csv(BytesIO(feat_bytes), low_memory=False)

            # 3) Else: start OK with no model/data
            else:
                app.logger.info("Warm-up: no local artifacts and no Blob config; starting without model/data")
                app.config.update(MODEL=None, FEATURES_DF=pd.DataFrame())
                app.config.update(STATUS="ok", LAST_ERROR=None)
                app.config["WARM"].update(done=True, error=None)
                return

            # Build store cache
            app.config["FEATURES_DF"] = df
            app.config["MODEL"] = model
            if not app.config.get("STORE_LIST_CACHE"):
                app.config["STORE_LIST_CACHE"] = _build_store_cache(df)

            app.config.update(STATUS="ok", LAST_ERROR=None)
            app.config["WARM"].update(done=True, error=None)
            app.logger.info("Warm-up: done (stores_cached=%d)", len(app.config["STORE_LIST_CACHE"]))

        except Exception as e:
            app.config.update(STATUS="error", LAST_ERROR=str(e))
            app.config["WARM"].update(done=True, error=str(e))
            app.logger.exception("Warm-up failed")

    # Kick it off once at import
    threading.Thread(target=warm_start, daemon=True).start()

    # If a worker recycles later and lost state, ensure warmup restarts automatically
    @app.before_request
    def _ensure_warm():
        if not app.config["WARM"]["started"]:
            threading.Thread(target=warm_start, daemon=True).start()

    # --------------------------------------------------------
    # Routes
    # --------------------------------------------------------
    @app.get("/api/health")
    def health():
        warm = app.config["WARM"]
        if app.config["STATUS"] == "ok":
            status = "ok"
        elif warm["started"] and not warm["done"]:
            status = "warming"
        elif warm["error"]:
            status = "error"
        else:
            status = "starting"

        return jsonify(
            service="backend",
            status=status,
            last_error=app.config.get("LAST_ERROR"),
            stores_cached=len(app.config.get("STORE_LIST_CACHE", [])),
        ), 200

    @app.get("/api/stores")
    def stores():
        # 1) Serve cache if available
        cache = app.config.get("STORE_LIST_CACHE") or []
        if cache:
            return jsonify({"source": "cache", "stores": cache}), 200

        # 2) If features already loaded, compute once and cache
        df = app.config.get("FEATURES_DF")
        if isinstance(df, pd.DataFrame) and not df.empty:
            cache = _build_store_cache(df)
            app.config["STORE_LIST_CACHE"] = cache
            return jsonify({"source": "computed", "stores": cache}), 200

        # 3) Warming: return fallback list (if provided) but never block
        if FALLBACK_STORES:
            return jsonify({
                "source": "fallback",
                "status": "warming",
                "stores": [{"id": s, "name": f"Store {s}"} for s in FALLBACK_STORES]
            }), 200

        return jsonify({"source": "warming", "stores": []}), 200

    @app.post("/api/forecast")
    def forecast():
        # Example stub — replace with your model’s feature engineering & predict
        if app.config.get("MODEL") is None:
            err = app.config.get("LAST_ERROR")
            msg = f"Model not ready{' - ' + err if err else ''}"
            return jsonify({"error": msg, "status": app.config.get("STATUS")}), 503

        payload = request.get_json(silent=True) or {}
        # TODO: transform payload -> feature vector(s) and call model.predict(...)
        return jsonify({"ok": True, "received": payload}), 200

    @app.get("/api/debug/columns")
    def debug_columns():
        df = app.config.get("FEATURES_DF")
        cols = [str(c) for c in getattr(df, "columns", [])]
        chosen = _resolve_store_column(df) if isinstance(df, pd.DataFrame) else None
        return jsonify(columns=cols, chosen_store_column=chosen), 200

    @app.get("/")
    def root():
        return "Hello from backend!"

    return app


app = create_app()

if __name__ == "__main__":
    # For local dev; on Azure, Gunicorn uses 'app:app'
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
