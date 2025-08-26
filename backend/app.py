import os
import json
import threading
from io import BytesIO
from typing import Optional, List, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib

# --------------------------------------------
# Optional Azure Blob dependency
# --------------------------------------------
try:
    from azure.storage.blob import BlobClient  # type: ignore
    HAS_AZURE = True
except Exception:
    HAS_AZURE = False


# --------------------------------------------
# Helpers
# --------------------------------------------
def _download_blob_bytes(
    *,
    connection_string: str,
    container: str,
    blob_name: str,
) -> bytes:
    if not HAS_AZURE:
        raise RuntimeError("azure-storage-blob is not installed")
    client = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container,
        blob_name=blob_name,
    )
    return client.download_blob().readall()


def _build_store_cache(df: pd.DataFrame, *, store_id_column: Optional[str]) -> List[Dict[str, Any]]:
    """
    Build a list of stores from the dataframe.

    Priority:
      1) If STORE_ID_COLUMN is provided (and exists), use it.
      2) else a single column named exactly one of:
         'store', 'store_id', 'store_number', 'store_num'
      3) else fallback to columns containing 'store', but IGNORE obvious
         aggregates/stat columns like mean/std/avg/min/max/median/etc.
    """
    if df is None or df.empty:
        return []

    # 1) Explicit column via env
    if store_id_column and store_id_column in df.columns:
        vals = (
            pd.Series(df[store_id_column])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        vals = sorted(vals)
        return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]

    # 2) Common id column names
    for c in df.columns:
        cl = c.lower()
        if cl in ("store", "store_id", "store_number", "store_num"):
            vals = (
                pd.Series(df[c])
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            vals = sorted(vals)
            return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]

    # 3) Fallback: any column containing 'store' but skip typical aggregate/stat columns
    skip_tokens = ("mean", "std", "avg", "median", "min", "max", "sum", "count", "rate", "pct", "percent")
    candidates = [
        c for c in df.columns
        if "store" in c.lower() and not any(tok in c.lower() for tok in skip_tokens)
    ]

    # In fallback mode we return column names (schema hints) rather than IDs
    return [{"id": c, "name": c.replace("_", " ").title()} for c in candidates[:5000]]


# --------------------------------------------
# App factory
# --------------------------------------------
def create_app() -> Flask:
    """
    Environment variables (Blob mode, optional):
      - AZURE_STORAGE_CONNECTION_STRING
      - AZURE_STORAGE_CONTAINER      (e.g., 'artifacts')
      - MODEL_BLOB_NAME              (default 'model.pkl')
      - FEATURES_BLOB_NAME           (default 'features.csv')
      - STORES_BLOB_NAME             (default 'stores.json', optional)

    Local mode (if files exist in the deployed site root):
      - MODEL_PATH                   (default 'backend/models/model.pkl')
      - FEATURES_PATH                (default 'backend/data/features.csv')

    Store ID selection:
      - STORE_ID_COLUMN              (exact header for the store id column, e.g. 'Store Number')

    Behavior:
      * If local files exist -> load them.
      * Else, if Blob vars are present -> download & load.
      * Else -> start OK with no model/data (health shows ok; /api/forecast returns 503).
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

    # Local paths (relative to site root by default)
    local_model_path = os.getenv("MODEL_PATH", "backend/models/model.pkl")
    local_features_path = os.getenv("FEATURES_PATH", "backend/data/features.csv")

    # Blob config
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    blob_container = os.getenv("AZURE_STORAGE_CONTAINER", "").strip()

    model_blob = os.getenv("MODEL_BLOB_NAME", "model.pkl").strip()
    features_blob = os.getenv("FEATURES_BLOB_NAME", "features.csv").strip()
    stores_blob = os.getenv("STORES_BLOB_NAME", "stores.json").strip()

    # Store id column (explicit)
    store_id_column = os.getenv("STORE_ID_COLUMN")

    # Optional quick fallback for stores
    try:
        FALLBACK_STORES = json.loads(os.getenv("FALLBACK_STORES_JSON", "[]"))
        if not isinstance(FALLBACK_STORES, list):
            FALLBACK_STORES = []
    except Exception:
        FALLBACK_STORES = []

    def _blob_configured() -> bool:
        return bool(conn_str and blob_container)

    def _try_load_stores_json_from_blob() -> List[Dict[str, Any]]:
        """Fast path: small stores.json in the same container."""
        if not _blob_configured():
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

    # --------------------------
    # Warm-up loader (background)
    # --------------------------
    def warm_start():
        if app.config["WARM"]["started"]:
            return
        app.config["WARM"]["started"] = True
        app.config["STATUS"] = "warming"
        app.logger.info(
            "Warm-up: starting | blob_container=%s | store_id_column=%s",
            blob_container or "<none>", store_id_column or "<auto>"
        )

        try:
            # Seed store list ASAP from Blob stores.json (if configured/present)
            if _blob_configured():
                quick_stores = _try_load_stores_json_from_blob()
                if quick_stores:
                    app.config["STORE_LIST_CACHE"] = quick_stores
                    app.logger.info("Warm-up: primed store cache from stores.json (%d)", len(quick_stores))

            model = None
            df = None

            # 1) Prefer local artifacts if both exist
            if os.path.exists(local_model_path) and os.path.exists(local_features_path):
                app.logger.info("Warm-up: loading local artifacts")
                model = joblib.load(local_model_path)
                df = pd.read_csv(local_features_path, low_memory=False)

            # 2) Else try Blob (if configured)
            elif _blob_configured():
                if not HAS_AZURE:
                    raise RuntimeError("Blob config provided but 'azure-storage-blob' is not installed")
                app.logger.info("Warm-up: downloading artifacts from Blob container '%s'", blob_container)
                model_bytes = _download_blob_bytes(
                    connection_string=conn_str,
                    container=blob_container,
                    blob_name=model_blob,
                )
                feat_bytes = _download_blob_bytes(
                    connection_string=conn_str,
                    container=blob_container,
                    blob_name=features_blob,
                )
                model = joblib.load(BytesIO(model_bytes))
                df = pd.read_csv(BytesIO(feat_bytes), low_memory=False)

            # 3) Else start OK with no model/data
            else:
                app.logger.info("Warm-up: no local artifacts and no Blob config; starting without model/data")
                app.config.update(MODEL=None, FEATURES_DF=pd.DataFrame())
                app.config.update(STATUS="ok", LAST_ERROR=None)
                app.config["WARM"].update(done=True, error=None)
                return

            # Build store list if not already set
            # Build/override store list from CSV once it's loaded
            computed = _build_store_cache(df, store_id_column=store_id_column)
            if computed:
                app.config["STORE_LIST_CACHE"] = computed


            app.config.update(MODEL=model, FEATURES_DF=df, STATUS="ok", LAST_ERROR=None)
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

    # --------------------------
    # Routes
    # --------------------------
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
            cache = _build_store_cache(df, store_id_column=store_id_column)
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
        # TODO: transform payload -> features, then call app.config["MODEL"].predict(...)
        return jsonify({"ok": True, "received": payload}), 200

    @app.get("/")
    def root():
        return "Hello from backend!"

    return app


app = create_app()

if __name__ == "__main__":
    # For local dev; on Azure, Gunicorn will use app:app
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
