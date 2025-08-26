import os
import json
import threading
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib

# Optional import; only needed when using blob storage
try:
    from azure.storage.blob import BlobClient
    HAS_AZURE = True
except Exception:
    HAS_AZURE = False


# --------------------------
# Helpers: Blob download
# --------------------------
def download_blob_bytes(
    *,
    blob_url: Optional[str] = None,
    account_url: Optional[str] = None,
    container: Optional[str] = None,
    blob_name: Optional[str] = None,
    sas_token: Optional[str] = None,
    connection_string: Optional[str] = None,
) -> bytes:
    if not HAS_AZURE:
        raise RuntimeError("azure-storage-blob not installed")

    if blob_url:
        client = BlobClient.from_blob_url(blob_url)
    elif connection_string:
        if not (container and blob_name):
            raise ValueError("container and blob_name required with connection_string")
        client = BlobClient.from_connection_string(
            conn_str=connection_string, container_name=container, blob_name=blob_name
        )
    else:
        if not (account_url and container and blob_name and sas_token):
            raise ValueError("account_url, container, blob_name, sas_token are required")
        client = BlobClient(
            account_url=account_url, container_name=container, blob_name=blob_name, credential=sas_token
        )
    return client.download_blob().readall()


# --------------------------
# App Factory
# --------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Runtime state
    app.config.update(
        MODEL=None,
        FEATURES_DF=None,
        STORE_LIST_CACHE=[],   # [{"id": "...", "name": "..."}]
        STATUS="starting",     # starting | warming | ok | error
        LAST_ERROR=None,
        WARM=dict(started=False, done=False, error=None),
    )

    # Local dev paths
    local_model_path = os.getenv("MODEL_PATH", "backend/models/model.pkl")
    local_features_path = os.getenv("FEATURES_PATH", "backend/data/features.csv")

    # Connection string mode (preferred)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_container = os.getenv("AZURE_STORAGE_CONTAINER") or os.getenv("BLOB_CONTAINER")

    # Blob names
    model_blob = os.getenv("MODEL_BLOB_NAME") or os.getenv("MODEL_BLOB") or "model.pkl"
    features_blob = os.getenv("FEATURES_BLOB_NAME") or os.getenv("FEATURES_BLOB") or "features.csv"
    stores_blob = os.getenv("STORES_BLOB_NAME", "stores.json")  # optional tiny file

    # Optional quick fallback while warming
    try:
        FALLBACK_STORES: List[str] = json.loads(os.getenv("FALLBACK_STORES_JSON", "[]"))
    except Exception:
        FALLBACK_STORES = []

    def _download_from_conn(blob_name: str) -> bytes:
        if not (conn_str and blob_container):
            raise RuntimeError("Connection string mode requires AZURE_STORAGE_CONNECTION_STRING and container")
        return download_blob_bytes(connection_string=conn_str, container=blob_container, blob_name=blob_name)

    def _build_store_cache(df: pd.DataFrame) -> List[Dict[str, Any]]:
        store_col = None
        for c in df.columns:
            if c.lower() in ("store", "store_id", "store_number", "store_num"):
                store_col = c
                break
        if store_col:
            vals = pd.Series(df[store_col]).dropna().astype(str).unique().tolist()
            vals = sorted(vals)
            return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]
        # fallback: columns that look like one-hots
        candidates = [c for c in df.columns if "store" in c.lower()]
        return [{"id": c, "name": c.replace("_", " ").title()} for c in candidates]

    def _try_load_stores_json() -> List[Dict[str, Any]]:
        """Fast path: tiny stores.json in the same container, [{"id":"2327","name":"Store 2327"}, ...]."""
        try:
            b = _download_from_conn(stores_blob)
            lst = json.loads(b.decode("utf-8"))
            # Accept either ["2327","2200"] or [{"id":"2327","name":"Store 2327"}, ...]
            if lst and isinstance(lst[0], str):
                return [{"id": s, "name": f"Store {s}"} for s in lst]
            if lst and isinstance(lst[0], dict) and "id" in lst[0]:
                return lst
        except Exception:
            pass
        return []

    def warm_start():
        app.config["WARM"]["started"] = True
        app.config["STATUS"] = "warming"
        app.logger.info("Warm-up: starting (container=%s)", blob_container)

        try:
            # 0) Prime store list fast if a tiny stores.json exists in Blob
            quick_stores = _try_load_stores_json()
            if quick_stores:
                app.config["STORE_LIST_CACHE"] = quick_stores
                app.logger.info("Warm-up: primed store cache from stores.json (%d)", len(quick_stores))

            model: Any = None
            df: pd.DataFrame | None = None

            # 1) Prefer local artifacts if they exist (works without any Azure config)
            if os.path.exists(local_model_path) and os.path.exists(local_features_path):
                app.logger.info("Warm-up: loading local artifacts")
                model = joblib.load(local_model_path)
                df = pd.read_csv(local_features_path, low_memory=False)

            else:
                # 2) If local files are missing, use Blob only when fully configured
                if conn_str and blob_container:
                    if not HAS_AZURE:
                        raise RuntimeError("azure-storage-blob not installed but Blob config provided")
                    app.logger.info("Warm-up: downloading artifacts from Blob container '%s'", blob_container)
                    model_bytes = _download_from_conn(model_blob)
                    feat_bytes = _download_from_conn(features_blob)
                    model = joblib.load(BytesIO(model_bytes))
                    df = pd.read_csv(BytesIO(feat_bytes), low_memory=False)
                else:
                    # 3) No local files and no Blob config -> start without model/data (still healthy)
                    app.logger.info("Warm-up: no local artifacts and no storage configured; starting without model/data")
                    app.config.update(MODEL=None, FEATURES_DF=pd.DataFrame(), STATUS="ok", LAST_ERROR=None)
                    app.config["WARM"].update(done=True, error=None)
                    return

            # 4) Build store cache if we haven't already (e.g., no stores.json)
            if not app.config.get("STORE_LIST_CACHE"):
                store_cache = _build_store_cache(df)
                app.config["STORE_LIST_CACHE"] = store_cache

            # 5) Publish artifacts and finish
            app.config.update(MODEL=model, FEATURES_DF=df, STATUS="ok", LAST_ERROR=None)
            app.config["WARM"].update(done=True, error=None)
            app.logger.info("Warm-up: done (stores_cached=%d)", len(app.config["STORE_LIST_CACHE"]))

        except Exception as e:
            app.config.update(STATUS="error", LAST_ERROR=str(e))
            app.config["WARM"].update(done=True, error=str(e))
            app.logger.exception("Warm-up failed")


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
        # 1) Serve cache if available (instant)
        cache = app.config.get("STORE_LIST_CACHE") or []
        if cache:
            return jsonify({"source": "cache", "stores": cache}), 200

        # 2) If features already loaded, compute once and cache (still reasonably fast)
        df = app.config.get("FEATURES_DF")
        if isinstance(df, pd.DataFrame):
            cache = _build_store_cache(df)
            app.config["STORE_LIST_CACHE"] = cache
            return jsonify({"source": "computed", "stores": cache}), 200

        # 3) Warming: return quick fallback, never block
        if FALLBACK_STORES:
            return jsonify({
                "source": "fallback",
                "status": "warming",
                "stores": [{"id": s, "name": f"Store {s}"} for s in FALLBACK_STORES]
            }), 200

        return jsonify({"source": "warming", "stores": []}), 200

    @app.post("/api/forecast")
    def forecast():
        if app.config.get("MODEL") is None:
            err = app.config.get("LAST_ERROR")
            msg = f"Model not ready{' - ' + err if err else ''}"
            return jsonify({"error": msg, "status": app.config.get("STATUS")}), 503

        payload = request.get_json(silent=True) or {}
        # TODO: transform payload -> feature vector(s) for your model
        return jsonify({"ok": True, "received": payload}), 200

    @app.get("/")
    def root():
        return "Hello from backend!"

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
