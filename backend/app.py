import os
import json
import threading
from io import BytesIO
from typing import Optional, List, Dict, Any

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
    """
    Download a blob's content as bytes.
    Supports:
      - full blob_url with SAS embedded
      - account_url + container + blob + sas_token
      - connection_string + container + blob
    """
    if not HAS_AZURE:
        raise RuntimeError("azure-storage-blob not installed")

    if blob_url:
        client = BlobClient.from_blob_url(blob_url)
    elif connection_string:
        if not (container and blob_name):
            raise ValueError("container and blob_name required with connection_string")
        client = BlobClient.from_connection_string(
            conn_str=connection_string,
            container_name=container,
            blob_name=blob_name,
        )
    else:
        if not (account_url and container and blob_name and sas_token):
            raise ValueError("account_url, container, blob_name, sas_token are required")
        client = BlobClient(
            account_url=account_url,
            container_name=container,
            blob_name=blob_name,
            credential=sas_token,
        )

    stream = client.download_blob()
    return stream.readall()


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
        STORE_LIST_CACHE=[],   # list[{"id": "...", "name": "..."}]
        STATUS="starting",     # starting | warming | ok | error
        LAST_ERROR=None,
        WARM=dict(started=False, done=False, error=None),
    )

    # --- Local dev paths (optional) ---
    local_model_path = os.getenv("MODEL_PATH", os.getenv("MODEL_FILE", "backend/models/model.pkl"))
    local_features_path = os.getenv("FEATURES_PATH", os.getenv("FEATURES_FILE", "backend/data/features.csv"))

    # --- Blob config (Option A: SAS URL parts) ---
    blob_base_url = os.getenv("BLOB_BASE_URL")  # e.g. https://<acct>.blob.core.windows.net/artifacts
    blob_sas = os.getenv("BLOB_SAS")            # query string without leading '?'

    # --- Blob config (Option B: connection string) ---
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_container = os.getenv("AZURE_STORAGE_CONTAINER") or os.getenv("BLOB_CONTAINER")

    # --- Blob names (align with your facts; keep old names as fallback) ---
    model_blob = os.getenv("MODEL_BLOB_NAME") or os.getenv("MODEL_BLOB") or "model.pkl"
    features_blob = os.getenv("FEATURES_BLOB_NAME") or os.getenv("FEATURES_BLOB") or "features.csv"

    # --- Optional tiny fallback list while warming (JSON string: ["2327","2200"]) ---
    fallback_stores_env = os.getenv("FALLBACK_STORES_JSON", "[]")
    try:
        FALLBACK_STORES: List[str] = json.loads(fallback_stores_env)
    except Exception:
        FALLBACK_STORES = []

    def _build_store_cache(df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Try to find the store id column
        store_col = None
        for c in df.columns:
            if c.lower() in ("store", "store_id", "store_number", "store_num"):
                store_col = c
                break

        if store_col:
            vals = pd.Series(df[store_col]).dropna().astype(str).unique().tolist()
            vals = sorted(vals)
            return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]  # safety cap

        # Fallback: expose any column names that look store-related
        candidates = []
        for c in df.columns:
            if "store" in c.lower():
                candidates.append(c)
        return [{"id": c, "name": c.replace("_", " ").title()} for c in candidates]

    def _download_artifacts() -> tuple[bytes, bytes]:
        # Prefer connection string if present
        if conn_str and blob_container:
            model_bytes = download_blob_bytes(
                connection_string=conn_str, container=blob_container, blob_name=model_blob
            )
            feat_bytes = download_blob_bytes(
                connection_string=conn_str, container=blob_container, blob_name=features_blob
            )
            return model_bytes, feat_bytes

        # Else use base + SAS
        if not (blob_base_url and blob_sas):
            raise RuntimeError("Blob config missing (set AZURE_STORAGE_CONNECTION_STRING + AZURE_STORAGE_CONTAINER "
                               "OR BLOB_BASE_URL + BLOB_SAS).")
        model_url = f"{blob_base_url}/{model_blob}?{blob_sas}"
        feat_url = f"{blob_base_url}/{features_blob}?{blob_sas}"
        return download_blob_bytes(blob_url=model_url), download_blob_bytes(blob_url=feat_url)

    # --------------------------
    # Background warm-up
    # --------------------------
    def warm_start():
        app.config["WARM"]["started"] = True
        app.logger.info("Warm-up: starting")

        try:
            # 1) Local fast path (keeps your local workflow intact)
            if os.path.exists(local_model_path) and os.path.exists(local_features_path):
                app.logger.info("Warm-up: loading local artifacts")
                model = joblib.load(local_model_path)
                df = pd.read_csv(local_features_path, low_memory=False)
            else:
                # 2) Download from Blob Storage
                if not HAS_AZURE:
                    raise RuntimeError("azure-storage-blob not installed and local files not found")
                app.logger.info("Warm-up: downloading artifacts from Azure Blob")
                model_bytes, feat_bytes = _download_artifacts()
                app.logger.info("Warm-up: loading model.pkl")
                model = joblib.load(BytesIO(model_bytes))
                app.logger.info("Warm-up: reading features.csv")
                df = pd.read_csv(BytesIO(feat_bytes), low_memory=False)

            # 3) Build store cache (lightweight)
            store_cache = _build_store_cache(df)

            # 4) Publish to app state
            app.config.update(
                MODEL=model,
                FEATURES_DF=df,
                STORE_LIST_CACHE=store_cache,
                STATUS="ok",
                LAST_ERROR=None,
            )
            app.config["WARM"].update(done=True, error=None)
            app.logger.info(f"Warm-up: done (stores_cached={len(store_cache)})")

        except Exception as e:
            app.config.update(STATUS="error", LAST_ERROR=str(e))
            app.config["WARM"].update(done=True, error=str(e))
            app.logger.exception("Warm-up failed")

    # Start warm-up in the background (non-blocking)
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
        # Serve cached if available
        cache = app.config.get("STORE_LIST_CACHE") or []
        if cache:
            return jsonify({"source": "cache", "stores": cache}), 200

        # If FEATURES_DF is ready, build now (first call after warm-up)
        df = app.config.get("FEATURES_DF")
        if isinstance(df, pd.DataFrame):
            cache = _build_store_cache(df)
            app.config["STORE_LIST_CACHE"] = cache
            return jsonify({"source": "computed", "stores": cache}), 200

        # Fallback while warming
        if FALLBACK_STORES:
            return jsonify({
                "source": "fallback",
                "status": "warming",
                "stores": [{"id": s, "name": f"Store {s}"} for s in FALLBACK_STORES]
            }), 200

        return jsonify({"source": "none", "status": "warming", "stores": []}), 200

    @app.post("/api/forecast")
    def forecast():
        if app.config.get("MODEL") is None:
            # Still warming or failed
            err = app.config.get("LAST_ERROR")
            msg = f"Model not ready{' - ' + err if err else ''}"
            return jsonify({"error": msg, "status": app.config.get("STATUS")}), 503

        payload = request.get_json(silent=True) or {}
        # TODO: transform payload -> feature vector(s) for your model
        # y_pred = app.config["MODEL"].predict(...)
        return jsonify({"ok": True, "received": payload}), 200

    @app.get("/")
    def root():
        return "Hello from backend!"

    return app


app = create_app()

if __name__ == "__main__":
    # PORT is provided by Azure App Service; default to 5000 for local dev
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
