import os
import json
from io import BytesIO
from typing import Optional

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
        # If blob_url already includes ?sv=..., no credential needed
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

    app.config["MODEL"] = None
    app.config["FEATURES_DF"] = None
    app.config["STATUS"] = "init"
    app.config["LAST_ERROR"] = None

    # Local paths (dev)
    local_model_path = os.getenv("MODEL_PATH", os.getenv("MODEL_FILE", "backend/models/model.pkl"))
    local_features_path = os.getenv("FEATURES_PATH", os.getenv("FEATURES_FILE", "backend/data/features.csv"))

    # Blob (Option A: SAS)
    blob_base_url = os.getenv("BLOB_BASE_URL")  # e.g., https://...blob.core.windows.net/artifacts
    blob_sas = os.getenv("BLOB_SAS")            # just the query string (no leading '?')
    model_blob = os.getenv("MODEL_BLOB", "model.pkl")
    features_blob = os.getenv("FEATURES_BLOB", "features.csv")

    # Blob (Option B: Connection string)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_container = os.getenv("BLOB_CONTAINER")

    def load_artifacts():
        # Try local first (keeps your current local workflow working)
        if os.path.exists(local_model_path) and os.path.exists(local_features_path):
            app.config["MODEL"] = joblib.load(local_model_path)
            app.config["FEATURES_DF"] = pd.read_csv(local_features_path, low_memory=False)
            app.config["STATUS"] = "ok"
            app.config["LAST_ERROR"] = None
            return

        # Else try Blob Storage
        if not HAS_AZURE:
            raise RuntimeError("azure-storage-blob not installed and local files not found")

        # Build download approach
        # Option B: connection string takes precedence if set
        if conn_str and blob_container:
            model_bytes = download_blob_bytes(
                connection_string=conn_str,
                container=blob_container,
                blob_name=model_blob,
            )
            feat_bytes = download_blob_bytes(
                connection_string=conn_str,
                container=blob_container,
                blob_name=features_blob,
            )
        else:
            # Option A: base URL + SAS
            if not (blob_base_url and blob_sas):
                raise RuntimeError("Blob config missing (set BLOB_BASE_URL & BLOB_SAS or a connection string).")
            # Compose full URLs (include '?' yourself)
            model_url = f"{blob_base_url}/{model_blob}?{blob_sas}"
            feat_url = f"{blob_base_url}/{features_blob}?{blob_sas}"
            model_bytes = download_blob_bytes(blob_url=model_url)
            feat_bytes = download_blob_bytes(blob_url=feat_url)

        # Load artifacts into memory
        app.config["MODEL"] = joblib.load(BytesIO(model_bytes))
        app.config["FEATURES_DF"] = pd.read_csv(BytesIO(feat_bytes), low_memory=False)
        app.config["STATUS"] = "ok"
        app.config["LAST_ERROR"] = None

    # Try loading once at startup; if it fails, /api/health will report the error.
    try:
        load_artifacts()
    except Exception as e:
        app.config["STATUS"] = "init"
        app.config["LAST_ERROR"] = str(e)

    # --------------------------
    # Routes
    # --------------------------
    @app.route("/api/health")
    def health():
        df = app.config.get("FEATURES_DF")
        return jsonify({
            "service": "backend",
            "status": app.config.get("STATUS"),
            "last_error": app.config.get("LAST_ERROR"),
            "n_features": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
        })

    @app.route("/api/stores")
    def stores():
        """
        Returns list of stores. If a column named 'store' (case-insensitive) exists,
        use its unique values. Otherwise, fall back to heuristics.
        """
        df = app.config.get("FEATURES_DF")
        if not isinstance(df, pd.DataFrame):
            return jsonify({"stores": [], "source": "none"})

        # Look for store-like column
        store_col = None
        for c in df.columns:
            if c.lower() in ("store", "store_id", "store_number", "store_num"):
                store_col = c
                break

        if store_col:
            values = sorted({str(v) for v in df[store_col].dropna().unique().tolist()})
            return jsonify({
                "source": f"blob:{store_col}",
                "stores": [{"id": v, "name": f"Store {v}"} for v in values[:5000]]  # safety cap
            })

        # Fallback: previous heuristic
        candidates = []
        for c in df.columns:
            if "store" in c.lower() or ("Number" in c and df[c].dtype != "O"):
                candidates.append(c)
        if candidates:
            stores = [{"id": c, "name": c.replace("_", " ").title()} for c in candidates]
            return jsonify({"source": "onehot_features", "stores": stores})

        return jsonify({"source": "unknown", "stores": []})

    @app.route("/api/forecast", methods=["POST"])
    def forecast():
        """
        Example: expects JSON with whatever features your model needs.
        This stub echoes back and ensures model is loaded.
        """
        if app.config.get("MODEL") is None:
            # Attempt lazy-load if startup failed
            try:
                load_artifacts()
            except Exception as e:
                return jsonify({"error": f"Model not ready: {e}"}), 503

        payload = request.get_json(silent=True) or {}
        # TODO: transform payload -> feature vector(s) for your model
        # pred = app.config["MODEL"].predict(...)
        return jsonify({"ok": True, "received": payload})

    @app.route("/")
    def root():
        return "Hello from backend!"

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
