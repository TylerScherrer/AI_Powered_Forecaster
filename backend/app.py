# app.py
import os
import json
import threading
from io import BytesIO
from typing import Optional, List, Dict, Any
import datetime as _dt
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from math import isfinite

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
def _store_key(x) -> str:
    s = str(x).strip()
    # collapse numeric-looking floats: "2553.0" -> "2553"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _pick_cols(df: pd.DataFrame):
    """Pick (store_col, date_col, y_col) using env overrides first, then heuristics."""
    if df is None or df.empty:
        return None, None, None

    # Env overrides (optional)
    store_override = os.getenv("STORE_ID_COLUMN", "").strip()
    date_override  = os.getenv("DATE_COLUMN", "").strip()
    y_override     = os.getenv("TARGET_COLUMN", "").strip()

    store_col = _resolve_store_column(df)
    if store_override and store_override in df.columns:
        store_col = store_override

    # heuristics for date / target
    norm = lambda c: str(c).strip().lower().replace(" ", "_")
    candidates_date = {"date", "ds", "month", "period", "order_date", "txn_date"}
    candidates_y    = {"total", "sales", "y", "target", "revenue", "amount", "qty", "units"}

    date_col = next((c for c in df.columns if norm(c) in candidates_date), None)
    y_col    = next((c for c in df.columns if norm(c) in candidates_y), None)

    if date_override and date_override in df.columns:
        date_col = date_override
    if y_override and y_override in df.columns:
        y_col = y_override

    return store_col, date_col, y_col


def _monthly_history_for_store(df, store_col, date_col, y_col, store_id):
    key = _store_key(store_id)  # << normalize requested id
    sdf = df[pd.Series(df[store_col]).map(_store_key) == key].copy()
    if sdf.empty:
        return []

    # more forgiving date parsing
    dt = pd.to_datetime(sdf[date_col], errors="coerce", infer_datetime_format=True)
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(sdf[date_col].astype(str), format="%Y-%m", errors="coerce"))
        dt = dt.fillna(pd.to_datetime(sdf[date_col].astype(str), format="%Y%m", errors="coerce"))
    sdf[date_col] = dt
    sdf = sdf.dropna(subset=[date_col, y_col])

    sdf[y_col] = pd.to_numeric(sdf[y_col], errors="coerce")
    sdf = sdf.dropna(subset=[y_col])

    m = (
        sdf.set_index(date_col)
           .groupby(pd.Grouper(freq="MS"))[y_col]
           .sum()
           .dropna()
           .rename("total")
           .reset_index()
    )
    m["date"] = m[date_col].dt.strftime("%Y-%m")
    return [{"date": r["date"], "total": float(r["total"])} for _, r in m.iterrows()]



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
      3) if exactly one column contains 'store', use it
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

    candidates = [c for c in cols if "store" in _norm(c)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _build_store_cache(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    store_col = _resolve_store_column(df)
    if store_col:
        vals = (
            pd.Series(df[store_col])
            .dropna()
            .map(_store_key)        # << normalize
            .unique()
            .tolist()
        )
        vals = sorted(vals)
        return [{"id": v, "name": f"Store {v}"} for v in vals[:5000]]

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
      * Else -> start OK with no model/data; health shows ok and endpoints still respond.
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
    local_model_path = os.getenv("MODEL_PATH", os.path.join("models", "model.pkl"))
    local_features_path = os.getenv("FEATURES_PATH", os.path.join("data", "features.csv"))

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
            # Optional quick prime from stores.json in Blob
            if blob_configured():
                try:
                    quick_stores = _try_load_stores_json_from_blob()
                    if quick_stores:
                        app.config["STORE_LIST_CACHE"] = quick_stores
                        app.logger.info(
                            "Warm-up: primed store cache from stores.json (%d)",
                            len(quick_stores),
                        )
                except Exception:
                    pass

            model = None
            df = None

            # ---- Local: allow features and/or model (features-only is OK)
            lf = next((p for p in local_features_candidates if os.path.exists(p)), None)
            if lf:
                app.logger.info("Warm-up: loading local features %s", lf)
                df = pd.read_csv(lf, low_memory=False)

            lm = next((p for p in local_model_candidates if os.path.exists(p)), None)
            if lm:
                app.logger.info("Warm-up: loading local model %s", lm)
                model = joblib.load(lm)

            # ---- Blob fallback for anything missing
            if df is None and blob_configured():
                app.logger.info(
                    "Warm-up: downloading features from Blob '%s'", features_blob
                )
                feat_bytes = _download_blob_bytes(
                    connection_string=conn_str,
                    container=blob_container,
                    blob_name=features_blob,
                )
                df = pd.read_csv(BytesIO(feat_bytes), low_memory=False)

            if model is None and blob_configured():
                try:
                    app.logger.info(
                        "Warm-up: downloading model from Blob '%s'", model_blob
                    )
                    model_bytes = _download_blob_bytes(
                        connection_string=conn_str,
                        container=blob_container,
                        blob_name=model_blob,
                    )
                    model = joblib.load(BytesIO(model_bytes))
                except Exception:
                    # OK if no model; we can still serve stores/endpoints that don't need it
                    app.logger.info(
                        "Warm-up: no model available; continuing without one"
                    )

            # ---- If we still have nothing, start "ok" with empty DF
            if df is None and model is None:
                app.logger.info(
                    "Warm-up: no artifacts found; starting without model/data"
                )
                app.config.update(
                    MODEL=None, FEATURES_DF=pd.DataFrame(), STATUS="ok", LAST_ERROR=None
                )
                app.config["WARM"].update(done=True, error=None)
                return

            # Build store cache if we have features
            if isinstance(df, pd.DataFrame) and not df.empty and not app.config.get(
                "STORE_LIST_CACHE"
            ):
                app.config["STORE_LIST_CACHE"] = _build_store_cache(df)

            app.config.update(
                MODEL=model,
                FEATURES_DF=(df if df is not None else pd.DataFrame()),
                STATUS="ok",
                LAST_ERROR=None,
            )
            app.config["WARM"].update(done=True, error=None)
            app.logger.info(
                "Warm-up: done (stores_cached=%d)",
                len(app.config["STORE_LIST_CACHE"]),
            )
        except Exception as e:
            app.config.update(STATUS="error", LAST_ERROR=str(e))
            app.config["WARM"].update(done=True, error=str(e))
            app.logger.exception("Warm-up failed")

    # Kick the warm-up once at import so /api/stores is ready soon
    threading.Thread(target=warm_start, daemon=True).start()

    # If a worker recycles later and lost state, ensure warmup restarts automatically
    @app.before_request
    def _ensure_warm():
        if not app.config["WARM"]["started"]:
            threading.Thread(target=warm_start, daemon=True).start()

    # --------------------------------------------------------
    # Routes
    # --------------------------------------------------------
    # ---------- DIAGNOSTICS (temporary endpoints) ----------
    @app.get("/api/debug/summary")
    def debug_summary():
        df = app.config.get("FEATURES_DF")
        rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        cols = list(getattr(df, "columns", []))
        sc, dc, yc = _pick_cols(df if rows else pd.DataFrame())

        def dtype_map():
            if not rows: return {}
            return {str(c): str(df[c].dtype) for c in cols[:50]}  # trim for size

        # sample store ids (raw and normalized)
        store_samples = []
        norm_store_samples = []
        unique_stores = None
        if rows and sc:
            s = pd.Series(df[sc]).dropna().astype(str).head(10)
            store_samples = s.tolist()
            try:
                norm_store_samples = s.map(_store_key).tolist()
                unique_stores = int(pd.Series(df[sc]).map(_store_key).nunique())
            except Exception:
                pass

        # date snapshot
        date_min = date_max = None
        if rows and dc:
            try:
                dt = pd.to_datetime(df[dc], errors="coerce")
                date_min = str(dt.min())
                date_max = str(dt.max())
            except Exception:
                pass

        # target snapshot
        target_min = target_max = None
        if rows and yc:
            try:
                y = pd.to_numeric(df[yc], errors="coerce")
                target_min = float(y.min())
                target_max = float(y.max())
            except Exception:
                pass

        return jsonify({
            "status": app.config.get("STATUS"),
            "has_model": app.config.get("MODEL") is not None,
            "features_rows": rows,
            "features_cols": len(cols),
            "chosen_columns": {"store": sc, "date": dc, "target": yc},
            "dtypes": dtype_map(),
            "store_samples_raw": store_samples,
            "store_samples_norm": norm_store_samples,
            "store_unique_norm": unique_stores,
            "date_min": date_min,
            "date_max": date_max,
            "target_min": target_min,
            "target_max": target_max,
            "stores_cached": len(app.config.get("STORE_LIST_CACHE") or []),
        }), 200


    @app.get("/api/debug/inspect")
    def debug_inspect():
        """?store_id=2602 checks row counts before/after normalization + date parse sample."""
        store_id = request.args.get("store_id", "").strip()
        df = app.config.get("FEATURES_DF")
        sc, dc, yc = _pick_cols(df if isinstance(df, pd.DataFrame) else pd.DataFrame())

        if not (isinstance(df, pd.DataFrame) and len(df) and sc and dc and yc and store_id):
            return jsonify({"ok": False, "reason": "missing df/columns/store_id"}), 200

        raw_match = df[df[sc].astype(str) == store_id]
        norm_match = df[pd.Series(df[sc]).map(_store_key) == _store_key(store_id)]

        # show a few dates before/after parsing
        sample = norm_match[[sc, dc, yc]].head(5).copy()
        try:
            parsed = pd.to_datetime(sample[dc], errors="coerce", infer_datetime_format=True)
            parsed2 = parsed.fillna(pd.to_datetime(sample[dc].astype(str), format="%Y-%m", errors="coerce"))
            parsed3 = parsed2.fillna(pd.to_datetime(sample[dc].astype(str), format="%Y%m", errors="coerce"))
            parsed_iso = [str(x) for x in parsed3]
        except Exception:
            parsed_iso = []

        return jsonify({
            "ok": True,
            "store_id_query": store_id,
            "chosen_columns": {"store": sc, "date": dc, "target": yc},
            "raw_match_rows": int(len(raw_match)),
            "normalized_match_rows": int(len(norm_match)),
            "date_parse_preview": parsed_iso,
            "sample_rows": sample.astype(str).to_dict(orient="records"),
        }), 200

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

        return (
            jsonify(
                service="backend",
                status=status,
                last_error=app.config.get("LAST_ERROR"),
                stores_cached=len(app.config.get("STORE_LIST_CACHE", [])),
            ),
            200,
        )

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
        try:
            fallback = FALLBACK_STORES
        except NameError:
            fallback = []
        if fallback:
            return (
                jsonify(
                    {
                        "source": "fallback",
                        "status": "warming",
                        "stores": [{"id": s, "name": f"Store {s}"} for s in fallback],
                    }
                ),
                200,
            )

        return jsonify({"source": "warming", "stores": []}), 200

    # ---- SINGLE FORECAST ROUTE (always returns a response)


    @app.route("/api/forecast", methods=["POST"], endpoint="api_forecast")
    def api_forecast():
        """
        POST JSON: { "store_id": "<id>", "horizon": 6 }
        Returns: { "store_id", "history": [...], "forecast": [...] }
        Uses FEATURES_DF; tries MODEL if available; otherwise naive seasonal/MA forecast.
        """
        payload = request.get_json(silent=True) or {}
        store_id = str(payload.get("store_id", "")).strip()
        horizon  = int(payload.get("horizon", 6))  # months to forecast

        df = app.config.get("FEATURES_DF")
        model = app.config.get("MODEL")

        # 1) Pick columns and build monthly history
        sc, dc, yc = _pick_cols(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
        if not (isinstance(df, pd.DataFrame) and not df.empty and sc and dc and yc and store_id):
            # fall back to deterministic stub if we can't build history yet
            base = 1000 + (abs(hash(store_id)) % 250)
            history = [
                {"date": "2023-01", "total": base},
                {"date": "2023-02", "total": int(base * 1.05)},
                {"date": "2023-03", "total": int(base * 0.98)},
            ]
            forecast_pts = [
                {"date": "2023-04", "total": int(history[-1]["total"] * 1.03)},
                {"date": "2023-05", "total": int(history[-1]["total"] * 1.06)},
            ][:horizon]
            return jsonify({"store_id": store_id, "history": history, "forecast": forecast_pts}), 200

        history = _monthly_history_for_store(df, sc, dc, yc, store_id)
        if not history:
            # no rows for that store
            return jsonify({"store_id": store_id, "history": [], "forecast": []}), 200

        # 2) Try MODEL if present (best-effort; keep it defensive)
        #    Expecting a regressor-like object with .predict(X_future)
        #    You'll need to adapt 'build_features_for_future' to your pipeline.
        try:
            if model is not None:
                # --- Example: build features for the next `horizon` months ---
                last_hist = pd.to_datetime(history[-1]["date"] + "-01")
                future_dates = [(last_hist + pd.DateOffset(months=i)).strftime("%Y-%m") 
                                for i in range(1, horizon + 1)]

                # This assumes you trained your model on rows from FEATURES_DF
                # with columns in app.config["MODEL_FEATURES"]
                feature_cols = getattr(model, "feature_names_in_", app.config.get("MODEL_FEATURES", []))
                if not feature_cols:
                    return jsonify({"error": "Model has no feature_names_in_"}), 500

                # Build feature DataFrame for the future horizon
                # TODO: adapt this to your real feature engineering
                X_future = pd.DataFrame([{
                    "Lag_1": history[-1]["total"],
                    "Lag_2": history[-2]["total"] if len(history) > 1 else history[-1]["total"],
                    "Lag_3": history[-3]["total"] if len(history) > 2 else history[-1]["total"],
                    "Month": int(d.split("-")[1]),
                    "Quarter": (int(d.split("-")[1]) - 1) // 3 + 1,
                } for d in future_dates])

                # Keep only model’s expected columns
                X_future = X_future[[c for c in feature_cols if c in X_future.columns]]

                yhat = model.predict(X_future)
                forecast_pts = [{"date": d, "total": float(v)} for d, v in zip(future_dates, yhat)]
                return jsonify({"store_id": store_id, "history": history, "forecast": forecast_pts}), 200

        except Exception as e:
            app.logger.warning("MODEL predict failed; falling back to naive forecast: %s", e)

        # 3) Naive seasonal / moving-average fallback (no extra deps)
        #    - If we have >= 12 months, use last year's same-month (seasonal naive)
        #    - Else use simple rolling average of last 3 months
        totals = [float(h["total"]) for h in history if isfinite(float(h["total"]))]
        # figure the last month as timestamp
        last_month = pd.to_datetime(history[-1]["date"] + "-01")
        dates = [(last_month + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, horizon + 1)]

        if len(totals) >= 12:
            # seasonal naive: y_{t+h} = y_{t-12+h}
            season_ref = totals[-12:]
            forecast_vals = [(season_ref[h % 12]) for h in range(horizon)]
            # small drift to avoid perfectly flat lines
            scale = totals[-1] / max(1e-9, season_ref[-1])
            forecast_vals = [float(v) * float(scale) for v in forecast_vals]
        else:
            # rolling average of last k (k=3 or len available)
            k = 3 if len(totals) >= 3 else max(1, len(totals))
            avg = sum(totals[-k:]) / k
            # gentle growth 1%/mo just to visualize trend
            forecast_vals = [avg * (1.01 ** i) for i in range(1, horizon + 1)]

        forecast_pts = [{"date": d, "total": float(v)} for d, v in zip(dates, forecast_vals)]
        return jsonify({"store_id": store_id, "history": history, "forecast": forecast_pts}), 200


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
