import { useEffect, useMemo, useState } from "react";
import ForecastChart from "./components/ForecastChart";

/** Build the API base once, with sensible fallbacks. (NO hooks here) */
function buildApiBase() {
  // Prefer VITE_API_BASE; keep backward compat with VITE_API_URL
  const rawEnv =
    (import.meta.env.VITE_API_BASE ||
      import.meta.env.VITE_API_URL ||
      "").toString().trim();

  // Optional runtime override if you ever set window.API_BASE in index.html
  const runtime =
    typeof window !== "undefined" && window.API_BASE ? window.API_BASE : "";

  const picked = (runtime || rawEnv).replace(/\/+$/, ""); // drop trailing '/'

  if (picked) {
    return picked.endsWith("/api") ? picked : `${picked}/api`;
  }

  // Absolute default to your working App Service backend:
  return "https://forecasting-ai-powered-ddarbmdqe3dzdccg.canadacentral-01.azurewebsites.net/api";
}

const API_BASE = buildApiBase();

async function getJSON(path) {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export default function App() {
  // ---------- original app state ----------
  const [stores, setStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState("");
  const [debugOut, setDebugOut] = useState("");
  const [healthMsg, setHealthMsg] = useState("loading…");
  const [loadingStores, setLoadingStores] = useState(true);
  const [storesError, setStoresError] = useState("");

  // ---------- forecasting state (moved out of buildApiBase) ----------
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [forecastStatus, setForecastStatus] = useState("idle"); // idle|loading|ready|error
  const [forecastErr, setForecastErr] = useState("");

  // Fallback only if API fails/returns empty
  const fallbackStores = useMemo(
    () => [
      { id: "001", name: "Store 001" },
      { id: "002", name: "Store 002" },
      { id: "003", name: "Store 003" },
    ],
    []
  );

  // Fetch health + stores
  useEffect(() => {
    // health
    getJSON("/health")
      .then((h) => setHealthMsg(h.status || JSON.stringify(h)))
      .catch((e) => setHealthMsg(`error: ${e.message}`));

    // stores
    setLoadingStores(true);
    setStoresError("");
    getJSON("/stores")
      .then((data) => {
        const list = Array.isArray(data?.stores) ? data.stores : [];
        if (list.length > 0) {
          setStores(list);
          setSelectedStore(list[0].id);
        } else {
          setStores(fallbackStores);
          setSelectedStore(fallbackStores[0].id);
        }
      })
      .catch((e) => {
        setStoresError(e.message);
        setStores(fallbackStores);
        setSelectedStore(fallbackStores[0].id);
      })
      .finally(() => setLoadingStores(false));
  }, [fallbackStores]);

  // Fetch forecast when a store is selected
  // replace your current forecast useEffect with this version
useEffect(() => {
  if (!selectedStore) return;

  let cancelled = false;
  setForecastStatus("loading");
  setForecastErr("");

  (async () => {
    try {
      const url = `${API_BASE}/forecast`;
      const res = await fetch(url, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ store_id: selectedStore }),
      });

      if (!res.ok) {
        // e.g., 503 Model not ready, or other error
        const msg = `${res.status} ${res.statusText}`;
        throw new Error(msg);
      }

      const data = await res.json();

      // Accept the current stub shape { ok: true, received: {...} }
      // and normalize future real shape { history: [...], forecast: [...] }
      const h = Array.isArray(data?.history) ? data.history : [];
      const f = Array.isArray(data?.forecast) ? data.forecast : [];

      if (!cancelled) {
        setHistory(h);
        setForecast(f);
        setForecastStatus("ready");
        // Optional: show what the server echoed for debugging
        console.log("forecast payload received:", data);
      }
    } catch (err) {
      if (!cancelled) {
        setForecastErr(err.message || String(err));
        setForecastStatus("error");
      }
    }
  })();

  return () => { cancelled = true; };
}, [selectedStore]);




  // Use a real backend route: /api/debug/columns
  const callDebug = async () => {
    try {
      const data = await getJSON(`/debug/columns`);
      setDebugOut(JSON.stringify(data, null, 2));
    } catch (e) {
      setDebugOut(`error: ${e.message}`);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", margin: "2rem auto", maxWidth: 720 }}>
      <h1>Frontend ready</h1>
      <p>SWA → Azure App Service API test</p>

      <div style={{ padding: "1rem", border: "1px solid #ddd", borderRadius: 8, marginBottom: 16 }}>
        <strong>API health:</strong> {healthMsg}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <label htmlFor="store">Store:</label>
        <select
          id="store"
          value={selectedStore}
          disabled={loadingStores}
          onChange={(e) => setSelectedStore(e.target.value)}
        >
          {stores.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name ?? s.id}
            </option>
          ))}
        </select>

        <button onClick={callDebug} disabled={!selectedStore}>
          Test /debug/columns
        </button>
      </div>

      {selectedStore && (
        <>
          {forecastStatus === "loading" && <p>Loading forecast…</p>}
          {forecastStatus === "error" && (
            <p style={{ color: "#b00" }}>Failed to load forecast: {forecastErr}</p>
          )}
          {(forecastStatus === "ready" || forecastStatus === "idle") && (
            <ForecastChart history={history} forecast={forecast} />
          )}
        </>
      )}

      {storesError && (
        <p style={{ color: "#b00", marginTop: 8 }}>
          (stores API error: {storesError} — showing fallback)
        </p>
      )}

      <pre style={{ marginTop: 16, background: "#f8f8f8", padding: 12, borderRadius: 6 }}>
        {debugOut}
      </pre>

      <div style={{ opacity: 0.6, marginTop: 24, fontSize: 12 }}>
        API base: <code>{API_BASE}</code>
      </div>
    </div>
  );
}
