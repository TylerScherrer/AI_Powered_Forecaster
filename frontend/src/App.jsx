import { useEffect, useMemo, useState } from "react";

/** Build the API base once, with sensible fallbacks. */
function buildApiBase() {
  // Prefer VITE_API_BASE; keep backward compat with VITE_API_URL
  const rawEnv =
    (import.meta.env.VITE_API_BASE ||
      import.meta.env.VITE_API_URL ||
      "").toString().trim();

  // optional runtime override if you ever set window.API_BASE in index.html
  const runtime =
    (typeof window !== "undefined" && window.API_BASE) ? window.API_BASE : "";

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
  const [stores, setStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState("");
  const [debugOut, setDebugOut] = useState("");
  const [healthMsg, setHealthMsg] = useState("loading…");
  const [loadingStores, setLoadingStores] = useState(true);
  const [storesError, setStoresError] = useState("");

  // Fallback only if API fails/returns empty
  const fallbackStores = useMemo(
    () => [
      { id: "001", name: "Store 001" },
      { id: "002", name: "Store 002" },
      { id: "003", name: "Store 003" },
    ],
    []
  );

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
