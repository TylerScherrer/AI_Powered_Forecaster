import { useEffect, useMemo, useState } from "react";

function buildApiBase() {
  const raw = (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");
  if (!raw) return "";                        // relative mode -> /api/*
  return raw.endsWith("/api") ? raw : `${raw}/api`;
}

const API_BASE = buildApiBase();

async function getJSON(path) {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export default function App() {
  const [stores, setStores] = useState([]);
  const [selectedStore, setSelectedStore] = useState("");
  const [hello, setHello] = useState("");
  const [healthMsg, setHealthMsg] = useState("loading…");
  const [loadingStores, setLoadingStores] = useState(true);
  const [storesError, setStoresError] = useState("");

  // Fallback used ONLY if API returns nothing
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
          // API returned empty -> use fallback ONCE
          setStores(fallbackStores);
          setSelectedStore(fallbackStores[0].id);
        }
      })
      .catch((e) => {
        // API failed -> use fallback
        setStoresError(e.message);
        setStores(fallbackStores);
        setSelectedStore(fallbackStores[0].id);
      })
      .finally(() => setLoadingStores(false));
  }, [fallbackStores]);

  const callHello = async () => {
    try {
      const q = new URLSearchParams({ store: selectedStore }).toString();
      const data = await getJSON(`/hello?${q}`);
      setHello(data?.message ?? JSON.stringify(data));
    } catch (e) {
      setHello(`error: ${e.message}`);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", margin: "2rem auto", maxWidth: 720 }}>
      <h1>Frontend ready</h1>
      <p>
        SWA → Azure API test:
      </p>

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
        <button onClick={callHello} disabled={!selectedStore}>
          Test /hello
        </button>
      </div>

      {storesError && (
        <p style={{ color: "#b00", marginTop: 8 }}>
          (stores API error: {storesError} — showing fallback)
        </p>
      )}

      <pre style={{ marginTop: 16 }}>{hello}</pre>

      <div style={{ opacity: 0.6, marginTop: 24, fontSize: 12 }}>
        API base: <code>{API_BASE || "(relative /api)"}</code>
      </div>
    </div>
  );
}
