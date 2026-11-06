import { useRef, useState } from "react";
import { uploadDoc, deleteDoc } from "./api.js";

export default function Sidebar({ docs, setDocs, useDocs, setUseDocs }) {
  const fileRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function onFiles(e) {
    const files = [...e.target.files];
    e.target.value = "";
    setBusy(true);
    setError("");
    for (const f of files) {
      try {
        const info = await uploadDoc(f);
        setDocs((d) => [...d, info]);
      } catch (err) {
        setError(`${f.name}: ${err.message}`);
      }
    }
    setBusy(false);
  }

  async function onDelete(id) {
    try {
      await deleteDoc(id);
      setDocs((d) => d.filter((x) => x.id !== id));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <aside className="side">
      <h2>Documents</h2>
      <input ref={fileRef} type="file" accept=".pdf,.txt,.md" multiple hidden onChange={onFiles} />
      <button className="upload" onClick={() => fileRef.current.click()} disabled={busy}>
        {busy ? "Reading and indexing…" : "Add PDF, TXT or MD"}
      </button>
      {error && <p className="side-error" role="alert">{error}</p>}

      {docs.length === 0 ? (
        <p className="hint">No documents yet. Add one to chat with it.</p>
      ) : (
        <ul className="docs">
          {docs.map((d) => (
            <li key={d.id}>
              <span className="doc-name" title={d.name}>{d.name}</span>
              <span className="doc-meta">{d.chunks} chunks</span>
              <button className="x" aria-label={`Remove ${d.name}`} onClick={() => onDelete(d.id)}>Remove</button>
            </li>
          ))}
        </ul>
      )}

      <label className="toggle">
        <input type="checkbox" checked={useDocs} onChange={(e) => setUseDocs(e.target.checked)} />
        Answer from my documents
      </label>
    </aside>
  );
}
