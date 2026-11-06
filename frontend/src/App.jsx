import { useEffect, useRef, useState } from "react";
import { streamChat, listDocs } from "./api.js";
import Sidebar from "./Sidebar.jsx";

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      {sources.map((s) => (
        <details key={s.n}>
          <summary>
            [{s.n}] {s.source}{s.page ? `, p. ${s.page}` : ""}
          </summary>
          <p>{s.snippet}…</p>
        </details>
      ))}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]); // [{role, content, sources?}]
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [docs, setDocs] = useState([]);
  const [useDocs, setUseDocs] = useState(true);
  const endRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    listDocs().then(setDocs).catch(() => {});
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function onSend(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const history = messages.map(({ role, content }) => ({ role, content }));
    // add the user message and an empty assistant message that fills up as tokens arrive
    setMessages([...messages, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setInput("");
    setError("");
    setLoading(true);

    const patchLast = (fn) =>
      setMessages((m) => m.map((msg, i) => (i === m.length - 1 ? fn(msg) : msg)));

    abortRef.current = new AbortController();
    try {
      await streamChat(text, history, useDocs, {
        signal: abortRef.current.signal,
        onSources: (sources) => patchLast((msg) => ({ ...msg, sources })),
        onToken: (t) => patchLast((msg) => ({ ...msg, content: msg.content + t })),
      });
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="layout">
      <Sidebar docs={docs} setDocs={setDocs} useDocs={useDocs} setUseDocs={setUseDocs} />

      <div className="shell">
        <header className="top">
          <h1>DocuChat</h1>
          <button className="ghost" onClick={() => setMessages([])} disabled={!messages.length}>
            New chat
          </button>
        </header>

        <main className="thread">
          {messages.length === 0 && (
            <div className="empty">
              <p className="empty-title">Ask a question about your documents</p>
              <p>Add files on the left. Answers cite the passages they came from.</p>
            </div>
          )}
          {messages.map((m, i) => (m.role === "assistant" && !m.content && !m.sources) ? null : (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">
                {m.content}
                <Sources sources={m.sources} />
              </div>
            </div>
          ))}
          {loading && messages.at(-1)?.content === "" && (
            <div className="msg assistant">
              <div className="bubble typing" aria-label="Thinking"><span /><span /><span /></div>
            </div>
          )}
          {error && <div className="error" role="alert">{error}</div>}
          <div ref={endRef} />
        </main>

        <form className="composer" onSubmit={onSend}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={useDocs ? "Ask about your documents" : "Type a message"}
            aria-label="Message"
            autoFocus
          />
          {loading ? (
            <button type="button" className="stop" onClick={() => abortRef.current?.abort()}>Stop</button>
          ) : (
            <button type="submit" disabled={!input.trim()}>Send</button>
          )}
        </form>
      </div>
    </div>
  );
}
