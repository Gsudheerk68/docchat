const BASE = import.meta.env.VITE_API_URL || "";

async function handle(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const listDocs = () => fetch(`${BASE}/api/documents`).then(handle);

export const uploadDoc = (file) => {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE}/api/documents`, { method: "POST", body: form }).then(handle);
};

export const deleteDoc = (id) =>
  fetch(`${BASE}/api/documents/${id}`, { method: "DELETE" }).then(handle);

/**
 * Stream a chat reply. Server sends Server-Sent Events: sources -> token* -> done.
 * Handlers: onSources(list), onToken(text). Pass an AbortSignal to support "Stop".
 */
export async function streamChat(message, history, useDocs, { onSources, onToken, signal }) {
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, use_docs: useDocs }),
    signal,
  });
  if (!res.ok || !res.body) await handle(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // keep any partial frame for the next read
    for (const frame of frames) {
      if (!frame.startsWith("data: ")) continue;
      const evt = JSON.parse(frame.slice(6));
      if (evt.type === "sources") onSources?.(evt.data);
      else if (evt.type === "token") onToken?.(evt.data);
      else if (evt.type === "error") throw new Error(evt.data);
    }
  }
}
