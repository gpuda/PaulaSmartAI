"use client";

import { useState, useRef, useEffect, FormEvent } from "react";

type Role = "user" | "assistant";

type Message = {
  id: number;
  role: Role;
  content: string;
};

type Mode = "normal" | "rag" | "web";

const SUGGESTIONS = [
  "Daj mi ideju za trening gimnastike",
  "Pomogni mi oko zadace",
  "Ispricaj mi kratku zabavnu pricu",
];

function CatMascot() {
  return (
    <div className="relative h-20 w-20 floating-cat">
      <div className="pointer-events-none absolute inset-0 -z-10 rounded-full bg-cat-glow blur-2xl" />
      <svg
        viewBox="0 0 120 120"
        className="h-full w-full drop-shadow-lg"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="catBody" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f9c6ff" />
            <stop offset="100%" stopColor="#b1c5ff" />
          </linearGradient>
          <linearGradient id="catInnerEar" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ff9ac6" />
            <stop offset="100%" stopColor="#ffb6e6" />
          </linearGradient>
        </defs>

        {/* Ears */}
        <path
          d="M25 40 L40 10 L55 40 Z"
          fill="url(#catBody)"
          stroke="#9159c7"
          strokeWidth="2"
        />
        <path
          d="M65 40 L80 10 L95 40 Z"
          fill="url(#catBody)"
          stroke="#9159c7"
          strokeWidth="2"
        />
        <path
          d="M30 38 L40 18 L50 38 Z"
          fill="url(#catInnerEar)"
          opacity="0.9"
        />
        <path
          d="M70 38 L80 18 L90 38 Z"
          fill="url(#catInnerEar)"
          opacity="0.9"
        />

        {/* Head */}
        <rect
          x="25"
          y="30"
          width="70"
          height="60"
          rx="24"
          fill="url(#catBody)"
          stroke="#9159c7"
          strokeWidth="2"
        />

        {/* Cheeks */}
        <circle cx="43" cy="68" r="6" fill="#ff9ac6" opacity="0.7" />
        <circle cx="77" cy="68" r="6" fill="#ff9ac6" opacity="0.7" />

        {/* Eyes */}
        <circle cx="45" cy="58" r="5" fill="#2b2147" />
        <circle cx="75" cy="58" r="5" fill="#2b2147" />
        <circle cx="43.5" cy="56.5" r="1.6" fill="#ffffff" />
        <circle cx="73.5" cy="56.5" r="1.6" fill="#ffffff" />

        {/* Nose */}
        <path
          d="M57 66 L63 66 L60 70 Z"
          fill="#ff9ac6"
          stroke="#b463a9"
          strokeWidth="1"
        />

        {/* Smile */}
        <path
          d="M52 74 C56 78, 64 78, 68 74"
          fill="none"
          stroke="#2b2147"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* Whiskers */}
        <path
          d="M32 64 C40 64, 44 64, 50 64"
          stroke="#2b2147"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        <path
          d="M32 70 C40 69, 44 69, 50 69"
          stroke="#2b2147"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        <path
          d="M70 64 C76 64, 80 64, 88 64"
          stroke="#2b2147"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        <path
          d="M70 70 C76 69, 80 69, 88 69"
          stroke="#2b2147"
          strokeWidth="1.6"
          strokeLinecap="round"
        />

        {/* Collar */}
        <rect
          x="35"
          y="82"
          width="50"
          height="8"
          rx="4"
          fill="#9159c7"
          opacity="0.95"
        />
        <circle cx="60" cy="86" r="4" fill="#ffd86b" />
      </svg>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("normal");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(1);
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    setError(null);
    setInput("");
    setIsLoading(true);

    // Create user and empty assistant messages immediately
    const userMessageId = nextId.current++;
    const assistantMessageId = nextId.current++;

    const userMessage: Message = {
      id: userMessageId,
      role: "user",
      content: trimmed,
    };

    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    try {
      const history = messages
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      const payload: Record<string, unknown> = {
        message: trimmed,
        mode,
        history,
      };

      // Only for RAG we pass top_k (optional)
      if (mode === "rag") payload.top_k = 5;

      const res = await fetch("http://localhost:8000/chat_router", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      // WEB mode is non-stream JSON
      if (mode === "web") {
        const data = (await res.json()) as {
          aiReply?: string;
          tool_used?: boolean;
          tool_names?: string[];
        };

        const reply = (data.aiReply || "").trim();
        const final = reply
          ? reply
          : "Ups, ne mogu sada odgovoriti. Pokusaj opet malo kasnije.";

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId ? { ...m, content: final } : m
          )
        );

        return;
      }

      // NORMAL + RAG are SSE streams
      if (!res.body) {
        throw new Error("No response body for streaming.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let assistantContent = "";
      let doneStreaming = false;

      while (!doneStreaming) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const event of events) {
          const lines = event.split("\n");
          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line.startsWith("data:")) continue;

            const jsonPart = line.slice(5).trim();
            if (!jsonPart) continue;

            let evt: unknown;
            try {
              evt = JSON.parse(jsonPart);
            } catch {
              continue;
            }

            if (typeof evt !== "object" || evt === null) continue;
            if (!("type" in evt)) continue;

            const type = (evt as { type: string }).type;

            if (type === "delta" && "delta" in evt) {
              const delta = (evt as { delta?: string }).delta ?? "";
              if (typeof delta === "string" && delta.length > 0) {
                assistantContent += delta;
                const snapshot = assistantContent;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId ? { ...m, content: snapshot } : m
                  )
                );
              }
            } else if (type === "done") {
              doneStreaming = true;
              break;
            } else if (type === "error" && "message" in evt) {
              // optional: backend can emit error
              const msg = String((evt as { message?: string }).message ?? "");
              setError(msg || "Dogodila se greska u streamu.");
            }
          }

          if (doneStreaming) break;
        }
      }

      if (!assistantContent) {
        const fallback =
          "Ups, ne mogu sada odgovoriti. Pokusaj opet malo kasnije.";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId ? { ...m, content: fallback } : m
          )
        );
      }
    } catch (e) {
      console.error(e);
      setError(
        "Ups, nesto je poslo po zlu. Provjeri radi li backend i pokusaj ponovno."
      );
      const errorText =
        "Cini se da imam malu poteskocu sa spajanjem. Mozes probati ponovo za koju sekundu?";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId && !m.content
            ? { ...m, content: errorText }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage(input);
  };

  const handleSuggestionClick = (text: string) => {
    void sendMessage(text);
  };

  const handleClearChat = () => {
    const confirmed =
      typeof window !== "undefined"
        ? window.confirm("Jesi sigurna da zelis ocistiti chat?")
        : false;

    if (!confirmed) return;

    setMessages([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-radial font-sans text-slate-50">
      <div className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6 lg:px-8">
        <main className="relative w-full max-w-4xl">
          {/* Decorative blur circles */}
          <div className="pointer-events-none absolute -top-24 -left-10 h-48 w-48 rounded-full bg-pink-500/25 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-16 -right-4 h-56 w-56 rounded-full bg-sky-400/25 blur-3xl" />

          <section className="relative overflow-hidden rounded-3xl bg-white/5 p-5 shadow-[0_22px_70px_rgba(15,23,42,0.65)] backdrop-blur-2xl sm:p-8 lg:p-10 border border-white/10">
            <button
              type="button"
              onClick={handleClearChat}
              className="absolute right-4 top-4 rounded-full border border-slate-500/40 bg-slate-900/40 px-3 py-1 text-xs font-medium text-slate-200/80 shadow-sm transition-colors hover:border-pink-200/70 hover:bg-pink-200/10 hover:text-pink-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-pink-300 focus-visible:ring-offset-1 focus-visible:ring-offset-slate-950"
            >
              Ocisti
            </button>

            {/* Header */}
            <header className="mb-6 flex flex-col items-center justify-center gap-4 sm:mb-8">
              <div className="flex flex-col items-center justify-center gap-4">
                <CatMascot />
                <h1 className="bg-gradient-to-r from-violet-200 via-pink-200 to-fuchsia-200 bg-clip-text text-4xl font-semibold tracking-tight text-transparent sm:text-5xl">
                  <span>PaulaSmartAI</span>
                </h1>
              </div>

              {/* Mode toggle */}
              <div className="flex flex-wrap items-center justify-center gap-2">
                {(
                  [
                    { key: "normal", label: "Normal" },
                    { key: "rag", label: "RAG (PDF)" },
                    { key: "web", label: "Web" },
                  ] as const
                ).map((m) => {
                  const active = mode === m.key;
                  return (
                    <button
                      key={m.key}
                      type="button"
                      onClick={() => setMode(m.key)}
                      disabled={isLoading}
                      className={`rounded-full border px-3 py-1 text-xs font-semibold transition-all sm:text-sm ${
                        active
                          ? "border-pink-200/80 bg-pink-300/15 text-pink-50 shadow-sm shadow-pink-500/20"
                          : "border-white/10 bg-white/5 text-slate-200/80 hover:border-pink-200/50 hover:bg-pink-200/10"
                      }`}
                      title={
                        m.key === "normal"
                          ? "Obicni chat (stream)"
                          : m.key === "rag"
                          ? "Odgovara iz PDF prirucnika (stream + izvori)"
                          : "Pretrazuje web (tool)"
                      }
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </header>

            {/* Suggestions */}
            <div className="mb-5 flex flex-wrap gap-2 sm:mb-6">
              {SUGGESTIONS.map((label) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => handleSuggestionClick(label)}
                  className="group rounded-full border border-pink-200/40 bg-white/5 px-4 py-2 text-xs font-medium text-pink-100 shadow-sm transition-all hover:-translate-y-0.5 hover:border-pink-300/80 hover:bg-pink-300/15 hover:text-pink-50 hover:shadow-pink-500/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-pink-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 sm:text-sm"
                  disabled={isLoading}
                >
                  <span className="mr-1 inline-block text-pink-200/80 group-hover:text-pink-100">
                    ✨
                  </span>
                  {label}
                </button>
              ))}
            </div>

            {/* Chat area */}
            <div className="mb-4 flex max-h-[480px] flex-col gap-3 overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/40 p-4 sm:p-5">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center gap-4 py-10 text-center text-sm text-slate-300/80 sm:text-base">
                  <CatMascot />
                  <p className="max-w-md text-base font-medium text-slate-100 sm:text-lg">
                    Bok Paula! Spremna sam za razgovor 🐱
                  </p>
                </div>
              )}

              {messages.map((message) => {
                const isUser = message.role === "user";
                return (
                  <div
                    key={message.id}
                    className={`message-fade-in flex w-full ${
                      isUser ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-md transition-transform hover:-translate-y-0.5 sm:text-base ${
                        isUser
                          ? "bg-gradient-to-r from-pink-400 via-fuchsia-400 to-sky-400 text-white"
                          : "bg-slate-900/80 text-slate-50 border border-slate-700/70"
                      }`}
                    >
                      {!isUser && (
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-pink-200/80">
                            PaulaSmartAI
                          </div>
                          <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-300/70">
                            {mode === "normal"
                              ? "NORMAL"
                              : mode === "rag"
                              ? "RAG"
                              : "WEB"}
                          </div>
                        </div>
                      )}
                      <p className="whitespace-pre-line leading-relaxed">
                        {message.content}
                      </p>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex items-center gap-2 text-xs text-slate-200/80 sm:text-sm">
                  <div className="h-2 w-2 animate-ping rounded-full bg-pink-300" />
                  <span>Paula razmislja...</span>
                </div>
              )}

              <div ref={endOfMessagesRef} />
            </div>

            {error && (
              <p className="mb-3 text-xs font-medium text-pink-200 sm:text-sm">
                {error}
              </p>
            )}

            {/* Input */}
            <form
              onSubmit={handleSubmit}
              className="mt-1 flex items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/60 p-2 shadow-inner"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (input.trim()) {
                      void sendMessage(input);
                    }
                  }
                }}
                placeholder="Upisi svoje pitanje ili misao..."
                className="h-11 flex-1 rounded-xl border-none bg-transparent px-3 text-sm text-slate-50 placeholder:text-slate-400 focus:outline-none focus:ring-0 sm:h-12 sm:text-base"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="inline-flex h-11 items-center justify-center rounded-xl bg-gradient-to-r from-pink-400 via-fuchsia-400 to-sky-400 px-4 text-sm font-semibold text-white shadow-lg shadow-pink-500/30 transition-all hover:-translate-y-0.5 hover:shadow-pink-500/50 disabled:cursor-not-allowed disabled:opacity-50 sm:h-12 sm:px-5"
              >
                <span className="mr-1">Posalji</span>
                <span aria-hidden="true">➤</span>
              </button>
            </form>
          </section>
        </main>
      </div>
    </div>
  );
}
