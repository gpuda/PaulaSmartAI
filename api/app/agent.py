import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent


# =======================
# PATHS + ENV
# =======================
API_DIR = Path(__file__).resolve().parent.parent  # .../api
ROOT_DIR = API_DIR.parent

# U tvom projektu .env je u /api
load_dotenv(API_DIR / ".env")
load_dotenv(API_DIR / ".env.local", override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


SYSTEM_PROMPT = """
You are PaulaSmartAI, a helpful exercise & health assistant.

Rules:
- Default language: Croatian WITHOUT diacritics.
- Be concise, friendly, and safe.
- If the user message starts with "Web search:", you MUST call the web_search tool first.
- When you use web_search, include a short 'IZVORI:' section with 2 links from the tool output.
- If you are not sure, say so.
""".strip()


# =======================
# TOOLS
# =======================
_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


@tool
def web_search(query: str) -> str:
    """Search the public web and return concise summary with sources."""
    # DEBUG: potvrdi da li agent uopce poziva tool
    print(f"[TOOL] web_search called | query={query!r}", flush=True)

    if _tavily is None:
        out = "Tavily nije konfiguriran (nema TAVILY_API_KEY)."
        print(f"[TOOL] web_search returning {len(out)} chars", flush=True)
        return out

    # Uzmi vise rezultata pa onda filtriraj kvalitetu
    result = _tavily.search(
        query=query,
        max_results=10,
        include_answer=True,
        include_raw_content=False,
    )

    lines = []

    if result.get("answer"):
        lines.append(f"SAZETAK: {result['answer']}")

    # Filtriraj low-quality/nezgodne domene
    urls = []
    blocked = ("linkedin.com", "medium.com", "sourceforge.net")
    for r in result.get("results", [])[:10]:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        if any(b in u for b in blocked):
            continue
        urls.append(u)
        if len(urls) >= 5:
            break

    if urls:
        lines.append("IZVORI:")
        # Po pravilu iz prompta: u odgovoru 2 linka
        for u in urls[:2]:
            lines.append(f"- {u}")

    out = "\n".join(lines).strip()
    print(f"[TOOL] web_search returning {len(out)} chars", flush=True)
    return out


# =======================
# SINGLETON AGENT
# =======================
_agent = None


def get_agent():
    global _agent

    if _agent is None:
        llm = ChatOpenAI(model=MODEL)

        # Pouzdano "system" u ovoj kombinaciji verzija
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("messages"),
            ]
        )

        _agent = create_react_agent(
            llm,
            tools=[web_search],
            prompt=prompt,
        )

    return _agent