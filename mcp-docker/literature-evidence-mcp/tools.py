import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from urllib.parse import quote

import httpx

from core import llm_tool


HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
BING_API_KEY = os.getenv("BING_API_KEY", "")
BING_ENDPOINT = os.getenv("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "MatBrain-MCP/0.1 (materials evidence search; contact: local)",
)


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


async def _get_json(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    merged_headers = {"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"}
    if headers:
        merged_headers.update(headers)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=True) as client:
        response = await client.get(url, params=params, headers=merged_headers)
        response.raise_for_status()
        return response.json()


async def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
    merged_headers = {"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"}
    if headers:
        merged_headers.update(headers)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=True) as client:
        response = await client.post(url, json=payload, headers=merged_headers)
        response.raise_for_status()
        return response.json()


def _error_text(exc: Exception) -> str:
    return str(exc) or repr(exc)


def _clean_text(text: str, max_len: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:max_len]


@llm_tool(
    name="bing_search",
    description="Search the web with Bing Web Search API when BING_API_KEY is configured; evidence-only tool.",
)
async def bing_search(query: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "bing", "results": [], "errors": []}
    if not BING_API_KEY:
        payload["ok"] = False
        payload["errors"].append("BING_API_KEY is not configured")
        return _json_text(payload)
    try:
        data = await _get_json(
            BING_ENDPOINT,
            params={"q": query, "count": max(1, int(top_k)), "responseFilter": "Webpages"},
            headers={"Ocp-Apim-Subscription-Key": BING_API_KEY},
        )
        payload["results"] = [
            {
                "title": item.get("name"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "date_last_crawled": item.get("dateLastCrawled"),
            }
            for item in data.get("webPages", {}).get("value", [])[: max(1, int(top_k))]
        ]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="tavily_search",
    description="Search web/literature context with Tavily when TAVILY_API_KEY is configured; evidence-only tool.",
)
async def tavily_search(query: str, top_k: int = 5, search_depth: str = "basic") -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "tavily", "results": [], "errors": []}
    if not TAVILY_API_KEY:
        payload["ok"] = False
        payload["errors"].append("TAVILY_API_KEY is not configured")
        return _json_text(payload)
    try:
        data = await _post_json(
            "https://api.tavily.com/search",
            {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max(1, int(top_k)),
                "search_depth": search_depth,
                "include_answer": False,
            },
        )
        payload["results"] = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": _clean_text(item.get("content")),
                "score": item.get("score"),
            }
            for item in data.get("results", [])[: max(1, int(top_k))]
        ]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="openalex_search",
    description="Search OpenAlex works for literature evidence; no API key required.",
)
async def openalex_search(query: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "openalex", "results": [], "errors": []}
    try:
        data = await _get_json("https://api.openalex.org/works", params={"search": query, "per-page": max(1, int(top_k))})
        payload["results"] = [
            {
                "title": item.get("title"),
                "doi": item.get("doi"),
                "publication_year": item.get("publication_year"),
                "cited_by_count": item.get("cited_by_count"),
                "url": item.get("id"),
                "open_access": item.get("open_access"),
            }
            for item in data.get("results", [])[: max(1, int(top_k))]
        ]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="crossref_search",
    description="Search Crossref works for DOI and publication metadata; no API key required.",
)
async def crossref_search(query: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "crossref", "results": [], "errors": []}
    try:
        data = await _get_json("https://api.crossref.org/works", params={"query": query, "rows": max(1, int(top_k))})
        rows = []
        for item in data.get("message", {}).get("items", [])[: max(1, int(top_k))]:
            title = item.get("title") or [""]
            rows.append({
                "title": title[0] if title else "",
                "doi": item.get("DOI"),
                "published": item.get("published-print") or item.get("published-online") or item.get("issued"),
                "publisher": item.get("publisher"),
                "container_title": (item.get("container-title") or [""])[0],
                "url": item.get("URL"),
                "is_referenced_by_count": item.get("is-referenced-by-count"),
            })
        payload["results"] = rows
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="semantic_scholar_search",
    description="Search Semantic Scholar papers for citations, abstracts, and paper IDs; API key optional.",
)
async def semantic_scholar_search(query: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "semantic_scholar", "results": [], "errors": []}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    try:
        data = await _get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": max(1, int(top_k)),
                "fields": "title,year,authors,abstract,citationCount,url,externalIds,venue",
            },
            headers=headers,
        )
        payload["results"] = [
            {
                "title": item.get("title"),
                "year": item.get("year"),
                "venue": item.get("venue"),
                "citation_count": item.get("citationCount"),
                "url": item.get("url"),
                "doi": (item.get("externalIds") or {}).get("DOI"),
                "abstract": _clean_text(item.get("abstract")),
                "authors": [a.get("name") for a in item.get("authors", [])[:8]],
            }
            for item in data.get("data", [])[: max(1, int(top_k))]
        ]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="arxiv_search",
    description="Search arXiv papers and return title, authors, abstract, and links; no API key required.",
)
async def arxiv_search(query: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "arxiv", "results": [], "errors": []}
    try:
        url = f"https://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results={max(1, int(top_k))}"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=True) as client:
            response = await client.get(url, headers={"User-Agent": HTTP_USER_AGENT})
            response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        rows = []
        for entry in root.findall("atom:entry", ns):
            rows.append({
                "title": _clean_text(entry.findtext("atom:title", default="", namespaces=ns), 300),
                "summary": _clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "updated": entry.findtext("atom:updated", default="", namespaces=ns),
                "url": entry.findtext("atom:id", default="", namespaces=ns),
                "authors": [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)[:8]],
            })
        payload["results"] = rows
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="pubchem_search",
    description="Search PubChem compounds and return CIDs plus basic names; no API key required.",
)
async def pubchem_search(query: str, top_k: int = 10) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": query, "source": "pubchem", "results": [], "errors": []}
    try:
        data = await _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(query)}/cids/JSON")
        cids = data.get("IdentifierList", {}).get("CID", [])[: max(1, int(top_k))]
        rows = []
        for cid in cids:
            try:
                props = await _get_json(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
                )
                item = props.get("PropertyTable", {}).get("Properties", [{}])[0]
                rows.append(item)
            except Exception:
                rows.append({"CID": cid})
        payload["results"] = rows
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(_error_text(exc))
    return _json_text(payload)


@llm_tool(
    name="evidence_search_all",
    description="Query stable evidence sources OpenAlex and Crossref, plus optional Tavily/Bing and unstable arXiv/Semantic Scholar on request.",
)
async def evidence_search_all(
    query: str,
    top_k_per_source: int = 3,
    include_web: bool = False,
    include_unstable_sources: bool = False,
) -> str:
    sources = {
        "openalex": json.loads(await openalex_search(query, top_k_per_source)),
        "crossref": json.loads(await crossref_search(query, top_k_per_source)),
    }
    if include_unstable_sources:
        sources["semantic_scholar"] = json.loads(await semantic_scholar_search(query, top_k_per_source))
        sources["arxiv"] = json.loads(await arxiv_search(query, top_k_per_source))
    if include_web:
        sources["tavily"] = json.loads(await tavily_search(query, top_k_per_source))
        sources["bing"] = json.loads(await bing_search(query, top_k_per_source))
    failed_sources = [name for name, data in sources.items() if not data.get("ok")]
    payload = {
        "ok": len(failed_sources) < len(sources),
        "query": query,
        "sources": sources,
        "failed_sources": failed_sources,
    }
    return _json_text(payload)
