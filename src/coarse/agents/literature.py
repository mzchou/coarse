"""Literature search agent — finds related papers to ground the review.

Primary path: Perplexity Sonar Pro Search via OpenRouter (web-grounded, ~12s, ~$0.03).
Fallback path: OpenAlex API + 2 LLM calls (free, no key, indexes 240M+ scholarly
works across journals, working papers, preprints, and books — including econ
sources arXiv misses, such as NBER, SSRN, RePEc, and major journal publishers).

Set OPENALEX_EMAIL in the environment to route through OpenAlex's polite pool.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests
from pydantic import BaseModel, Field

from coarse.config import has_provider_key
from coarse.llm import LLMClient
from coarse.models import LITERATURE_SEARCH_MODEL
from coarse.prompts import (
    LITERATURE_QUERY_GEN_SYSTEM,
    LITERATURE_RANKING_SYSTEM,
    PERPLEXITY_SYSTEM,
    perplexity_user,
)

logger = logging.getLogger(__name__)

_OPENALEX_API = "https://api.openalex.org/works"
_MAX_RESULTS_PER_QUERY = 15
_MAX_ITERATIONS = 2
_TOP_K = 15
_PERPLEXITY_TEMPERATURE = 0.3
_QUERY_GEN_TEMPERATURE = 0.5
_RANKING_TEMPERATURE = 0.2
_OPENALEX_FIELDS = (
    "id,title,display_name,publication_year,authorships,"
    "abstract_inverted_index,primary_location,cited_by_count,doi"
)


@dataclass
class OpenAlexWork:
    """Minimal representation of an OpenAlex /works search result."""

    work_id: str
    title: str
    authors: list[str]
    abstract: str
    year: str
    venue: str
    cited_by_count: int = 0
    doi: str = ""


class _SearchQueries(BaseModel):
    """LLM-generated search queries for OpenAlex."""

    queries: list[str] = Field(min_length=1, max_length=5)


class _RankedResult(BaseModel):
    """A single ranked search result."""

    work_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    reason: str


class _RankedResults(BaseModel):
    """LLM-ranked search results with optional refinement queries."""

    ranked: list[_RankedResult]
    refinement_queries: list[str] = Field(default_factory=list, max_length=3)


def _search_perplexity(title: str, abstract: str, client: LLMClient) -> str:
    """Primary path: web-grounded literature search via Perplexity Sonar Pro."""
    perplexity_client = LLMClient(model=LITERATURE_SEARCH_MODEL)
    messages = [
        {"role": "system", "content": PERPLEXITY_SYSTEM},
        {"role": "user", "content": perplexity_user(title, abstract[:1500])},
    ]
    content = perplexity_client.complete_text(
        messages,
        max_tokens=4096,
        temperature=_PERPLEXITY_TEMPERATURE,
        timeout=60,
    )
    client.add_cost(perplexity_client.cost_usd)
    return content


def _openalex_user_agent() -> str:
    email = os.environ.get("OPENALEX_EMAIL", "").strip()
    if email:
        return f"coarse-ink (mailto:{email})"
    return "coarse-ink (https://github.com/Davidvandijcke/coarse)"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex returns abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idx_list in inverted_index.items():
        for pos in idx_list:
            positions[pos] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def _extract_venue(work: dict) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or ""
    if venue:
        return venue
    host = work.get("host_venue") or {}
    return host.get("display_name") or ""


def _parse_openalex_work(work: dict) -> OpenAlexWork | None:
    raw_id = work.get("id") or ""
    work_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
    title = (work.get("title") or work.get("display_name") or "").strip()
    if not title:
        return None
    authors: list[str] = []
    for au in work.get("authorships", []) or []:
        name = (au.get("author") or {}).get("display_name") or ""
        if name:
            authors.append(name.strip())
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    year = work.get("publication_year")
    doi_raw = work.get("doi") or ""
    return OpenAlexWork(
        work_id=work_id,
        title=" ".join(title.split()),
        authors=authors,
        abstract=abstract[:1200],
        year=str(year) if year else "",
        venue=_extract_venue(work),
        cited_by_count=int(work.get("cited_by_count") or 0),
        doi=doi_raw.replace("https://doi.org/", "") if doi_raw else "",
    )


def _search_openalex(
    query: str, max_results: int = _MAX_RESULTS_PER_QUERY
) -> list[OpenAlexWork]:
    params = {
        "search": query,
        "per-page": max_results,
        "select": _OPENALEX_FIELDS,
    }
    try:
        resp = requests.get(
            _OPENALEX_API,
            params=params,
            headers={"User-Agent": _openalex_user_agent()},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        logger.warning("OpenAlex API request failed for query: %s", query)
        return []

    works: list[OpenAlexWork] = []
    for raw in payload.get("results") or []:
        parsed = _parse_openalex_work(raw)
        if parsed:
            works.append(parsed)
    return works


def _search_openalex_pipeline(
    title: str,
    abstract: str,
    client: LLMClient,
) -> str:
    """Run the OpenAlex literature search (fallback path)."""
    queries = _generate_queries(title, abstract, client)
    if not queries:
        return ""

    all_works: dict[str, OpenAlexWork] = {}
    ranked: list[_RankedResult] = []

    for iteration in range(_MAX_ITERATIONS):
        for query in queries:
            for w in _search_openalex(query):
                if w.work_id and w.work_id not in all_works:
                    all_works[w.work_id] = w

        if not all_works:
            break

        ranked, refinement_queries = _rank_results(
            title, abstract, list(all_works.values()), client
        )

        if iteration < _MAX_ITERATIONS - 1 and refinement_queries:
            queries = refinement_queries
        else:
            break

    if not all_works:
        logger.info("Literature search found no results")
        return ""

    return _compile_context(ranked[:_TOP_K], all_works)


def search_literature(
    title: str,
    abstract: str,
    client: LLMClient,
) -> str:
    """Run the literature search. Signature unchanged for pipeline.py.

    Uses Perplexity Sonar Pro Search if OPENROUTER_API_KEY is set,
    falling back to the OpenAlex pipeline on failure or missing key.
    """
    if has_provider_key("openrouter"):
        try:
            result = _search_perplexity(title, abstract, client)
            logger.info("Literature search completed via Perplexity")
            return result
        except Exception:
            logger.warning(
                "Perplexity search failed, falling back to OpenAlex", exc_info=True
            )

    return _search_openalex_pipeline(title, abstract, client)


def _generate_queries(title: str, abstract: str, client: LLMClient) -> list[str]:
    messages = [
        {"role": "system", "content": LITERATURE_QUERY_GEN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Generate search queries for finding related work to this paper.\n\n"
                f"**Title**: {title}\n\n"
                f"**Abstract**: {abstract[:1000]}"
            ),
        },
    ]
    try:
        result = client.complete(
            messages,
            _SearchQueries,
            max_tokens=512,
            temperature=_QUERY_GEN_TEMPERATURE,
        )
        return result.queries
    except Exception:
        logger.warning("Query generation failed, using title as fallback")
        return [title]


def _rank_results(
    title: str,
    abstract: str,
    works: list[OpenAlexWork],
    client: LLMClient,
) -> tuple[list[_RankedResult], list[str]]:
    results_block = "\n\n".join(
        _format_work_for_ranking(w) for w in works[:20]
    )

    messages = [
        {"role": "system", "content": LITERATURE_RANKING_SYSTEM},
        {
            "role": "user",
            "content": (
                f"**Target paper**: {title}\n"
                f"**Abstract**: {abstract[:500]}\n\n"
                f"**Search results**:\n{results_block}\n\n"
                f"Rank these results by relevance and suggest refinement queries "
                f"if important related work areas are missing."
            ),
        },
    ]
    try:
        result = client.complete(
            messages,
            _RankedResults,
            max_tokens=2048,
            temperature=_RANKING_TEMPERATURE,
        )
        ranked = sorted(result.ranked, key=lambda r: r.relevance_score, reverse=True)
        return ranked, result.refinement_queries
    except Exception:
        logger.warning("Ranking failed, returning unranked results")
        unranked = [
            _RankedResult(work_id=w.work_id, relevance_score=0.5, reason="unranked")
            for w in works[:_TOP_K]
        ]
        return unranked, []


def _format_work_for_ranking(w: OpenAlexWork) -> str:
    authors = ", ".join(w.authors[:3])
    meta_bits = [b for b in (w.venue, w.year) if b]
    if w.cited_by_count:
        meta_bits.append(f"{w.cited_by_count} cites")
    meta = " · ".join(meta_bits) if meta_bits else "venue/year unknown"
    return (
        f"- **{w.work_id}**: {w.title}\n"
        f"  Authors: {authors}\n"
        f"  Venue: {meta}\n"
        f"  Abstract: {w.abstract[:500]}"
    )


def _compile_context(
    ranked: list[_RankedResult], works: dict[str, OpenAlexWork]
) -> str:
    lines: list[str] = []
    for i, r in enumerate(ranked, 1):
        w = works.get(r.work_id)
        if not w:
            continue
        authors_str = ", ".join(w.authors[:3])
        if len(w.authors) > 3:
            authors_str += " et al."
        meta_bits = [b for b in (w.venue, w.year) if b]
        if w.cited_by_count:
            meta_bits.append(f"{w.cited_by_count} citations")
        meta_str = " · ".join(meta_bits) if meta_bits else "venue/year unknown"
        ref = f"DOI:{w.doi}" if w.doi else f"OpenAlex:{w.work_id}"
        lines.append(
            f"{i}. **{w.title}** ({authors_str})\n"
            f"   {meta_str}\n"
            f"   {ref} — {r.reason}"
        )

    if not lines:
        return ""
    return "\n".join(lines)
