"""Web ingestion with Playwright.

A category URL is expanded into candidate recipe links and traversed; a direct
recipe URL is read as-is. One progress frame is emitted per page fetched.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from ..config import settings
from ..jobs import Job
from ..models import IngestionRun, JobState, ProgressEvent, RawRecipe, SourceType, utcnow

RECIPE_HINT = re.compile(r"recipe|dish|cook|food|kitchen|meal", re.I)
ING_WORDS = re.compile(
    r"\b(cup|cups|tbsp|tsp|tablespoon|teaspoon|gram|grams|kg|ml|litre|ounce|oz|lb|"
    r"pinch|clove|cloves)\b", re.I)


class UnsafeUrl(ValueError):
    """Raised for URLs that resolve to private or loopback addresses."""


def assert_public_url(url: str) -> None:
    """Block non-HTTP schemes and hosts that resolve inside the network.

    Without this, a user-supplied URL can be used to probe internal services.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrl(f"only http/https allowed, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeUrl("no host in URL")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrl(f"cannot resolve {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise UnsafeUrl(f"{host} resolves to non-public address {ip}")


UA = "CognitiveKitchen/0.1"
_ROBOTS: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_allow(url: str) -> bool:
    """Honour robots.txt. A host we cannot ask is treated as allowed."""
    p = urlparse(url)
    key = f"{p.scheme}://{p.netloc}"
    if key not in _ROBOTS:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{key}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _ROBOTS[key] = rp
    rp = _ROBOTS[key]
    if rp is None:
        return True
    try:
        return rp.can_fetch(UA, url)
    except Exception:
        return True


def _text_lines(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    h1 = soup.find(["h1", "h2"])
    if h1:
        title = h1.get_text(strip=True) or title
    lines = [re.sub(r"\s+", " ", t.strip())
             for t in soup.get_text("\n").splitlines() if t.strip()]
    return title[:180], lines


def _jsonld_recipes(html: str) -> list[dict]:
    """Recipe schema.org blocks, the most reliable signal when present."""
    soup = BeautifulSoup(html, "lxml")
    found = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(tag.string or "")
        except Exception:
            continue
        stack = [blob]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                t = node.get("@type")
                types = t if isinstance(t, list) else [t]
                if any(str(x).lower() == "recipe" for x in types if x):
                    found.append(node)
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return found


def _as_lines(value) -> list[str]:
    """schema.org fields arrive as a string, a list, or nested dicts.

    Iterating a string yields characters, which previously turned one field into
    123 single-letter ingredients.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\r\n]+|\s{2,}", value)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, dict):
        return _as_lines(value.get("text") or value.get("name")
                         or value.get("itemListElement"))
    out: list[str] = []
    for item in value:
        out.extend(_as_lines(item))
    return out


def _from_jsonld(node: dict, url: str, idx: int) -> RawRecipe:
    ings = _as_lines(node.get("recipeIngredient") or node.get("ingredients"))
    steps = _as_lines(node.get("recipeInstructions"))
    return RawRecipe(
        recipe_id=f"r_{idx:04d}",
        title=str(node.get("name") or "")[:180] or url,
        source_type=SourceType.url, origin=url, url=url,
        ingredient_lines=[i for i in ings if i.strip()],
        step_lines=[s for s in steps if s.strip()],
        servings_hint=str(node.get("recipeYield")) if node.get("recipeYield") else None,
        time_hint=str(node.get("totalTime")) if node.get("totalTime") else None,
        detected_by="json-ld",
        raw_text="\n".join(ings + steps),
    )


def _from_text(title: str, lines: list[str], url: str, idx: int) -> RawRecipe | None:
    ing = [l for l in lines if ING_WORDS.search(l) and len(l) < 160]
    steps = [l for l in lines if len(l) > 60]
    # A page with no ingredient lines is not a recipe, whatever else it contains.
    # Accepting them turned 404 pages into recipes and suppressed crawling.
    if len(ing) < 3 or len(steps) < 2:
        return None
    return RawRecipe(
        recipe_id=f"r_{idx:04d}", title=title or url,
        source_type=SourceType.url, origin=url, url=url,
        ingredient_lines=ing[:80], step_lines=steps[:80],
        detected_by="heuristic-text", raw_text="\n".join(lines)[:20000],
    )


# path segments that mean "index of recipes", not a recipe
# A path containing any of these is a listing, wherever the segment appears.
# Checking only the segment after /recipes/ let /category/recipes/<slug> score as
# highly as a real dish, so a whole crawl budget went on category pages.
INDEX_SEGMENTS = {"collection", "collections", "category", "categories", "cuisine",
                  "cuisines", "course", "courses", "tag", "tags", "topic", "topics",
                  "howto", "how-to", "occasion", "ingredient", "ingredients",
                  "search", "page", "archive", "archives", "index", "browse",
                  "all", "list", "type", "types"}
# paths that never hold a recipe
JUNK = re.compile(
    r"/(subscribe|newsletters?|news-trends|news|feature|features|author|authors|"
    r"competition|competitions|login|register|account|app|gift|shop|store|video|"
    r"videos|advertise|about|contact|privacy|terms|cookies|sitemap|reviews?|"
    r"health|diet-plans?|magazine|binge|podcast)(/|$)", re.I)


# A slug naming the plural is a listing: "south-indian-recipes", "sweets-recipes".
# The singular is a dish: "skillet-sicilian-chicken-recipe".
PLURAL_INDEX = re.compile(r"(^|-)recipes(-|$)|(^|-)dishes(-|$)|(^|-)ideas(-|$)", re.I)
PAGINATION = re.compile(r"/page/\d+|[?&]page=\d+", re.I)


def _link_score(url: str, base: str) -> int:
    """Higher is more likely to be an individual dish page."""
    p = urlparse(url)
    path = p.path.rstrip("/")
    if not path or JUNK.search(path):
        return 0
    if PAGINATION.search(url):
        return 1                                       # more of the same listing
    parts = [s.lower() for s in path.split("/") if s]
    if not parts:
        return 0
    if any(s in INDEX_SEGMENTS for s in parts):
        return 2                                       # a listing, not a dish
    last = parts[-1]
    if PLURAL_INDEX.search(last):
        return 2                                       # "south-indian-recipes"
    if "recipe" in parts or "recipes" in parts:
        i = parts.index("recipe") if "recipe" in parts else parts.index("recipes")
        if not parts[i + 1:]:
            return 1                                   # the /recipes index itself
        return 10 if "-" in last else 6
    # Dishes published outside a /recipe/ prefix, e.g. /2015/03/dish-name
    if "-" in last and len(last) > 8:
        return 7
    return 0


def _discover_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    host = urlparse(base).netloc
    scored: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"].split("#")[0])
        p = urlparse(href)
        if p.scheme not in ("http", "https") or p.netloc != host or href == base:
            continue
        score = _link_score(href, base)
        if score <= 0:
            continue
        text = a.get_text(" ", strip=True)
        if RECIPE_HINT.search(text):
            score += 1
        scored[href] = max(scored.get(href, 0), score)
    # richest candidates first so a small page budget is spent on real recipes
    return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])]


def ingest_urls(job: Job, urls: list[str], crawl_category: bool = True,
                max_pages: int | None = None) -> IngestionRun:
    limit = max_pages or settings.max_pages_per_run
    t0 = time.perf_counter()
    origin = urls[0] if urls else ""
    run = IngestionRun(run_id=job.job_id, source_type=SourceType.url, origin=origin)

    job.emit(ProgressEvent(job_id=job.job_id, kind="started", state=JobState.running,
                           message=f"Launching browser for {len(urls)} seed URL(s)",
                           unit_label=origin))

    safe_seeds: list[str] = []
    for u in urls:
        try:
            assert_public_url(u)
            safe_seeds.append(u)
        except UnsafeUrl as exc:
            run.warnings.append(f"rejected {u}: {exc}")
            job.emit(ProgressEvent(job_id=job.job_id, kind="warning",
                                   state=JobState.running,
                                   message=f"Rejected {u}: {exc}", unit_label=u))
    if not safe_seeds:
        raise UnsafeUrl("no usable URLs after safety checks")

    queue = list(safe_seeds)
    visited: set[str] = set()
    seed_ref = safe_seeds[0]          # sent as Referer, as a browser would

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-dev-shm-usage"])
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/151.0 Safari/537.36 CognitiveKitchen/0.1"),
            viewport={"width": 1400, "height": 900})
        # Only the HTML matters here. Images, fonts, media and stylesheets are the
        # bulk of load time on a recipe page and none of them are ever parsed.
        ctx.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in
            ("image", "media", "font", "stylesheet") else route.continue_()))
        page = ctx.new_page()

        while queue and len(visited) < limit:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            run.n_units_total = max(run.n_units_total, len(visited) + len(queue))

            if not robots_allow(url):
                run.warnings.append(f"{url}: disallowed by robots.txt")
                job.emit(ProgressEvent(
                    job_id=job.job_id, kind="warning", state=JobState.running,
                    message=f"Skipped (robots.txt): {url}",
                    unit_index=len(visited), unit_total=run.n_units_total,
                    unit_label=url, units_read=run.n_units_read,
                    recipes_found=len(run.recipes),
                    elapsed_seconds=round(time.perf_counter() - t0, 3)))
                continue

            # Pace before the request, not after a successful load. Applying it
            # afterwards meant a run of failures fired ten requests in 0.7s.
            if len(visited) > 1:
                page.wait_for_timeout(settings.politeness_delay_ms)

            try:
                nav = page.goto(url, timeout=settings.request_timeout_ms,
                                wait_until="domcontentloaded",
                                referer=seed_ref)
                status = nav.status if nav is not None else 0
                if status >= 400:
                    run.warnings.append(f"{url}: HTTP {status}")
                    job.emit(ProgressEvent(
                        job_id=job.job_id, kind="warning", state=JobState.running,
                        message=f"HTTP {status} for {url}",
                        unit_index=len(visited), unit_total=run.n_units_total,
                        unit_label=url, units_read=run.n_units_read,
                        recipes_found=len(run.recipes),
                        elapsed_seconds=round(time.perf_counter() - t0, 3)))
                    continue
                page.wait_for_timeout(settings.politeness_delay_ms)
                html = page.content()
            except Exception as exc:
                run.warnings.append(f"{url}: {type(exc).__name__}")
                job.emit(ProgressEvent(
                    job_id=job.job_id, kind="warning", state=JobState.running,
                    message=f"Failed {url}: {type(exc).__name__}",
                    unit_index=len(visited), unit_total=run.n_units_total,
                    unit_label=url, units_read=run.n_units_read,
                    recipes_found=len(run.recipes),
                    elapsed_seconds=round(time.perf_counter() - t0, 3)))
                continue

            run.n_units_read += 1
            title, lines = _text_lines(html)

            before = len(run.recipes)
            nodes = _jsonld_recipes(html)
            accepted = 0
            for node in nodes:
                cand = _from_jsonld(node, url, len(run.recipes) + 1)
                # Collection pages carry a Recipe stub with no usable ingredient
                # list. Treating it as a recipe both produced junk and stopped
                # the crawl from reaching the real recipes.
                if len(cand.ingredient_lines) >= 2 and cand.title:
                    run.recipes.append(cand)
                    accepted += 1
            if not accepted:
                rec = _from_text(title, lines, url, len(run.recipes) + 1)
                if rec is not None:
                    run.recipes.append(rec)

            for rec in run.recipes[before:]:
                job.emit(ProgressEvent(
                    job_id=job.job_id, kind="recipe", state=JobState.running,
                    message=f"Recipe: {rec.title[:60]}",
                    unit_index=len(visited), unit_total=run.n_units_total,
                    unit_label=url, units_read=run.n_units_read,
                    recipes_found=len(run.recipes),
                    elapsed_seconds=round(time.perf_counter() - t0, 3),
                    payload={"title": rec.title, "url": url,
                             "detected_by": rec.detected_by}))

            # An index page carries no schema.org Recipe block, so absence of one
            # is the signal to look for links rather than absence of any match.
            if crawl_category and not accepted:
                for link in _discover_links(html, url):
                    if link not in visited and link not in queue:
                        queue.append(link)
                run.n_units_total = max(run.n_units_total, len(visited) + len(queue))

            job.emit(ProgressEvent(
                job_id=job.job_id, kind="unit", state=JobState.running,
                message=f"Read {url}",
                unit_index=len(visited), unit_total=min(run.n_units_total, limit),
                unit_label=url, units_read=run.n_units_read,
                recipes_found=len(run.recipes), chars_read=len(html),
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                payload={"queued": len(queue), "title": title}))

        page.close()
        ctx.close()
        browser.close()

    run.n_recipes = len(run.recipes)
    run.elapsed_seconds = round(time.perf_counter() - t0, 3)
    run.finished_at = utcnow()
    return run