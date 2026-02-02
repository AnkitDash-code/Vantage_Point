from __future__ import annotations

from functools import lru_cache
import asyncio
import json
import logging
from pathlib import Path
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.analyzer import ScoutingAnalyzer, _convert_to_serializable
from app.env import load_env
from app.grid_client import GridAPIError, GridClient, AsyncGridClient
from app.models import ScoutReport, ScoutRequest
from app.rag_engine import RAGEngine
from app.settings import Settings

load_env()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Vantage Point API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings.from_env()

PRECOMPUTED_DIR = settings.precomputed_dir or (
    Path(__file__).resolve().parents[2] / "frontend" / "public" / "precomputed"
)


def _slugify_team(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower()).strip("_")


@lru_cache
def _load_precomputed_manifest() -> dict | None:
    manifest_path = PRECOMPUTED_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_precomputed_slug(team_name: str) -> str:
    manifest = _load_precomputed_manifest()
    if manifest:
        for team in manifest.get("teams", []):
            if team.get("name", "").strip().lower() == team_name.strip().lower():
                slug = team.get("slug")
                if slug:
                    return slug
    return _slugify_team(team_name)


def _load_precomputed_report(team_name: str, match_limit: int | None = None) -> dict | None:
    slug = _get_precomputed_slug(team_name)
    report_path = PRECOMPUTED_DIR / "teams" / f"{slug}.json"
    if not report_path.exists():
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if match_limit is not None and data.get("match_limit"):
            if int(data.get("match_limit")) != int(match_limit):
                return None
        return data
    except Exception:
        return None


def _normalize_precomputed_report(data: dict) -> ScoutReport:
    return ScoutReport(
        team_name=data.get("team_name", ""),
        matches_analyzed=int(data.get("matches_analyzed", 0) or 0),
        metrics=data.get("metrics") or {},
        insights=data.get("insights") or {},
    )

# Max parallel series event fetches (tuned via test runs)
EVENT_FETCH_CONCURRENCY = 8

grid_client = GridClient(
    api_key=settings.grid_api_key,
    debug_mode=settings.debug_mode,
    force_live=settings.grid_force_live,
    api_mode=settings.grid_api_mode,
    fallback_to_rest=settings.grid_fallback_to_rest,
    graphql_url=settings.grid_graphql_url,
    central_data_url=settings.grid_central_data_url,
    file_download_url=settings.grid_file_download_url,
    end_state_path=settings.grid_end_state_path,
    file_download_list_path=settings.grid_file_download_list_path,
    file_download_events_path=settings.grid_file_download_events_path,
    series_page_size=settings.grid_series_page_size,
    series_max_pages=settings.grid_series_max_pages,
    include_player_team=settings.grid_include_player_team,
    graphql_query_path=settings.grid_graphql_query_path,
)

async_grid_client = AsyncGridClient(
    api_key=settings.grid_api_key,
    debug_mode=settings.debug_mode,
    force_live=settings.grid_force_live,
    api_mode=settings.grid_api_mode,
    fallback_to_rest=settings.grid_fallback_to_rest,
    central_data_url=settings.grid_central_data_url,
    file_download_url=settings.grid_file_download_url,
    series_page_size=settings.grid_series_page_size,
    series_max_pages=settings.grid_series_max_pages,
    include_player_team=settings.grid_include_player_team,
)


@lru_cache
def get_rag_engine() -> RAGEngine:
    rag_urls = settings.rag_urls or [
        "https://www.valorantzone.gg/guides/agent-counters",
        "https://www.thespike.gg/tactics",
    ]
    return RAGEngine(
        use_web=settings.rag_use_web,
        urls=rag_urls,
    )


def _build_report(request: ScoutRequest) -> ScoutReport:
    total_start = time.perf_counter()
    
    # Step 1: Fetch matches
    t0 = time.perf_counter()
    matches = grid_client.fetch_team_matches(
        request.team_name,
        limit=request.match_limit,
        game_title=request.game_title,
    )
    logger.info(f"[TIMING] fetch_matches: {time.perf_counter() - t0:.2f}s ({len(matches)} matches)")

    if request.map_filter:
        map_filter = request.map_filter.strip().lower()
        matches = [
            match
            for match in matches
            if (match.get("map") or {}).get("name", "").strip().lower()
            == map_filter
        ]

    if not matches:
        raise HTTPException(status_code=404, detail="No matches found for team.")

    events_by_series = {}
    cached_metrics = None
    
    # Step 2: Check cache
    t0 = time.perf_counter()
    if not settings.grid_force_live:
        metrics_cache_key = f"{request.team_name}_{request.match_limit}_{request.game_title}_{request.map_filter or 'all'}"
        cached_metrics = grid_client._load_metrics_from_cache(metrics_cache_key)
    logger.info(f"[TIMING] cache_check: {time.perf_counter() - t0:.2f}s (hit={cached_metrics is not None})")
    
    # Step 3: Fetch events (skip if cached)
    if cached_metrics is None and settings.grid_include_events and settings.grid_api_key:
        t0 = time.perf_counter()
        series_ids = []
        for match in matches:
            series_id = str(match.get("series_id") or "").strip()
            if series_id:
                series_ids.append(series_id)
        unique_series_ids = list(dict.fromkeys(series_ids))
        max_series = max(1, settings.grid_events_max_series)
        selected_series = unique_series_ids[:max_series]
        if selected_series:
            max_workers = max(1, min(EVENT_FETCH_CONCURRENCY, len(selected_series)))
            start_times = {series_id: time.perf_counter() for series_id in selected_series}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(grid_client.fetch_series_events, series_id): series_id
                    for series_id in selected_series
                }
                for index, future in enumerate(as_completed(future_map), start=1):
                    series_id = future_map[future]
                    try:
                        events = future.result()
                    except GridAPIError:
                        events = None
                    except Exception:
                        events = None
                    if events:
                        events_by_series[series_id] = events
                    elapsed = time.perf_counter() - start_times.get(series_id, t0)
                    logger.info(
                        f"[TIMING] fetch_events[{index}/{len(selected_series)}]: {elapsed:.2f}s"
                    )
        logger.info(
            f"[TIMING] fetch_events_total: {time.perf_counter() - t0:.2f}s ({len(events_by_series)} series)"
        )

    # Step 4: Create analyzer and compute metrics
    t0 = time.perf_counter()
    analyzer = ScoutingAnalyzer(
        matches, team_name=request.team_name, events_by_series=events_by_series,
        cached_full_metrics=cached_metrics
    )
    logger.info(f"[TIMING] analyzer_init: {time.perf_counter() - t0:.2f}s")
    
    t0 = time.perf_counter()
    metrics = analyzer.generate_metrics_summary()
    logger.info(f"[TIMING] generate_metrics: {time.perf_counter() - t0:.2f}s")
    
    # Save to cache
    if not settings.grid_force_live and cached_metrics is None:
        t0 = time.perf_counter()
        grid_client._save_metrics_to_cache(metrics_cache_key, metrics)
        logger.info(f"[TIMING] save_cache: {time.perf_counter() - t0:.2f}s")
    
    # Step 5: Generate insights
    t0 = time.perf_counter()
    insights = get_rag_engine().generate_insights(metrics, request.team_name)
    logger.info(f"[TIMING] generate_insights: {time.perf_counter() - t0:.2f}s")
    
    logger.info(f"[TIMING] TOTAL: {time.perf_counter() - total_start:.2f}s")

    return ScoutReport(
        team_name=request.team_name,
        matches_analyzed=len(matches),
        metrics=metrics,
        insights=insights,
    )


@app.post("/api/scout", response_model=ScoutReport)
async def generate_scout_report(request: ScoutRequest) -> ScoutReport:
    # Always check precomputed data first (fast path)
    precomputed = _load_precomputed_report(request.team_name, request.match_limit)
    if precomputed:
        logger.info(f"[PRECOMPUTED] Returning precomputed report for {request.team_name}")
        return _normalize_precomputed_report(precomputed)
    
    # Fall back to generating live report
    try:
        return _build_report(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except GridAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/scout/stream")
async def stream_scout_report(
    team_name: str = Query(..., min_length=1),
    match_limit: int = Query(50, ge=1, le=100),
    map_filter: str | None = Query(None),
    game_title: str = Query("VALORANT", min_length=1),
) -> StreamingResponse:
    request = ScoutRequest(
        team_name=team_name,
        match_limit=match_limit,
        map_filter=map_filter,
        game_title=game_title,
    )

    def to_payload(report: ScoutReport) -> dict:
        if hasattr(report, "model_dump"):
            return report.model_dump()
        return report.dict()

    def to_metrics_payload(metrics: MetricsSummary) -> dict:
        if hasattr(metrics, "model_dump"):
            return metrics.model_dump()
        if hasattr(metrics, "dict"):
            return metrics.dict()
        return metrics

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream():
        try:
            total_start = time.perf_counter()
            
            # Always check precomputed data first (fast path)
            precomputed = _load_precomputed_report(request.team_name)
            if precomputed:
                logger.info(f"[PRECOMPUTED] Returning precomputed report for {request.team_name}")
                yield event({
                    "type": "progress",
                    "stage": "precomputed",
                    "progress": 50,
                    "message": "Loading precomputed data",
                })
                report = _normalize_precomputed_report(precomputed)
                yield event({
                    "type": "metrics",
                    "team_name": report.team_name,
                    "matches_analyzed": report.matches_analyzed,
                    "metrics": to_metrics_payload(report.metrics),
                })
                for section, content in report.insights.items():
                    yield event({
                        "type": "insight_chunk",
                        "section": section,
                        "content": content,
                    })
                yield event({"type": "done", "report": to_payload(report)})
                return
            
            # Fall back to generating live report
            yield event(
                {
                    "type": "progress",
                    "stage": "fetch_matches",
                    "progress": 15,
                    "message": "Fetching matches",
                }
            )
            t0 = time.perf_counter()
            if settings.grid_api_mode in ("graphql", "auto"):
                matches = await run_in_threadpool(
                    grid_client.fetch_team_matches,
                    request.team_name,
                    request.match_limit,
                    request.game_title,
                )
            else:
                matches = await async_grid_client.fetch_team_matches(
                    request.team_name,
                    limit=request.match_limit,
                    game_title=request.game_title,
                )
            logger.info(f"[STREAM TIMING] fetch_matches: {time.perf_counter() - t0:.2f}s ({len(matches)} matches)")

            if request.map_filter:
                yield event(
                    {
                        "type": "progress",
                        "stage": "filter_matches",
                        "progress": 25,
                        "message": "Filtering matches",
                    }
                )
                map_filter_value = request.map_filter.strip().lower()
                matches = [
                    match
                    for match in matches
                    if (match.get("map") or {}).get("name", "").strip().lower()
                    == map_filter_value
                ]

            if not matches:
                raise HTTPException(status_code=404, detail="No matches found for team.")

            cached_metrics = None
            metrics_cache_key = f"{request.team_name}_{request.match_limit}_{request.game_title}_{request.map_filter or 'all'}"
            # Check cache
            t0 = time.perf_counter()
            if not settings.grid_force_live:
                cached_metrics = grid_client._load_metrics_from_cache(metrics_cache_key)
            logger.info(f"[STREAM TIMING] cache_check: {time.perf_counter() - t0:.2f}s (hit={cached_metrics is not None})")
            
            events_by_series = {}
            # Skip event fetching if we have cached metrics
            if cached_metrics is None and settings.grid_include_events and settings.grid_api_key:
                t0 = time.perf_counter()
                series_ids = []
                for match in matches:
                    series_id = str(match.get("series_id") or "").strip()
                    if series_id:
                        series_ids.append(series_id)
                unique_series_ids = list(dict.fromkeys(series_ids))
                max_series = max(1, settings.grid_events_max_series)
                selected_series = unique_series_ids[:max_series]
                if selected_series and (len(selected_series) >= 5 or request.match_limit >= 30):
                    estimated_seconds = max(1, len(selected_series) * 12)
                    estimated_min = max(1, estimated_seconds // 60)
                    estimated_max = max(estimated_min + 1, (estimated_seconds + 30) // 60)
                    yield event(
                        {
                            "type": "warning",
                            "message": (
                                f"Loading {len(selected_series)} series for {len(matches)} matches may take "
                                f"~{estimated_min}-{estimated_max} minutes. Reduce match count for faster results."
                            ),
                        }
                    )
                total = len(selected_series) or 1
                yield event(
                    {
                        "type": "progress",
                        "stage": "fetch_events",
                        "progress": 35,
                        "message": "Loading event data",
                    }
                )

                semaphore = asyncio.Semaphore(max(1, min(EVENT_FETCH_CONCURRENCY, total)))

                async def _fetch_one(series_id: str) -> tuple[str, list[dict] | None, float]:
                    async with semaphore:
                        start_event = time.perf_counter()
                        try:
                            if settings.grid_api_mode in ("graphql", "auto"):
                                events = await run_in_threadpool(
                                    grid_client.fetch_series_events,
                                    series_id,
                                )
                            else:
                                events = await async_grid_client.fetch_series_events(series_id)
                        except GridAPIError:
                            events = None
                        return series_id, events, time.perf_counter() - start_event

                tasks = [asyncio.create_task(_fetch_one(series_id)) for series_id in selected_series]
                for index, task in enumerate(asyncio.as_completed(tasks), start=1):
                    series_id, events, elapsed = await task
                    if events:
                        events_by_series[series_id] = events
                    logger.info(
                        f"[STREAM TIMING] fetch_events[{index}/{total}]: {elapsed:.2f}s"
                    )
                    yield event(
                        {
                            "type": "progress",
                            "stage": "fetch_events",
                            "progress": 35 + int(((index) / total) * 15),
                            "message": "Loading event data",
                        }
                    )
                logger.info(
                    f"[STREAM TIMING] fetch_events_total: {time.perf_counter() - t0:.2f}s ({len(events_by_series)} series)"
                )
            elif cached_metrics is not None:
                # Skip event loading progress when using cache
                yield event(
                    {
                        "type": "progress",
                        "stage": "fetch_events",
                        "progress": 50,
                        "message": "Using cached data",
                    }
                )

            # Create analyzer
            t0 = time.perf_counter()
            analyzer = ScoutingAnalyzer(
                matches, team_name=request.team_name, events_by_series=events_by_series,
                cached_full_metrics=cached_metrics
            )
            logger.info(f"[STREAM TIMING] analyzer_init: {time.perf_counter() - t0:.2f}s")

            # If we have cached metrics, skip all computation phases
            if cached_metrics is not None:
                yield event(
                    {
                        "type": "progress",
                        "stage": "fast_metrics",
                        "progress": 60,
                        "message": "Using cached data",
                    }
                )
                metrics = cached_metrics
                logger.info(f"[STREAM TIMING] using_cached_metrics")
            else:
                # PHASE 1: Fast metrics (no event parsing) - shows data in ~1-2 seconds
                yield event(
                    {
                        "type": "progress",
                        "stage": "fast_metrics",
                        "progress": 55,
                        "message": "Loading basic stats",
                    }
                )
                t0 = time.perf_counter()
                summary = analyzer.generate_fast_metrics()
                logger.info(f"[STREAM TIMING] generate_fast_metrics: {time.perf_counter() - t0:.2f}s")
                
                yield event(
                    {
                        "type": "metrics",
                        "team_name": request.team_name,
                        "matches_analyzed": len(matches),
                        "metrics": summary,
                    }
                )

                # PHASE 2: Detailed metrics (triggers event parsing)
                yield event(
                    {
                        "type": "progress",
                        "stage": "detailed_metrics",
                        "progress": 60,
                        "message": "Analyzing combat data",
                    }
                )
                t0 = time.perf_counter()
                detailed = analyzer.generate_detailed_metrics()
                logger.info(f"[STREAM TIMING] generate_detailed_metrics: {time.perf_counter() - t0:.2f}s")
                summary.update(detailed)

                # Map event_metrics keys to frontend-expected names
                side_performance = summary.get("side_performance", {})
                if side_performance:
                    attack_data = side_performance.get("attack", {})
                    defense_data = side_performance.get("defense", {})
                    summary["side_metrics"] = {
                        "attack_rounds": attack_data.get("rounds", 0),
                        "attack_wins": attack_data.get("wins", 0),
                        "attack_win_rate": attack_data.get("win_rate", 0.0),
                        "attack_kills": attack_data.get("kills", 0),
                        "attack_deaths": attack_data.get("deaths", 0),
                        "attack_kd": attack_data.get("kd_ratio", 0.0),
                        "defense_rounds": defense_data.get("rounds", 0),
                        "defense_wins": defense_data.get("wins", 0),
                        "defense_win_rate": defense_data.get("win_rate", 0.0),
                        "defense_kills": defense_data.get("kills", 0),
                        "defense_deaths": defense_data.get("deaths", 0),
                        "defense_kd": defense_data.get("kd_ratio", 0.0),
                    }

                combat_data = summary.get("combat", {})
                if combat_data:
                    summary["combat_metrics"] = combat_data

                # Map advanced metrics for frontend
                pace_data = summary.get("pace", {})
                if pace_data:
                    summary["pace_metrics"] = pace_data

                site_bias_data = summary.get("site_bias", {})
                if site_bias_data:
                    summary["site_bias"] = site_bias_data

                first_death_data = summary.get("first_death_context", {})
                if first_death_data:
                    summary["first_death_context"] = first_death_data

                ult_data = summary.get("ultimate_impact", {})
                if ult_data:
                    summary["ultimate_impact"] = ult_data

                man_adv_data = summary.get("man_advantage", {})
                if man_adv_data:
                    summary["man_advantage"] = man_adv_data

                discipline_data = summary.get("discipline", {})
                if discipline_data:
                    summary["discipline"] = discipline_data

                metrics = _convert_to_serializable(summary)
                
                # Save metrics to cache (unless force_live is enabled)
                if not settings.grid_force_live:
                    t0 = time.perf_counter()
                    grid_client._save_metrics_to_cache(metrics_cache_key, metrics)
                    logger.info(f"[STREAM TIMING] save_cache: {time.perf_counter() - t0:.2f}s")
            
            yield event(
                {
                    "type": "metrics",
                    "team_name": request.team_name,
                    "matches_analyzed": len(matches),
                    "metrics": to_metrics_payload(metrics),
                }
            )

            yield event(
                {
                    "type": "progress",
                    "stage": "rag_init",
                    "progress": 70,
                    "message": "Preparing RAG context",
                }
            )
            t0 = time.perf_counter()
            rag = get_rag_engine()
            logger.info(f"[STREAM TIMING] rag_init: {time.perf_counter() - t0:.2f}s")
            yield event(
                {
                    "type": "progress",
                    "stage": "rag_generate",
                    "progress": 85,
                    "message": "Generating insights",
                }
            )
            t0 = time.perf_counter()
            insights: dict = {}
            for section, chunk in rag.generate_insights_stream(
                metrics, request.team_name
            ):
                if section not in insights:
                    insights[section] = ""
                insights[section] += chunk
                yield event(
                    {
                        "type": "insight_chunk",
                        "section": section,
                        "content": chunk,
                    }
                )
            logger.info(f"[STREAM TIMING] generate_insights: {time.perf_counter() - t0:.2f}s")
            logger.info(f"[STREAM TIMING] TOTAL: {time.perf_counter() - total_start:.2f}s")

            report = ScoutReport(
                team_name=request.team_name,
                matches_analyzed=len(matches),
                metrics=metrics,
                insights=insights,
            )
            yield event({"type": "done", "report": to_payload(report)})
        except HTTPException as exc:
            yield event({"type": "error", "message": str(exc.detail)})
        except GridAPIError as exc:
            yield event({"type": "error", "message": str(exc)})
        except Exception as exc:
            yield event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/teams")
async def list_available_teams() -> dict:
    """List all teams with cached match data."""
    cache_dir = Path(__file__).parent.parent / "data" / "debug_cache"
    teams = []
    
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*_matches.json"):
            # Skip event files
            if cache_file.name.startswith("events_"):
                continue
            
            # Extract team name from filename
            team_slug = cache_file.stem.replace("_matches", "")
            
            # Convert slug to display name
            # Handle special cases
            if team_slug == "100_thieves":
                display_name = "100 Thieves"
            elif team_slug == "2game_esports":
                display_name = "2GAME eSports"
            elif team_slug == "loud__1":
                display_name = "LOUD (1)"
            elif team_slug == "leviatán_esports":
                display_name = "Leviatán Esports"
            elif team_slug == "krü_esports":
                display_name = "KRÜ Esports"
            elif team_slug.endswith("_1"):
                # Handle (1) suffix for teams like "mibr_1" -> "MIBR (1)"
                base = team_slug[:-2].replace("_", " ").title()
                display_name = f"{base} (1)"
            else:
                # Standard conversion: "g2_esports" -> "G2 Esports"
                display_name = team_slug.replace("_", " ").title()
                # Fix common abbreviations
                display_name = re.sub(r'\bNrg\b', 'NRG', display_name)
                display_name = re.sub(r'\bGen G\b', 'Gen.G', display_name)
                display_name = re.sub(r'\bMibr\b', 'MIBR', display_name)
                display_name = re.sub(r'\bFuria\b', 'FURIA', display_name)
                display_name = re.sub(r'\bLoud\b', 'LOUD', display_name)
            
            # Get file size to check if it has data
            file_size = cache_file.stat().st_size
            if file_size > 100:  # Only include files with actual data
                # Load match count
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        match_count = len(data) if isinstance(data, list) else 0
                except Exception:
                    match_count = 0
                
                teams.append({
                    "name": display_name,
                    "slug": team_slug,
                    "match_count": match_count,
                    "file_size": file_size
                })
    
    # Sort by name
    teams.sort(key=lambda x: x["name"])
    
    return {
        "teams": teams,
        "count": len(teams)
    }


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "debug_mode": grid_client.debug_mode,
        "api_mode": grid_client.api_mode,
        "use_precomputed": settings.use_precomputed,
        "precomputed_dir": str(PRECOMPUTED_DIR),
    }


@app.get("/api/precomputed/teams")
async def list_precomputed_teams() -> dict:
    """List all teams with precomputed reports."""
    manifest = _load_precomputed_manifest()
    if not manifest:
        return {"teams": [], "count": 0, "available": False}
    
    teams = manifest.get("teams", [])
    return {
        "teams": teams,
        "count": len(teams),
        "available": True,
        "generated_at": manifest.get("generated_at"),
        "match_limit": manifest.get("match_limit"),
    }


@app.get("/api/precomputed/{team_slug}")
async def get_precomputed_report(team_slug: str) -> dict:
    """Get a precomputed report by team slug."""
    report_path = PRECOMPUTED_DIR / "teams" / f"{team_slug}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No precomputed report for {team_slug}")
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
