#!/usr/bin/env python3
"""
Pre-compute metrics for all cached teams to speed up future dashboard loads.

This script:
1. Finds all team match files in the debug_cache
2. Computes full metrics for each team
3. Saves the metrics to cache files

Usage:
    python scripts/precompute_metrics.py
    python scripts/precompute_metrics.py --teams "Cloud9" "Sentinels"
    python scripts/precompute_metrics.py --limit 20
    python scripts/precompute_metrics.py --force  # Recompute even if cache exists
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_env  # noqa: E402
from app.analyzer import ScoutingAnalyzer, _convert_to_serializable  # noqa: E402
from app.grid_client import GridClient, GridAPIError  # noqa: E402
from app.settings import Settings  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute metrics cache for all teams."
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help="Specific team names to process. If omitted, processes all cached teams.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of matches to use for metrics (default: 20).",
    )
    parser.add_argument(
        "--game-title",
        default="VALORANT",
        help="Game title filter (default: VALORANT).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if cache already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be computed without actually computing.",
    )
    return parser.parse_args()


def get_cached_teams(cache_dir: Path) -> list[str]:
    """Find all team names from cached match files."""
    teams = []
    for f in cache_dir.glob("*_matches.json"):
        # Extract team name from filename (e.g., "cloud9_matches.json" -> "Cloud9")
        name = f.stem.replace("_matches", "")
        # Convert snake_case back to proper name
        # e.g., "100_thieves" -> "100 Thieves", "paper_rex" -> "Paper Rex"
        name = name.replace("_", " ").title()
        # Special case for numbers at start
        if name[0].isdigit():
            name = name  # Keep as is for names like "100 Thieves"
        teams.append(name)
    return sorted(teams)


def get_cache_key(team_name: str, limit: int, game_title: str) -> str:
    """Generate cache key matching main.py format."""
    return f"{team_name}_{limit}_{game_title}_all"


def metrics_cache_exists(cache_dir: Path, cache_key: str) -> bool:
    """Check if metrics cache file already exists."""
    safe_key = re.sub(r"[^\w\-]", "_", cache_key.lower())
    cache_file = cache_dir / f"metrics_{safe_key}.json"
    return cache_file.exists()


def compute_team_metrics(
    grid_client: GridClient,
    team_name: str,
    limit: int,
    game_title: str,
    settings: Settings,
) -> dict | None:
    """Compute full metrics for a team."""
    try:
        # Fetch matches
        matches = grid_client.fetch_team_matches(
            team_name,
            limit=limit,
            game_title=game_title,
        )

        if not matches:
            print(f"  ⚠️  No matches found for {team_name}")
            return None

        # Fetch events for each series
        events_by_series = {}
        if settings.grid_include_events and settings.grid_api_key:
            series_ids = []
            for match in matches:
                series_id = str(match.get("series_id") or "").strip()
                if series_id:
                    series_ids.append(series_id)
            
            unique_series_ids = list(dict.fromkeys(series_ids))
            max_series = max(1, settings.grid_events_max_series)
            
            print(f"  📡 Fetching events for {min(len(unique_series_ids), max_series)} series...")
            
            for i, series_id in enumerate(unique_series_ids[:max_series]):
                try:
                    events = grid_client.fetch_series_events(series_id)
                    if events:
                        events_by_series[series_id] = events
                except GridAPIError as e:
                    print(f"    ⚠️  Failed to fetch events for series {series_id}: {e}")
                    continue

        # Compute metrics
        print(f"  🔢 Computing metrics from {len(matches)} matches...")
        analyzer = ScoutingAnalyzer(
            matches,
            team_name=team_name,
            events_by_series=events_by_series,
            cached_event_metrics=None,  # Force fresh computation
        )
        metrics = analyzer.generate_metrics_summary()
        
        return metrics

    except Exception as e:
        print(f"  ❌ Error computing metrics for {team_name}: {e}")
        return None


def save_metrics_to_cache(grid_client: GridClient, cache_key: str, metrics: dict) -> None:
    """Save metrics to cache file."""
    grid_client._save_metrics_to_cache(cache_key, metrics)


def main():
    load_env()
    args = _parse_args()
    settings = Settings.from_env()
    
    cache_dir = Path(__file__).parent.parent / "data" / "debug_cache"
    
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    # Get teams to process
    if args.teams:
        teams = args.teams
    else:
        teams = get_cached_teams(cache_dir)
    
    if not teams:
        print("❌ No teams found to process.")
        sys.exit(1)
    
    print(f"🎯 Found {len(teams)} teams to process")
    print(f"📊 Settings: limit={args.limit}, game={args.game_title}, force={args.force}")
    print("-" * 60)
    
    # Initialize client
    grid_client = GridClient(
        api_key=settings.grid_api_key,
        debug_mode=True,  # Always use debug mode for caching
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
    
    # Process each team
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, team_name in enumerate(teams, 1):
        cache_key = get_cache_key(team_name, args.limit, args.game_title)
        
        print(f"\n[{i}/{len(teams)}] {team_name}")
        
        # Check if cache exists
        if not args.force and metrics_cache_exists(cache_dir, cache_key):
            print(f"  ✅ Cache already exists (use --force to recompute)")
            skip_count += 1
            continue
        
        if args.dry_run:
            print(f"  🔍 Would compute: {cache_key}")
            continue
        
        start_time = time.time()
        metrics = compute_team_metrics(
            grid_client,
            team_name,
            args.limit,
            args.game_title,
            settings,
        )
        
        if metrics:
            save_metrics_to_cache(grid_client, cache_key, metrics)
            elapsed = time.time() - start_time
            print(f"  ✅ Cached in {elapsed:.1f}s")
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  ✅ Successfully computed: {success_count}")
    print(f"  ⏭️  Skipped (cached):      {skip_count}")
    print(f"  ❌ Failed:                {fail_count}")
    print(f"  📁 Total teams:           {len(teams)}")


if __name__ == "__main__":
    main()
