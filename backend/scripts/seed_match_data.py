from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.env import load_env  # noqa: E402
from app.grid_client import GridAPIError, GridClient  # noqa: E402
from app.settings import Settings  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed GRID match data cache.")
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help=(
            "Team names. Supports comma-separated or space-delimited input. "
            'Example: --teams Cloud9 Sentinels "100 Thieves"'
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("GRID_MATCH_LIMIT", "50")),
        help="Number of matches to cache per team.",
    )
    parser.add_argument(
        "--game-title",
        default=os.getenv("GRID_GAME_TITLE", "VALORANT"),
        help="Game title to filter (e.g., VALORANT).",
    )
    parser.add_argument(
        "--series-page-size",
        type=int,
        default=int(os.getenv("GRID_SERIES_PAGE_SIZE", "50")),
        help="Series page size for GraphQL queries.",
    )
    parser.add_argument(
        "--series-max-pages",
        type=int,
        default=int(os.getenv("GRID_SERIES_MAX_PAGES", "6")),
        help="Max number of GraphQL pages to scan.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("GRID_MAX_RETRIES", "0")),
        help="Maximum retries per team when requests fail. 0 = retry forever.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=int(os.getenv("GRID_RETRY_DELAY", "10")),
        help="Base delay in seconds for retry backoff.",
    )
    return parser.parse_args()


def _parse_team_args(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        values = [str(value).strip() for value in raw if str(value).strip()]
        if not values:
            return []
        if len(values) == 1:
            raw_text = values[0]
            if "," in raw_text:
                return [value.strip() for value in raw_text.split(",") if value.strip()]
            return values
        if any("," in value for value in values):
            joined = " ".join(values)
            return [value.strip() for value in joined.split(",") if value.strip()]
        return values
    raw_text = str(raw).strip()
    if not raw_text:
        return []
    return [value.strip() for value in raw_text.split(",") if value.strip()]


def _discover_all_teams(settings: Settings, args: argparse.Namespace) -> list[str]:
    """Discover all unique teams from recent series."""
    client = GridClient(
        api_key=settings.grid_api_key,
        debug_mode=False,
        api_mode=settings.grid_api_mode,
        fallback_to_rest=settings.grid_fallback_to_rest,
        graphql_url=settings.grid_graphql_url,
        central_data_url=settings.grid_central_data_url,
        file_download_url=settings.grid_file_download_url,
        end_state_path=settings.grid_end_state_path,
        file_download_list_path=settings.grid_file_download_list_path,
        file_download_events_path=settings.grid_file_download_events_path,
        series_page_size=args.series_page_size,
        series_max_pages=args.series_max_pages,
        include_player_team=settings.grid_include_player_team,
        graphql_query_path=settings.grid_graphql_query_path,
    )
    
    # Fetch recent series to discover teams
    print(f"Fetching recent series (limit=500)...")
    series_nodes = client._fetch_series_nodes_graphql(500, args.game_title)
    print(f"Found {len(series_nodes)} recent series")
    
    # Extract unique team names
    team_names = set()
    for node in series_nodes:
        teams = node.get("teams", [])
        for team in teams:
            if isinstance(team, dict):
                # Teams have baseInfo.name structure in GraphQL
                base_info = team.get("baseInfo", {})
                if isinstance(base_info, dict):
                    team_name = base_info.get("name", "").strip()
                else:
                    team_name = team.get("name", "").strip()
                
                if team_name:
                    team_names.add(team_name)
    
    teams = sorted(team_names)
    print(f"Discovered {len(teams)} unique teams: {', '.join(teams)}")
    return teams


def _fetch_team_with_retry(
    client: GridClient,
    team: str,
    limit: int,
    game_title: str,
    max_retries: int,
    base_delay: int,
) -> tuple[list, str | None]:
    attempt = 0
    retry_forever = max_retries <= 0
    while retry_forever or attempt < max_retries:
        try:
            matches = client.fetch_team_matches(
                team, limit=limit, game_title=game_title
            )
            return matches, None
        except GridAPIError as exc:
            error_msg = str(exc)
            is_rate_limit = "429" in error_msg or "too many requests" in error_msg.lower()
            is_auth_error = (
                "403" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg.lower()
            )
            if retry_forever or attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if is_rate_limit:
                    print(
                        f"    ⚠ Rate limit hit, retrying in {delay}s... "
                        f"(attempt {attempt + 1}/{max_retries if not retry_forever else '∞'})"
                    )
                elif is_auth_error:
                    print(
                        f"    ⚠ Auth error, retrying in {delay}s... "
                        f"(attempt {attempt + 1}/{max_retries if not retry_forever else '∞'})"
                    )
                else:
                    print(
                        f"    ⚠ Error: {exc}, retrying in {delay}s... "
                        f"(attempt {attempt + 1}/{max_retries if not retry_forever else '∞'})"
                    )
                time.sleep(delay)
            else:
                return [], error_msg
        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            if retry_forever or attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(
                    f"    ⚠ {error_msg}, retrying in {delay}s... "
                    f"(attempt {attempt + 1}/{max_retries if not retry_forever else '∞'})"
                )
                time.sleep(delay)
            else:
                return [], error_msg
        attempt += 1
    return [], "Max retries exceeded"


def seed_data(args: argparse.Namespace) -> None:
    load_env()

    settings = Settings.from_env()
    if not settings.grid_api_key:
        raise ValueError("GRID_API_KEY not found in environment.")

    teams = _parse_team_args(args.teams)
    if not teams:
        teams = _parse_team_args(os.getenv("GRID_TEAMS", ""))
    if not teams:
        # Discover all teams from recent series
        print("No teams specified. Discovering teams from recent series...")
        teams = _discover_all_teams(settings, args)

    client = GridClient(
        api_key=settings.grid_api_key,
        debug_mode=False,
        api_mode=settings.grid_api_mode,
        fallback_to_rest=settings.grid_fallback_to_rest,
        graphql_url=settings.grid_graphql_url,
        central_data_url=settings.grid_central_data_url,
        file_download_url=settings.grid_file_download_url,
        end_state_path=settings.grid_end_state_path,
        file_download_list_path=settings.grid_file_download_list_path,
        file_download_events_path=settings.grid_file_download_events_path,
        series_page_size=args.series_page_size,
        series_max_pages=args.series_max_pages,
        include_player_team=settings.grid_include_player_team,
        graphql_query_path=settings.grid_graphql_query_path,
    )

    print(f"\nFetching matches for {len(teams)} teams...")
    print(
        f"Retry settings: max_retries={args.max_retries if args.max_retries > 0 else '∞'}, "
        f"base_delay={args.retry_delay}s"
    )
    success_count = 0
    failed_teams: dict[str, str] = {}
    
    for i, team in enumerate(teams, 1):
        print(f"[{i}/{len(teams)}] Fetching {team} (limit={args.limit})...")
        try:
            cached = client._load_from_cache(team)
            if cached is not None and len(cached) > 0:
                print(f"  ✓ Already cached ({len(cached)} matches)")
                success_count += 1
                continue

            matches, error = _fetch_team_with_retry(
                client,
                team,
                args.limit,
                args.game_title,
                args.max_retries,
                args.retry_delay,
            )
            if error is None:
                print(f"  ✓ Cached {len(matches)} matches")
                success_count += 1
            else:
                print(f"  ✗ Failed after {args.max_retries} attempts: {error}")
                failed_teams[team] = error

            if i < len(teams):
                time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted by user. Continuing with summary...")
            break

    print(f"\n{'='*60}")
    print(f"Data seeding complete.")
    print(f"Success: {success_count}/{len(teams)} teams")
    if failed_teams:
        print("Failed teams:")
        for team, error in failed_teams.items():
            print(f"  - {team}: {error}")
    print(f"{'='*60}")


if __name__ == "__main__":
    seed_data(_parse_args())
