"""Script to seed match data from specific VCT tournament series."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.env import load_env
from app.grid_client import GridAPIError, GridClient
from app.settings import Settings


# VCT Americas tournament series IDs
VCT_TOURNAMENTS = {
    "757371": "VCT Americas - Kickoff 2024",
    "757481": "VCT Americas - Stage 1 2024",
    "774782": "VCT Americas - Stage 2 2024",
    "775516": "VCT Americas - Kickoff 2025",
    "800675": "VCT Americas - Stage 1 2025",
    "826660": "VCT Americas - Stage 2 2025",
    "757614": "VALORANT Masters - Masters Madrid",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed GRID match data from specific VCT tournaments."
    )
    parser.add_argument(
        "--tournament-ids",
        nargs="*",
        default=None,
        help=(
            "Tournament series IDs. If not provided, all VCT Americas tournaments will be fetched. "
            "Example: --tournament-ids 757371 757481"
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
        default=int(os.getenv("GRID_SERIES_MAX_PAGES", "20")),
        help="Max number of GraphQL pages to scan per tournament.",
    )
    return parser.parse_args()


def _discover_teams_from_tournaments(
    client: GridClient, tournament_ids: list[str], game_title: str
) -> dict[str, int]:
    """Discover all teams that participated in the specified tournaments."""
    print(f"\nDiscovering teams from {len(tournament_ids)} tournaments...")
    print("=" * 80)
    
    all_series = []
    for tournament_id in tournament_ids:
        tournament_name = VCT_TOURNAMENTS.get(tournament_id, f"Tournament {tournament_id}")
        print(f"Fetching series for: {tournament_name}")
        
        try:
            # Fetch all series (matches) from this tournament
            # We'll fetch more pages to ensure we get all matches
            series_nodes = client._fetch_series_nodes_graphql(1000, game_title)
            
            # Filter by tournament ID
            tournament_series = [
                node for node in series_nodes
                if node.get("tournament", {}).get("id") == tournament_id
            ]
            
            print(f"  Found {len(tournament_series)} series in this tournament")
            all_series.extend(tournament_series)
            
        except Exception as exc:
            print(f"  ✗ Error fetching tournament data: {exc}")
            continue
    
    # Extract unique teams
    team_match_counts = defaultdict(int)
    for node in all_series:
        teams = node.get("teams", [])
        for team in teams:
            if isinstance(team, dict):
                base_info = team.get("baseInfo", {})
                if isinstance(base_info, dict):
                    team_name = base_info.get("name", "").strip()
                else:
                    team_name = team.get("name", "").strip()
                
                if team_name:
                    team_match_counts[team_name] += 1
    
    print(f"\n{'=' * 80}")
    print(f"Discovered {len(team_match_counts)} unique teams from tournaments")
    print(f"Total series found: {len(all_series)}")
    print(f"{'=' * 80}\n")
    
    # Print teams sorted by match count
    sorted_teams = sorted(team_match_counts.items(), key=lambda x: x[1], reverse=True)
    print("Teams by match count:")
    for team_name, count in sorted_teams:
        print(f"  • {team_name}: {count} matches")
    
    return team_match_counts


def seed_tournament_data(args: argparse.Namespace) -> None:
    load_env()

    settings = Settings.from_env()
    if not settings.grid_api_key:
        raise ValueError("GRID_API_KEY not found in environment.")

    # Determine which tournaments to fetch
    tournament_ids = args.tournament_ids or list(VCT_TOURNAMENTS.keys())
    
    print("VCT Tournament Data Seeding")
    print("=" * 80)
    print("Tournaments to fetch:")
    for tid in tournament_ids:
        name = VCT_TOURNAMENTS.get(tid, f"Unknown tournament {tid}")
        print(f"  • {tid}: {name}")
    
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

    # Discover teams from tournaments
    team_match_counts = _discover_teams_from_tournaments(
        client, tournament_ids, args.game_title
    )
    
    if not team_match_counts:
        print("No teams found in specified tournaments.")
        return
    
    # Fetch match data for each team
    teams = list(team_match_counts.keys())
    print(f"\nFetching matches for {len(teams)} teams...")
    print("=" * 80)
    
    success_count = 0
    failed_teams = []
    
    for i, team in enumerate(teams, 1):
        match_count = team_match_counts[team]
        print(f"[{i}/{len(teams)}] Fetching {team} ({match_count} matches in tournaments, requesting {args.limit} total)...")
        
        try:
            matches = client.fetch_team_matches(
                team, limit=args.limit, game_title=args.game_title
            )
            print(f"  ✓ Cached {len(matches)} matches")
            success_count += 1
            
            # Add delay to avoid rate limiting (5 seconds between teams)
            if i < len(teams):
                time.sleep(5)
                
        except (GridAPIError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                print(f"\n  ⚠ Interrupted by user. Continuing with summary...")
                break
            print(f"  ✗ Failed to fetch matches: {exc}")
            failed_teams.append(team)
            
            # Add delay even on failure
            if i < len(teams):
                time.sleep(5)
            continue
            
        except Exception as exc:
            print(f"  ✗ Unexpected error: {exc}")
            failed_teams.append(team)
            if i < len(teams):
                time.sleep(5)
            continue

    print(f"\n{'=' * 80}")
    print(f"Data seeding complete.")
    print(f"Success: {success_count}/{len(teams)} teams")
    if failed_teams:
        print(f"Failed: {', '.join(failed_teams)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    seed_tournament_data(_parse_args())
