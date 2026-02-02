"""Seed match data for all available VALORANT teams."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.grid_client import GridClient
from app.env import load_env
import json
from collections import defaultdict

def sanitize_filename(team_name: str) -> str:
    """Convert team name to safe filename."""
    return team_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

def main():
    load_env()
    
    api_key = os.getenv("GRID_API_KEY")
    if not api_key:
        print("Error: GRID_API_KEY not found")
        return
    
    print("Discovering all VALORANT teams...")
    print("=" * 80)
    
    client = GridClient(api_key=api_key)
    
    # Discover teams from series
    series_nodes = client._fetch_series_nodes_graphql(
        limit=500,
        game_title="VALORANT",
        team_name=None
    )
    
    teams_data = defaultdict(lambda: {"matches": 0})
    
    for node in series_nodes:
        teams = node.get("teams", [])
        for team in teams:
            if isinstance(team, dict):
                base_info = team.get("baseInfo", {})
                if isinstance(base_info, dict):
                    team_name = base_info.get("name", "").strip()
                else:
                    team_name = team.get("name", "").strip()
                
                if team_name:
                    teams_data[team_name]["matches"] += 1
    
    # Sort by match count
    sorted_teams = sorted(
        teams_data.keys(),
        key=lambda x: teams_data[x]["matches"],
        reverse=True
    )
    
    print(f"Found {len(sorted_teams)} teams")
    print(f"Will cache data for all teams...")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data" / "debug_cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    
    for i, team_name in enumerate(sorted_teams, 1):
        match_count = teams_data[team_name]["matches"]
        print(f"\n[{i}/{len(sorted_teams)}] Fetching {team_name} ({match_count} matches)...")
        
        try:
            matches = client.fetch_team_matches(
                team_name=team_name,
                limit=50,
                game_title="VALORANT"
            )
            
            if matches:
                filename = f"{sanitize_filename(team_name)}_matches.json"
                filepath = data_dir / filename
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(matches, f, indent=2, ensure_ascii=False)
                
                print(f"  ✓ Cached {len(matches)} matches to {filename}")
                successful.append(team_name)
            else:
                print(f"  ⚠ No matches returned")
                failed.append((team_name, "No matches"))
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append((team_name, str(e)))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully cached: {len(successful)} teams")
    print(f"Failed: {len(failed)} teams")
    
    if successful:
        print("\n✓ Cached teams:")
        for team in successful:
            print(f"  • {team}")
    
    if failed:
        print("\n✗ Failed teams:")
        for team, reason in failed:
            print(f"  • {team}: {reason}")

if __name__ == "__main__":
    main()
