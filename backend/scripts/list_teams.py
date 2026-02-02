"""Script to discover available teams from Grid API series data."""
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.grid_client import GridClient
from app.env import load_env

def main():
    # Load environment variables
    load_env()
    
    api_key = os.getenv("GRID_API_KEY")
    if not api_key:
        print("Error: GRID_API_KEY not found in environment")
        return
    
    print("Discovering VALORANT teams from recent series...")
    print("=" * 80)
    
    client = GridClient(api_key=api_key)
    
    try:
        # Fetch recent VALORANT series to discover teams
        print("Fetching recent VALORANT series...")
        series_nodes = client._fetch_series_nodes_graphql(
            limit=500,  # Increased to find more teams
            game_title="VALORANT",
            team_name=None
        )
        
        print(f"Found {len(series_nodes)} recent series\n")
        
        # Extract all unique teams
        teams_data = defaultdict(lambda: {"matches": 0, "series_ids": set()})
        
        for node in series_nodes:
            series_id = node.get("id", "")
            teams = node.get("teams", [])
            
            for team in teams:
                if isinstance(team, dict):
                    # Teams have baseInfo.name structure in GraphQL
                    base_info = team.get("baseInfo", {})
                    if isinstance(base_info, dict):
                        team_name = base_info.get("name", "").strip()
                    else:
                        team_name = team.get("name", "").strip()
                    
                    team_id = team.get("id", "")
                    
                    if team_name:
                        teams_data[team_name]["id"] = team_id
                        teams_data[team_name]["matches"] += 1
                        teams_data[team_name]["series_ids"].add(series_id)
        
        print(f"Discovered {len(teams_data)} unique teams\n")
        
        # Sort by number of matches (most active first)
        sorted_teams = sorted(
            teams_data.items(),
            key=lambda x: x[1]["matches"],
            reverse=True
        )
        
        print("Top 50 Most Active Teams:")
        print("-" * 80)
        print(f"{'Team Name':<40} | {'Matches':<8} | {'Team ID':<10}")
        print("-" * 80)
        
        for team_name, data in sorted_teams[:50]:
            print(f"{team_name:<40} | {data['matches']:<8} | {data.get('id', 'N/A'):<10}")
        
        # Show teams we have cached data for
        print("\n" + "=" * 80)
        print("Teams with cached match data:")
        print("-" * 80)
        
        data_dir = Path(__file__).parent.parent / "data" / "debug_cache"
        if data_dir.exists():
            cache_files = list(data_dir.glob("*_matches.json"))
            cached_teams = []
            for f in cache_files:
                team_name = f.stem.replace("_matches", "").replace("_", " ")
                # Try to match with discovered teams (case insensitive)
                matched = False
                for discovered_team, _ in sorted_teams:
                    if team_name.lower() in discovered_team.lower() or discovered_team.lower() in team_name.lower():
                        cached_teams.append(discovered_team)
                        matched = True
                        break
                if not matched:
                    cached_teams.append(team_name.title())
            
            for cached in sorted(set(cached_teams)):
                print(f"  ✓ {cached}")
        
        # Suggest popular teams not yet cached
        print("\n" + "=" * 80)
        print("Popular teams you could add (Top 20 by match count):")
        print("-" * 80)
        
        cached_names_lower = {t.lower() for t in cached_teams}
        suggestions = []
        
        for team_name, data in sorted_teams[:30]:
            if team_name.lower() not in cached_names_lower:
                suggestions.append((team_name, data['matches']))
        
        for team_name, match_count in suggestions[:20]:
            print(f"  • {team_name} ({match_count} recent matches)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
