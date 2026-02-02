#!/usr/bin/env python3
"""
Generate precomputed JSON files for frontend-only deployment.

This script calls the running backend API to get scout reports and saves them
as static JSON files for frontend-only deployment.

REQUIRES: Backend server running at http://localhost:8000

Usage:
    python scripts/precompute_for_frontend.py
    python scripts/precompute_for_frontend.py --teams "Cloud9" "Sentinels"
    python scripts/precompute_for_frontend.py --limit 20
    python scripts/precompute_for_frontend.py --backend-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate precomputed JSON files by calling backend API."
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help="Specific team names to process. If omitted, processes all available teams.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of matches to request (default: 20).",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8080",
        help="Backend API URL (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: ../frontend/public/precomputed",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300 = 5 min).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually calling API.",
    )
    return parser.parse_args()


def get_available_teams(backend_url: str) -> list[dict]:
    """Fetch available teams from backend API."""
    try:
        response = requests.get(f"{backend_url}/api/teams", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("teams", [])
    except Exception as e:
        print(f"❌ Failed to fetch teams: {e}")
        return []


def slugify(name: str) -> str:
    """Convert team name to slug for filenames."""
    import re
    return re.sub(r"[^\w\-]", "_", name.lower()).strip("_")


def fetch_scout_report(
    backend_url: str,
    team_name: str,
    match_limit: int,
    timeout: int,
) -> dict | None:
    """Fetch scout report from backend API using SSE stream."""
    url = f"{backend_url}/api/scout/stream"
    params = {
        "team_name": team_name,
        "match_limit": match_limit,
        "game_title": "VALORANT",
    }
    
    try:
        # Use streaming to handle SSE
        with requests.get(url, params=params, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            
            report = None
            metrics = None
            insights = {}
            team_name_out = team_name
            matches_analyzed = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    msg_type = data.get("type")
                    
                    if msg_type == "progress":
                        stage = data.get("stage", "")
                        progress = data.get("progress", 0)
                        message = data.get("message", "")
                        print(f"    [{progress:3d}%] {message}", end="\r")
                    
                    elif msg_type == "metrics":
                        metrics = data.get("metrics")
                        team_name_out = data.get("team_name", team_name)
                        matches_analyzed = data.get("matches_analyzed", 0)
                        print(f"    ✓ Metrics received ({matches_analyzed} matches)       ")
                    
                    elif msg_type == "insight_chunk":
                        section = data.get("section", "")
                        content = data.get("content", "")
                        insights[section] = insights.get(section, "") + content
                    
                    elif msg_type == "done":
                        report = data.get("report")
                        print(f"    ✓ Report complete                              ")
                        break
                    
                    elif msg_type == "error":
                        print(f"    ✗ Error: {data.get('message')}")
                        return None
                
                except json.JSONDecodeError:
                    continue
            
            # Build report from collected data if not provided directly
            if report:
                return report
            elif metrics:
                return {
                    "team_name": team_name_out,
                    "matches_analyzed": matches_analyzed,
                    "metrics": metrics,
                    "insights": insights,
                }
            
            return None
            
    except requests.exceptions.Timeout:
        print(f"    ✗ Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def write_json(path: Path, data: dict) -> None:
    """Write JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    print(f"  → Wrote {path.name} ({size_kb:.1f} KB)")


def main() -> None:
    args = _parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parents[2] / "frontend" / "public" / "precomputed"
    
    teams_dir = output_dir / "teams"
    
    print(f"Backend URL: {args.backend_url}")
    print(f"Output directory: {output_dir}")
    print(f"Match limit: {args.limit}")
    print(f"Timeout: {args.timeout}s")
    print()
    
    # Check backend is running
    try:
        health = requests.get(f"{args.backend_url}/", timeout=5)
        print(f"✓ Backend is running")
    except Exception as e:
        print(f"❌ Backend not reachable at {args.backend_url}")
        print(f"   Error: {e}")
        print(f"\n   Start the backend first: cd backend && uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Get available teams
    all_teams = get_available_teams(args.backend_url)
    
    if not all_teams:
        print("❌ No teams available from backend.")
        sys.exit(1)
    
    # Filter to specified teams if provided
    if args.teams:
        team_names_lower = [t.lower() for t in args.teams]
        all_teams = [
            t for t in all_teams 
            if t["slug"].lower() in team_names_lower or t["name"].lower() in team_names_lower
        ]
    
    # Filter out teams with 0 matches
    all_teams = [t for t in all_teams if t.get("match_count", 0) > 0]
    
    if not all_teams:
        print("❌ No matching teams found with matches.")
        sys.exit(1)
    
    print(f"\nProcessing {len(all_teams)} team(s):")
    for t in all_teams:
        print(f"  - {t['name']} ({t.get('match_count', '?')} matches)")
    print()
    
    if args.dry_run:
        print("Dry run - no API calls will be made.")
        return
    
    # Generate reports
    manifest_entries = []
    successful = 0
    failed = 0
    
    for i, team_info in enumerate(all_teams, 1):
        team_name = team_info["name"]
        slug = team_info.get("slug") or slugify(team_name)
        
        print(f"\n[{i}/{len(all_teams)}] {team_name}")
        print(f"{'─' * 50}")
        
        start_time = time.perf_counter()
        report = fetch_scout_report(
            backend_url=args.backend_url,
            team_name=team_name,
            match_limit=args.limit,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - start_time
        
        if report:
            # Add metadata
            report["generated_at"] = datetime.now(timezone.utc).isoformat()
            report["match_limit"] = args.limit
            
            # Write team report
            team_file = teams_dir / f"{slug}.json"
            write_json(team_file, report)
            
            # Add to manifest
            manifest_entries.append({
                "name": team_name,
                "slug": slug,
                "match_count": report.get("matches_analyzed", 0),
                "has_insights": bool(report.get("insights")),
                "generated_at": report["generated_at"],
            })
            
            successful += 1
            print(f"  ✓ Done in {elapsed:.1f}s")
        else:
            failed += 1
            print(f"  ✗ Failed after {elapsed:.1f}s")
    
    # Write manifest
    manifest = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_limit": args.limit,
        "teams": manifest_entries,
    }
    
    manifest_file = output_dir / "manifest.json"
    write_json(manifest_file, manifest)
    
    print(f"\n{'═' * 50}")
    print(f"Done! Generated {successful} team reports ({failed} failed).")
    print(f"Manifest: {manifest_file}")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
