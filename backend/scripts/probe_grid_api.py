from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.env import load_env  # noqa: E402
from app.grid_client import _auth_header_variants, _is_auth_error  # noqa: E402


def _load_env() -> None:
    load_env()


@dataclass(frozen=True)
class GraphQLTest:
    name: str
    target: str
    query: str
    variables: Optional[Dict[str, Any]] = None


def _split_urls(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _graphql_targets() -> Dict[str, List[str]]:
    central_urls = _split_urls(
        os.getenv("GRID_GRAPHQL_URLS", "").strip()
    ) or [
        os.getenv("GRID_GRAPHQL_URL", "https://api-op.grid.gg/central-data/graphql").strip()
    ]
    series_state_urls = _split_urls(
        os.getenv("GRID_SERIES_STATE_GRAPHQL_URLS", "").strip()
    )
    if not series_state_urls:
        series_state_url = os.getenv(
            "GRID_SERIES_STATE_GRAPHQL_URL",
            "https://api-op.grid.gg/live-data-feed/series-state/graphql",
        ).strip()
        if series_state_url:
            series_state_urls = [series_state_url]
    stats_urls = _split_urls(os.getenv("GRID_STATS_GRAPHQL_URLS", "").strip())
    if not stats_urls:
        stats_url = os.getenv("GRID_STATS_GRAPHQL_URL", "").strip()
        if stats_url:
            stats_urls = [stats_url]

    targets = {
        "central-data": [url for url in central_urls if url],
    }
    if series_state_urls:
        targets["series-state"] = [url for url in series_state_urls if url]
    if stats_urls:
        targets["stats"] = [url for url in stats_urls if url]
    return {name: urls for name, urls in targets.items() if urls}


def _build_queries() -> List[GraphQLTest]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    in_24h = now + timedelta(hours=24)
    gte = now.isoformat().replace("+00:00", "Z")
    lte = in_24h.isoformat().replace("+00:00", "Z")

    team_id = os.getenv("GRID_TEST_TEAM_ID", "83").strip()
    player_id = os.getenv("GRID_TEST_PLAYER_ID", "244").strip()
    org_id = os.getenv("GRID_TEST_ORG_ID", "1").strip()
    tournament_ids = os.getenv("GRID_TEST_TOURNAMENT_IDS", "1,2").strip()
    tournament_list = [tid.strip() for tid in tournament_ids.split(",") if tid.strip()]
    series_id = os.getenv("GRID_TEST_SERIES_ID", "").strip()

    tournament_fragment = """
fragment tournamentFields on Tournament {
  id
  name
  nameShortened
}
""".strip()

    series_fragment = """
fragment seriesFields on Series {
  id
  title {
    nameShortened
  }
  tournament {
    nameShortened
  }
  startTimeScheduled
  format {
    name
    nameShortened
  }
  teams {
    baseInfo {
      name
    }
    scoreAdvantage
  }
}
""".strip()

    org_fragment = """
fragment organizationFields on Organization {
  id
  name
  teams {
    name
  }
}
""".strip()

    team_fragment = """
fragment teamFields on Team {
  id
  name
  colorPrimary
  colorSecondary
  logoUrl
  externalLinks {
    dataProvider {
      name
    }
    externalEntity {
      id
    }
  }
}
""".strip()

    player_fragment = """
fragment playerFields on Player {
  id
  nickname
  title {
    name
  }
}
""".strip()

    tests: List[GraphQLTest] = [
        GraphQLTest(
            name="GetTournaments",
            target="central-data",
            query=f"""
query GetTournaments {{
  tournaments {{
    pageInfo {{
      hasPreviousPage
      hasNextPage
      startCursor
      endCursor
    }}
    totalCount
    edges {{
      cursor
      node {{
        ...tournamentFields
      }}
    }}
  }}
}}
{tournament_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetAllSeriesInNext24Hours",
            target="central-data",
            query=f"""
query GetAllSeriesInNext24Hours {{
  allSeries(
    filter:{{
      startTimeScheduled:{{
        gte: "{gte}"
        lte: "{lte}"
      }}
    }}
    orderBy: StartTimeScheduled
  ) {{
    totalCount
    pageInfo {{
      hasPreviousPage
      hasNextPage
      startCursor
      endCursor
    }}
    edges {{
      cursor
      node {{
        ...seriesFields
      }}
    }}
  }}
}}
{series_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="SeriesFormats",
            target="central-data",
            query="""
query SeriesFormats {
  seriesFormats {
    id
    name
    nameShortened
  }
}
""".strip(),
        ),
        GraphQLTest(
            name="GetOrganization",
            target="central-data",
            query=f"""
query GetOrganization {{
  organization(id: "{org_id}") {{
    ...organizationFields
  }}
}}
{org_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetOrganizations",
            target="central-data",
            query=f"""
query GetOrganizations {{
  organizations(first: 5) {{
    edges {{
      node {{
        ...organizationFields
      }}
    }}
  }}
}}
{org_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetTeams",
            target="central-data",
            query=f"""
query GetTeams {{
  teams(first: 5, after: null) {{
    totalCount
    pageInfo {{
      hasPreviousPage
      hasNextPage
      startCursor
      endCursor
    }}
    edges {{
      cursor
      node {{
        ...teamFields
      }}
    }}
  }}
}}
{team_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetPlayer",
            target="central-data",
            query=f"""
query GetPlayer {{
  player(id: "{player_id}") {{
    ...playerFields
  }}
}}
{player_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetPlayers",
            target="central-data",
            query=f"""
query GetPlayers {{
  players(first: 5) {{
    edges {{
      node {{
        ...playerFields
      }}
    }}
  }}
}}
{player_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="GetTeamRoster",
            target="central-data",
            query=f"""
query GetTeamRoster {{
  players(filter: {{teamIdFilter: {{id: "{team_id}"}}}}) {{
    edges {{
      node {{
        ...playerFields
      }}
    }}
    pageInfo {{
      hasNextPage
      hasPreviousPage
    }}
  }}
}}
{player_fragment}
""".strip(),
        ),
        GraphQLTest(
            name="TeamStatisticsForLastThreeMonths",
            target="stats",
            query=f"""
query TeamStatisticsForLastThreeMonths {{
  teamStatistics(teamId: "{team_id}", filter: {{ timeWindow: LAST_3_MONTHS }}) {{
    id
    aggregationSeriesIds
    series {{
      count
      kills {{
        sum
        min
        max
        avg
      }}
    }}
    game {{
      count
      wins {{
        value
        count
        percentage
        streak {{
          min
          max
          current
        }}
      }}
    }}
    segment {{
      type
      count
      deaths {{
        sum
        min
        max
        avg
      }}
    }}
  }}
}}
""".strip(),
        ),
        GraphQLTest(
            name="TeamStatisticsForChosenTournaments",
            target="stats",
            query=f"""
query TeamStatisticsForChosenTournaments {{
  teamStatistics(teamId: "{team_id}", filter: {{ tournamentIds: {{ in: {json.dumps(tournament_list)} }} }}) {{
    id
    aggregationSeriesIds
    series {{
      count
      kills {{
        sum
        min
        max
        avg
      }}
    }}
    game {{
      count
      wins {{
        value
        count
        percentage
        streak {{
          min
          max
          current
        }}
      }}
    }}
    segment {{
      type
      count
      deaths {{
        sum
        min
        max
        avg
      }}
    }}
  }}
}}
""".strip(),
        ),
        GraphQLTest(
            name="PlayerStatisticsForLastThreeMonths",
            target="stats",
            query=f"""
query PlayerStatisticsForLastThreeMonths {{
  playerStatistics(playerId: "{player_id}", filter: {{ timeWindow: LAST_3_MONTHS }}) {{
    id
    aggregationSeriesIds
    series {{
      count
      kills {{
        sum
        min
        max
        avg
      }}
    }}
    game {{
      count
      wins {{
        value
        count
        percentage
        streak {{
          min
          max
          current
        }}
      }}
    }}
    segment {{
      type
      count
      deaths {{
        sum
        min
        max
        avg
      }}
    }}
  }}
}}
""".strip(),
        ),
        GraphQLTest(
            name="PlayerStatisticsForChosenTournaments",
            target="stats",
            query=f"""
query PlayerStatisticsForChosenTournaments {{
  playerStatistics(playerId: "{player_id}", filter: {{ tournamentIds: {{ in: {json.dumps(tournament_list)} }} }}) {{
    id
    aggregationSeriesIds
    series {{
      count
      kills {{
        sum
        min
        max
        avg
      }}
    }}
    game {{
      count
      wins {{
        value
        count
        percentage
        streak {{
          min
          max
          current
        }}
      }}
    }}
    segment {{
      type
      count
      deaths {{
        sum
        min
        max
        avg
      }}
    }}
  }}
}}
""".strip(),
        ),
    ]
    if series_id:
        tests.append(
            GraphQLTest(
                name="SeriesState",
                target="series-state",
                query=(
                    f"""
query SeriesState {{
  seriesState(id: "{series_id}") {{
    id
    started
    finished
    teams {{
      id
      name
      won
    }}
  }}
}}
""".strip()
                ),
            )
        )
    return tests


def _post_graphql(
    url: str, api_key: str, query: str, variables: Optional[Dict]
) -> tuple[requests.Response, str]:
    payload = {"query": query, "variables": variables or {}}
    variants = _auth_header_variants(api_key)
    response: Optional[requests.Response] = None
    header_used = ""
    for index, headers in enumerate(variants):
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        header_used = ",".join(sorted(headers.keys()))
        if response.status_code == 401 and index < len(variants) - 1:
            continue
        if response.status_code == 200:
            try:
                payload_data = response.json()
            except ValueError:
                break
            if payload_data.get("errors") and _is_auth_error(payload_data["errors"]):
                if index < len(variants) - 1:
                    continue
        break
    return response, header_used


def _evaluate_graphql(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"FAIL {response.status_code} non-JSON response"

    if response.status_code != 200:
        return f"FAIL {response.status_code} {payload}"
    if "errors" in payload:
        first = payload["errors"][0] if payload["errors"] else {}
        message = first.get("message", "Unknown error")
        return f"FAIL 200 {message}"

    data_keys = list((payload.get("data") or {}).keys())
    return f"PASS 200 data_keys={data_keys}"


def _probe_rest_endpoints(api_key: str) -> List[str]:
    results = []
    file_base = os.getenv("GRID_FILE_DOWNLOAD_URL", "https://api.grid.gg/file-download")
    end_state_path = os.getenv(
        "GRID_END_STATE_PATH", "/end-state/grid/series/{series_id}"
    )
    list_path = os.getenv("GRID_FILE_DOWNLOAD_LIST_PATH", "/list/{series_id}")
    live_base = os.getenv("GRID_LIVE_FEED_URL", "https://api.grid.gg/live")
    live_state_path = os.getenv(
        "GRID_LIVE_SERIES_STATE_PATH", "/series/{series_id}/state"
    )
    live_events_path = os.getenv(
        "GRID_LIVE_SERIES_EVENTS_PATH", "/series/{series_id}/events"
    )
    stats_base = os.getenv("GRID_STATS_FEED_URL", "https://api.grid.gg/stats")
    stats_match_path = os.getenv("GRID_STATS_MATCH_PATH", "/matches/{match_id}/stats")

    series_id = os.getenv("GRID_TEST_SERIES_ID", "").strip()
    match_id = os.getenv("GRID_TEST_MATCH_ID", "").strip()

    def do_get(url: str) -> str:
        response: Optional[requests.Response] = None
        variants = _auth_header_variants(api_key)
        for index, headers in enumerate(variants):
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 401 or index == len(variants) - 1:
                break
        status = response.status_code
        if status in (200, 204):
            return f"PASS {status}"
        return f"FAIL {status} {response.text[:200].strip()}"

    if series_id:
        url = f"{file_base}{list_path.format(series_id=series_id)}"
        results.append(f"file-download list: {do_get(url)}")
        url = f"{file_base}{end_state_path.format(series_id=series_id)}"
        results.append(f"file-download end-state: {do_get(url)}")
        url = f"{live_base}{live_state_path.format(series_id=series_id)}"
        results.append(f"live series-state: {do_get(url)}")
        url = f"{live_base}{live_events_path.format(series_id=series_id)}"
        results.append(f"live series-events: {do_get(url)}")
    else:
        results.append("file-download/live: SKIP (set GRID_TEST_SERIES_ID)")

    if match_id:
        url = f"{stats_base}{stats_match_path.format(match_id=match_id)}"
        results.append(f"stats match: {do_get(url)}")
    else:
        results.append("stats feed: SKIP (set GRID_TEST_MATCH_ID)")

    return results


def main() -> int:
    _load_env()
    parser = argparse.ArgumentParser(description="Probe GRID API access.")
    parser.add_argument("--graphql-only", action="store_true")
    parser.add_argument("--rest-only", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GRID_API_KEY", "").strip()
    if not api_key:
        print("GRID_API_KEY missing in environment.")
        return 1

    targets = _graphql_targets()
    tests = _build_queries()

    if not args.rest_only:
        print("GraphQL probes:")
        for test in tests:
            urls = targets.get(test.target)
            if not urls:
                print(f"- {test.name}: SKIP (no GraphQL URL for {test.target})")
                continue
            for url in urls:
                response, header_used = _post_graphql(
                    url, api_key, test.query, test.variables
                )
                result = _evaluate_graphql(response)
                print(f"- {test.name} [{test.target}] {url} ({header_used}): {result}")

    if not args.graphql_only:
        print("REST probes:")
        for line in _probe_rest_endpoints(api_key):
            print(f"- {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
