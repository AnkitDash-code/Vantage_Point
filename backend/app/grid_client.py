from __future__ import annotations

import asyncio
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
import requests


class GridAPIError(RuntimeError):
    pass


class GridConfigError(ValueError):
    pass


def _build_auth_headers(
    api_key: str, header_name: str, token_prefix: Optional[str]
) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if not api_key:
        return headers
    value = f"{token_prefix} {api_key}".strip() if token_prefix else api_key
    headers[header_name] = value
    return headers


def _auth_header_variants(api_key: str) -> List[Dict[str, str]]:
    header_name = os.getenv("GRID_API_AUTH_HEADER", "Authorization").strip()
    token_prefix = os.getenv("GRID_API_TOKEN_PREFIX", "Bearer").strip()
    auth_mode = os.getenv("GRID_API_AUTH_MODE", "auto").strip().lower()

    variants: List[Dict[str, str]] = []
    variants.append(_build_auth_headers(api_key, header_name, token_prefix))

    if auth_mode == "auto":
        if not (
            header_name.lower() == "authorization"
            and token_prefix.lower() == "bearer"
        ):
            variants.append(_build_auth_headers(api_key, "Authorization", "Bearer"))
        if header_name.lower() != "x-api-key":
            variants.append(_build_auth_headers(api_key, "x-api-key", None))

    # De-duplicate by header content.
    unique = []
    seen = set()
    for headers in variants:
        key = tuple(sorted(headers.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(headers)
    return unique


def _safe_path(path: Optional[Path]) -> Optional[Path]:
    if not path:
        return None
    resolved = path.resolve()
    allowed_root = Path(__file__).resolve().parents[2]
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise GridConfigError(
            "GRID_GRAPHQL_QUERY_PATH must be inside the code directory."
        )
    return resolved


@dataclass(frozen=True)
class CentralDataConfig:
    base_url: str
    matches_path: str
    match_path: str
    teams_path: str
    team_param: str
    game_param: str
    limit_param: str
    order_param: str
    order_value: str

    @classmethod
    def from_env(cls) -> "CentralDataConfig":
        return cls(
            base_url=os.getenv("GRID_CENTRAL_DATA_URL", "https://api.grid.gg/central-data"),
            matches_path=os.getenv("GRID_CENTRAL_MATCHES_PATH", "/matches"),
            match_path=os.getenv("GRID_CENTRAL_MATCH_PATH", "/matches/{match_id}"),
            teams_path=os.getenv("GRID_CENTRAL_TEAMS_PATH", "/teams"),
            team_param=os.getenv("GRID_CENTRAL_TEAM_PARAM", "teamName"),
            game_param=os.getenv("GRID_CENTRAL_GAME_PARAM", "gameTitle"),
            limit_param=os.getenv("GRID_CENTRAL_LIMIT_PARAM", "limit"),
            order_param=os.getenv("GRID_CENTRAL_ORDER_PARAM", "orderBy"),
            order_value=os.getenv("GRID_CENTRAL_ORDER_VALUE", "startTime:desc"),
        )


@dataclass(frozen=True)
class StatsFeedConfig:
    base_url: str
    match_stats_path: str

    @classmethod
    def from_env(cls) -> "StatsFeedConfig":
        return cls(
            base_url=os.getenv("GRID_STATS_FEED_URL", "https://api.grid.gg/stats"),
            match_stats_path=os.getenv(
                "GRID_STATS_MATCH_PATH", "/matches/{match_id}/stats"
            ),
        )


@dataclass(frozen=True)
class LiveFeedConfig:
    base_url: str
    series_state_path: str
    series_events_path: str

    @classmethod
    def from_env(cls) -> "LiveFeedConfig":
        return cls(
            base_url=os.getenv("GRID_LIVE_FEED_URL", "https://api.grid.gg/live"),
            series_state_path=os.getenv(
                "GRID_LIVE_SERIES_STATE_PATH", "/series/{series_id}/state"
            ),
            series_events_path=os.getenv(
                "GRID_LIVE_SERIES_EVENTS_PATH", "/series/{series_id}/events"
            ),
        )


@dataclass(frozen=True)
class FileDownloadConfig:
    base_url: str
    list_path: str
    end_state_path: str
    events_path: str

    @classmethod
    def from_env(cls) -> "FileDownloadConfig":
        return cls(
            base_url=os.getenv("GRID_FILE_DOWNLOAD_URL", "https://api.grid.gg/file-download"),
            list_path=os.getenv("GRID_FILE_DOWNLOAD_LIST_PATH", "/list/{series_id}"),
            end_state_path=os.getenv(
                "GRID_END_STATE_PATH", "/end-state/grid/series/{series_id}"
            ),
            events_path=os.getenv(
                "GRID_FILE_DOWNLOAD_EVENTS_PATH", "/events/grid/series/{series_id}"
            ),
        )


class GridRestClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = self._request_with_auth("GET", url, params=params)
        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise GridAPIError(f"GRID API error: {response.status_code} - {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc

    def _request_with_auth(
        self, method: str, url: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        variants = _auth_header_variants(self.api_key)
        response: Optional[requests.Response] = None
        for index, headers in enumerate(variants):
            response = self.session.request(
                method, url, headers=headers, params=params, timeout=30
            )
            if response.status_code not in (401, 403) or index == len(variants) - 1:
                break
        return response


class CentralDataFeedClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[CentralDataConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or CentralDataConfig.from_env()
        self.client = GridRestClient(api_key, self.config.base_url, session=session)

    def fetch_matches(
        self, team_name: str, limit: int = 50, game_title: str = "VALORANT"
    ) -> List[Dict]:
        params = {
            self.config.team_param: team_name,
            self.config.game_param: game_title,
            self.config.limit_param: limit,
        }
        if self.config.order_param and self.config.order_value:
            params[self.config.order_param] = self.config.order_value

        payload = self.client.get(self.config.matches_path, params=params)
        return self._extract_list(payload, ("matches", "items", "data"))

    def fetch_match(self, match_id: str) -> Dict:
        path = self.config.match_path.format(match_id=match_id)
        return self.client.get(path)

    def fetch_teams(self) -> List[Dict]:
        payload = self.client.get(self.config.teams_path)
        return self._extract_list(payload, ("teams", "items", "data"))

    @staticmethod
    def _extract_list(payload: Any, keys: tuple[str, ...]) -> List[Dict]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested in keys + ("items", "data"):
                    nested_value = value.get(nested)
                    if isinstance(nested_value, list):
                        return nested_value
        return []


class StatsFeedClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[StatsFeedConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or StatsFeedConfig.from_env()
        self.client = GridRestClient(api_key, self.config.base_url, session=session)

    def fetch_match_stats(self, match_id: str) -> Dict:
        path = self.config.match_stats_path.format(match_id=match_id)
        return self.client.get(path)


class LiveDataFeedClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[LiveFeedConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or LiveFeedConfig.from_env()
        self.client = GridRestClient(api_key, self.config.base_url, session=session)

    def fetch_series_state(self, series_id: str) -> Dict:
        path = self.config.series_state_path.format(series_id=series_id)
        return self.client.get(path)

    def fetch_series_events(self, series_id: str, since: Optional[str] = None) -> Dict:
        path = self.config.series_events_path.format(series_id=series_id)
        params = {"since": since} if since else None
        return self.client.get(path, params=params)


class FileDownloadClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[FileDownloadConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config or FileDownloadConfig.from_env()
        self.client = GridRestClient(api_key, self.config.base_url, session=session)

    def list_files(self, series_id: str) -> List[Dict]:
        path = self.config.list_path.format(series_id=series_id)
        payload = self.client.get(path)
        if isinstance(payload, dict):
            files = payload.get("files")
            if isinstance(files, list):
                return files
        return []

    def fetch_series_end_state(self, series_id: str) -> Dict:
        files = self.list_files(series_id)
        file_entry = _select_state_file(files)
        if file_entry:
            full_url = file_entry.get("fullURL") or file_entry.get("fullUrl")
            if full_url:
                return self._download_full_url(full_url)

        path = self.config.end_state_path.format(series_id=series_id)
        return self.client.get(path)

    def fetch_series_events(self, series_id: str) -> List[Dict]:
        files = self.list_files(series_id)
        file_entry = _select_events_file(files)
        if file_entry:
            full_url = file_entry.get("fullURL") or file_entry.get("fullUrl")
            if full_url:
                return self._download_events_url(full_url)

        path = self.config.events_path.format(series_id=series_id)
        url = f"{self.config.base_url.rstrip('/')}{path}"
        return self._download_events_url(url)

    def _download_full_url(self, url: str) -> Dict:
        variants = _auth_header_variants(self.client.api_key)
        response: Optional[requests.Response] = None
        for index, headers in enumerate(variants):
            response = self.client.session.get(url, headers=headers, timeout=60)
            if response.status_code not in (401, 403) or index == len(variants) - 1:
                break
        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise GridAPIError(f"GRID API error: {response.status_code} - {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc

    def _download_events_url(self, url: str) -> List[Dict]:
        variants = _auth_header_variants(self.client.api_key)
        response: Optional[requests.Response] = None
        for index, headers in enumerate(variants):
            response = self.client.session.get(url, headers=headers, timeout=60)
            if response.status_code not in (401, 403) or index == len(variants) - 1:
                break
        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise GridAPIError(f"GRID API error: {response.status_code} - {detail}")

        content_type = (response.headers.get("Content-Type") or "").lower()
        content = response.content
        if content.startswith(b"PK") or "zip" in content_type:
            return _read_events_zip(content)
        if content.startswith(b"{") or content.startswith(b"["):
            try:
                return json.loads(content.decode("utf-8"))
            except ValueError as exc:
                raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc
        return _read_events_jsonl(content)


class GridClient:
    """Client facade for scouting, using GraphQL or Central Data Feed."""

    def __init__(
        self,
        api_key: str,
        debug_mode: bool = False,
        force_live: bool = False,
        api_mode: str = "graphql",
        fallback_to_rest: bool = False,
        graphql_url: Optional[str] = None,
        central_data_url: Optional[str] = None,
        file_download_url: Optional[str] = None,
        end_state_path: Optional[str] = None,
        file_download_list_path: Optional[str] = None,
        file_download_events_path: Optional[str] = None,
        series_page_size: int = 50,
        series_max_pages: int = 6,
        cache_dir: Optional[Path] = None,
        include_player_team: Optional[bool] = None,
        graphql_query_path: Optional[Path] = None,
    ) -> None:
        if force_live and not api_key:
            raise ValueError("GRID_API_KEY is required when force_live is True.")

        if not api_key and not debug_mode:
            raise ValueError("GRID_API_KEY is required when debug_mode is False.")

        self.api_key = api_key
        self.debug_mode = debug_mode
        self.force_live = force_live
        self.api_mode = api_mode
        self.fallback_to_rest = fallback_to_rest
        self.graphql_url = graphql_url or os.getenv(
            "GRID_GRAPHQL_URL", "https://api-op.grid.gg/central-data/graphql"
        )
        self.series_page_size = series_page_size
        self.series_max_pages = series_max_pages
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[1] / "data" / "debug_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.include_player_team = (
            include_player_team
            if include_player_team is not None
            else os.getenv("GRID_INCLUDE_PLAYER_TEAM", "true").lower() == "true"
        )
        self.graphql_query_path = _safe_path(graphql_query_path)

        if central_data_url:
            os.environ.setdefault("GRID_CENTRAL_DATA_URL", central_data_url)
        self.central_client = CentralDataFeedClient(api_key, session=self.session)
        if file_download_url:
            os.environ.setdefault("GRID_FILE_DOWNLOAD_URL", file_download_url)
        if end_state_path:
            os.environ.setdefault("GRID_END_STATE_PATH", end_state_path)
        if file_download_list_path:
            os.environ.setdefault("GRID_FILE_DOWNLOAD_LIST_PATH", file_download_list_path)
        if file_download_events_path:
            os.environ.setdefault(
                "GRID_FILE_DOWNLOAD_EVENTS_PATH", file_download_events_path
            )
        self.file_download_client = FileDownloadClient(api_key, session=self.session)

    def fetch_team_matches(
        self, team_name: str, limit: int = 50, game_title: str = "VALORANT"
    ) -> List[Dict]:
        if not self.force_live:
            cached = self._load_from_cache(team_name)
            if cached is not None:
                return cached

            if self.debug_mode:
                raise FileNotFoundError(
                    f"No cache found for {team_name}. Run seed_match_data.py first."
                )

        max_attempts = 3
        base_delay = 2
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                if self.api_mode in ("graphql", "auto"):
                    try:
                        matches = self._fetch_team_matches_graphql(
                            team_name, limit, game_title
                        )
                    except GridAPIError as exc:
                        detail = str(exc).lower()
                        timeout_error = "timeout" in detail or "timed out" in detail
                        auth_error = (
                            "401" in detail
                            or "403" in detail
                            or "unauthorized" in detail
                            or "forbidden" in detail
                        )
                        if auth_error:
                            matches = self._fetch_team_matches_rest(
                                team_name, limit, game_title, exc
                            )
                        else:
                            if not self.fallback_to_rest and not timeout_error:
                                raise
                            matches = self._fetch_team_matches_rest(
                                team_name, limit, game_title, exc
                            )
                elif self.api_mode in ("central-data", "rest"):
                    matches = self.central_client.fetch_matches(
                        team_name, limit, game_title
                    )
                else:
                    raise GridConfigError(
                        f"Unsupported GRID_API_MODE: {self.api_mode}"
                    )

                self._save_to_cache(team_name, matches)
                return matches
            except GridAPIError as exc:
                detail = str(exc).lower()
                retryable = (
                    "429" in detail
                    or "too many requests" in detail
                    or "timeout" in detail
                    or "timed out" in detail
                )
                last_error = exc
                if attempt < max_attempts - 1 and retryable:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise

        if last_error:
            raise last_error

        raise GridAPIError("Unable to fetch matches after retries.")

    def fetch_series_events(self, series_id: str) -> List[Dict]:
        if self.debug_mode and not self.force_live:
            cached = self._load_events_from_cache(series_id)
            if cached is not None:
                return cached
        events = self.file_download_client.fetch_series_events(series_id)
        if self.debug_mode:
            self._save_events_to_cache(series_id, events)
        return events

    def _fetch_team_matches_rest(
        self,
        team_name: str,
        limit: int,
        game_title: str,
        graphql_error: GridAPIError,
    ) -> List[Dict]:
        detail = str(graphql_error)
        if "Cannot query field" in detail or "GRAPHQL_VALIDATION_FAILED" in detail:
            mode_note = "GraphQL schema mismatch detected; falling back to REST feed."
        else:
            mode_note = "GraphQL request failed; falling back to REST feed."
        print(mode_note)
        return self.central_client.fetch_matches(team_name, limit, game_title)

    def _fetch_team_matches_graphql(
        self, team_name: str, limit: int, game_title: str
    ) -> List[Dict]:
        from app.end_state_adapter import normalize_end_state_series

        series_nodes = self._fetch_series_nodes_graphql(
            limit, game_title, team_name=team_name
        )
        matches: List[Dict] = []
        team_name_lower = team_name.strip().lower()

        for node in series_nodes:
            series_id = str(node.get("id") or "").strip()
            if not series_id:
                continue
            try:
                end_state = self.file_download_client.fetch_series_end_state(series_id)
            except GridAPIError as exc:
                detail = str(exc)
                if "404" in detail or "Not Found" in detail:
                    continue
                raise
            series_matches = normalize_end_state_series(
                end_state, series_id=series_id, team_name=team_name_lower
            )
            matches.extend(series_matches)
            if len(matches) >= limit:
                break

        return matches[:limit]

    def _load_graphql_query(self) -> Optional[str]:
        if not self.graphql_query_path:
            default_path = (
                Path(__file__).resolve().parent / "graphql" / "all_series.graphql"
            )
            if default_path.exists():
                self.graphql_query_path = _safe_path(default_path)
        if not self.graphql_query_path or not self.graphql_query_path.exists():
            return None
        return self.graphql_query_path.read_text(encoding="utf-8")

    def _fetch_series_nodes_graphql(
        self, limit: int, game_title: str, team_name: Optional[str] = None
    ) -> List[Dict]:
        query = self._load_graphql_query()
        if not query:
            raise GridConfigError("GraphQL query template not found.")

        title_id = _title_id_for_game(game_title)
        page_size = max(1, min(50, self.series_page_size))
        after: Optional[str] = None
        nodes: List[Dict] = []
        max_pages = max(1, self.series_max_pages)
        team_name_lower = team_name.strip().lower() if team_name else None

        for _ in range(max_pages):
            variables = {"first": page_size, "after": after, "titleId": title_id}
            data = self._graphql_request(query, variables)
            payload = data.get("data", {}).get("allSeries", {})
            edges = payload.get("edges", []) or []
            for edge in edges:
                node = edge.get("node") or {}
                if node:
                    if team_name_lower and not _series_has_team(node, team_name_lower):
                        continue
                    nodes.append(node)

            page_info = payload.get("pageInfo", {}) or {}
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            if not after:
                break

        return nodes

    def _build_team_matches_query(self) -> str:
        player_team_field = "team { id name }" if self.include_player_team else ""
        return f"""
        query TeamMatches($teamName: String!, $limit: Int!, $gameTitle: String!) {{
            matches(
                filter: {{
                    teams: {{ name: {{ eq: $teamName }} }}
                    game: {{ title: {{ eq: $gameTitle }} }}
                }}
                limit: $limit
                orderBy: {{ field: START_TIME, direction: DESC }}
            ) {{
                id
                startTime
                teams {{
                    id
                    name
                }}
                map {{
                    name
                }}
                segments {{
                    segmentIndex
                    winner {{ id name }}
                    endReason
                    duration
                    plantLocation
                }}
                players {{
                    id
                    inGameName
                    {player_team_field}
                    agent {{ name role }}
                    playerStats {{
                        kills
                        deaths
                        assists
                        acs
                        economy
                    }}
                }}
            }}
        }}
        """

    def _graphql_request(self, query: str, variables: Dict) -> Dict:
        variants = _auth_header_variants(self.api_key)
        last_error: Optional[str] = None
        timeout = float(os.getenv("GRID_GRAPHQL_TIMEOUT", "30"))
        for index, headers in enumerate(variants):
            response = None
            for attempt in range(2):
                try:
                    response = self.session.post(
                        self.graphql_url,
                        headers=headers,
                        json={"query": query, "variables": variables},
                        timeout=timeout,
                    )
                    break
                except requests.Timeout:
                    last_error = f"GRID API error: timeout after {timeout}s"
                    if attempt == 0:
                        continue
                    response = None
                except requests.RequestException as exc:
                    last_error = f"GRID API error: {exc}"
                    response = None
                    break
            if response is None:
                if index < len(variants) - 1:
                    continue
                raise GridAPIError(last_error or "GRID API error: No response received.")
            if response.status_code != 200:
                detail = _extract_error_detail(response)
                last_error = f"GRID API error: {response.status_code} - {detail}"
                if response.status_code in (401, 403) and index < len(variants) - 1:
                    continue
                raise GridAPIError(last_error)

            try:
                payload = response.json()
            except ValueError as exc:
                raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc

            if payload.get("errors"):
                if _is_auth_error(payload["errors"]) and index < len(variants) - 1:
                    last_error = f"GRID API error: {payload['errors']}"
                    continue
                raise GridAPIError(f"GRID API error: {payload['errors']}")

            return payload

        raise GridAPIError(last_error or "GRID API error: No response received.")

    def _load_from_cache(self, team_name: str) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"{self._slugify(team_name)}_matches.json"
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_to_cache(self, team_name: str, matches: List[Dict]) -> None:
        cache_file = self.cache_dir / f"{self._slugify(team_name)}_matches.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(matches, handle, indent=2)

    def _load_events_from_cache(self, series_id: str) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"events_{series_id}.json"
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_events_to_cache(self, series_id: str, events: List[Dict]) -> None:
        cache_file = self.cache_dir / f"events_{series_id}.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(events, handle)

    def _load_metrics_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load pre-computed metrics from cache."""
        cache_file = self.cache_dir / f"metrics_{self._slugify(cache_key)}.json"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_metrics_to_cache(self, cache_key: str, metrics: Dict) -> None:
        """Save computed metrics to cache."""
        cache_file = self.cache_dir / f"metrics_{self._slugify(cache_key)}.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        return cleaned.strip("_")


def _extract_error_detail(response: requests.Response) -> str:
    text = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        return text or "No response body"

    if isinstance(payload, dict):
        if "errors" in payload:
            return json.dumps(payload["errors"])
        if "error" in payload:
            return str(payload["error"])
        if "message" in payload:
            return str(payload["message"])
        if "error_message" in payload:
            return str(payload["error_message"])
        return json.dumps(payload)

    return text or "No response body"


def _is_auth_error(errors: Any) -> bool:
    messages: List[str] = []
    if isinstance(errors, list):
        for item in errors:
            if isinstance(item, dict):
                message = item.get("message")
                if message:
                    messages.append(str(message))
            else:
                messages.append(str(item))
    elif isinstance(errors, dict):
        message = errors.get("message")
        if message:
            messages.append(str(message))
    else:
        messages.append(str(errors))

    for message in messages:
        lowered = message.lower()
        if "unauthorized" in lowered or "forbidden" in lowered:
            return True
    return False


def _title_id_for_game(game_title: str) -> str:
    mapping = {
        "VALORANT": "6",
        "LEAGUE OF LEGENDS": "3",
        "LOL": "3",
        "RAINBOW 6: SIEGE": "25",
        "R6": "25",
    }
    key = game_title.strip().upper()
    return mapping.get(key, "6")


def _series_has_team(series_node: Dict[str, Any], team_name_lower: str) -> bool:
    teams = series_node.get("teams") or []
    if not teams:
        return True
    for team in teams:
        if not isinstance(team, dict):
            continue
        base_info = team.get("baseInfo") or {}
        name = (base_info.get("name") or team.get("name") or "").strip().lower()
        if name == team_name_lower:
            return True
    return False


def _select_state_file(files: List[Dict]) -> Optional[Dict]:
    if not files:
        return None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        file_id = str(entry.get("id") or "").lower()
        description = str(entry.get("description") or "").lower()
        if "state" in file_id or "state" in description:
            return entry
    return files[0]


def _select_events_file(files: List[Dict]) -> Optional[Dict]:
    if not files:
        return None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        file_id = str(entry.get("id") or "").lower()
        description = str(entry.get("description") or "").lower()
        if "event" in file_id or "event" in description:
            return entry
    return None


def _read_events_zip(content: bytes) -> List[Dict]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            if not names:
                return []
            with archive.open(names[0]) as handle:
                return _read_events_jsonl(handle.read())
    except (zipfile.BadZipFile, OSError) as exc:
        raise GridAPIError(f"Failed to read events zip: {exc}") from exc


def _read_events_jsonl(content: bytes) -> List[Dict]:
    events: List[Dict] = []
    for line in content.splitlines():
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


class AsyncGridRestClient:
    """Async version of GridRestClient using aiohttp."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session
        self._own_session = session is None

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = await self._request_with_auth("GET", url, params=params)
        if response.status >= 400:
            detail = await self._extract_error_detail(response)
            raise GridAPIError(f"GRID API error: {response.status} - {detail}")
        try:
            return await response.json()
        except ValueError as exc:
            raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc

    async def _request_with_auth(
        self, method: str, url: str, params: Optional[Dict[str, Any]] = None
    ) -> aiohttp.ClientResponse:
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        variants = _auth_header_variants(self.api_key)
        response: Optional[aiohttp.ClientResponse] = None
        for index, headers in enumerate(variants):
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                response = await self.session.request(
                    method, url, headers=headers, params=params, timeout=timeout
                )
                if response.status not in (401, 403) or index == len(variants) - 1:
                    break
            except asyncio.TimeoutError:
                if index < len(variants) - 1:
                    continue
                raise GridAPIError("GRID API error: timeout after 30s")
            except aiohttp.ClientError as exc:
                if index < len(variants) - 1:
                    continue
                raise GridAPIError(f"GRID API error: {exc}")
        
        if response is None:
            raise GridAPIError("GRID API error: No response received.")
        return response

    async def _extract_error_detail(self, response: aiohttp.ClientResponse) -> str:
        text = await response.text()
        text = text.strip()
        try:
            payload = await response.json()
        except ValueError:
            return text or "No response body"

        if isinstance(payload, dict):
            if "errors" in payload:
                return json.dumps(payload["errors"])
            if "error" in payload:
                return str(payload["error"])
            if "message" in payload:
                return str(payload["message"])
            if "error_message" in payload:
                return str(payload["error_message"])
            return json.dumps(payload)

        return text or "No response body"

    async def close(self) -> None:
        if self._own_session and self.session:
            await self.session.close()


class AsyncCentralDataFeedClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[CentralDataConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.config = config or CentralDataConfig.from_env()
        self.client = AsyncGridRestClient(api_key, self.config.base_url, session=session)

    async def fetch_matches(
        self, team_name: str, limit: int = 50, game_title: str = "VALORANT"
    ) -> List[Dict]:
        params = {
            self.config.team_param: team_name,
            self.config.game_param: game_title,
            self.config.limit_param: limit,
        }
        if self.config.order_param and self.config.order_value:
            params[self.config.order_param] = self.config.order_value

        payload = await self.client.get(self.config.matches_path, params=params)
        return self._extract_list(payload, ("matches", "items", "data"))

    @staticmethod
    def _extract_list(payload: Any, keys: tuple[str, ...]) -> List[Dict]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested in keys + ("items", "data"):
                    nested_value = value.get(nested)
                    if isinstance(nested_value, list):
                        return nested_value
        return []


class AsyncFileDownloadClient:
    def __init__(
        self,
        api_key: str,
        config: Optional[FileDownloadConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.config = config or FileDownloadConfig.from_env()
        self.client = AsyncGridRestClient(api_key, self.config.base_url, session=session)

    async def fetch_series_events(self, series_id: str) -> List[Dict]:
        files = await self.list_files(series_id)
        file_entry = _select_events_file(files)
        if file_entry:
            full_url = file_entry.get("fullURL") or file_entry.get("fullUrl")
            if full_url:
                return await self._download_events_url(full_url)

        path = self.config.events_path.format(series_id=series_id)
        url = f"{self.config.base_url.rstrip('/')}{path}"
        return await self._download_events_url(url)

    async def list_files(self, series_id: str) -> List[Dict]:
        path = self.config.list_path.format(series_id=series_id)
        payload = await self.client.get(path)
        if isinstance(payload, dict):
            files = payload.get("files")
            if isinstance(files, list):
                return files
        return []

    async def _download_events_url(self, url: str) -> List[Dict]:
        if self.client.session is None:
            self.client.session = aiohttp.ClientSession()
            
        variants = _auth_header_variants(self.client.api_key)
        response: Optional[aiohttp.ClientResponse] = None
        for index, headers in enumerate(variants):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                response = await self.client.session.get(url, headers=headers, timeout=timeout)
                if response.status not in (401, 403) or index == len(variants) - 1:
                    break
            except asyncio.TimeoutError:
                if index < len(variants) - 1:
                    continue
                raise GridAPIError("GRID API error: timeout after 60s")
            except aiohttp.ClientError as exc:
                if index < len(variants) - 1:
                    continue
                raise GridAPIError(f"GRID API error: {exc}")
        
        if response is None:
            raise GridAPIError("GRID API error: No response received.")
            
        if response.status >= 400:
            detail = await self.client._extract_error_detail(response)
            raise GridAPIError(f"GRID API error: {response.status} - {detail}")

        content = await response.read()
        content_type = (response.headers.get("Content-Type") or "").lower()
        
        if content.startswith(b"PK") or "zip" in content_type:
            return _read_events_zip(content)
        if content.startswith(b"{") or content.startswith(b"["):
            try:
                return json.loads(content.decode("utf-8"))
            except ValueError as exc:
                raise GridAPIError(f"Non-JSON response from GRID API: {exc}") from exc
        return _read_events_jsonl(content)


class AsyncGridClient:
    """Async version of GridClient for non-blocking operations."""
    
    def __init__(
        self,
        api_key: str,
        debug_mode: bool = False,
        force_live: bool = False,
        api_mode: str = "graphql",
        fallback_to_rest: bool = False,
        central_data_url: Optional[str] = None,
        file_download_url: Optional[str] = None,
        series_page_size: int = 50,
        series_max_pages: int = 6,
        cache_dir: Optional[Path] = None,
        include_player_team: Optional[bool] = None,
    ) -> None:
        if force_live and not api_key:
            raise ValueError("GRID_API_KEY is required when force_live is True.")

        if not api_key and not debug_mode:
            raise ValueError("GRID_API_KEY is required when debug_mode is False.")

        self.api_key = api_key
        self.debug_mode = debug_mode
        self.force_live = force_live
        self.api_mode = api_mode
        self.fallback_to_rest = fallback_to_rest
        self.series_page_size = series_page_size
        self.series_max_pages = series_max_pages
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[1] / "data" / "debug_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.include_player_team = (
            include_player_team
            if include_player_team is not None
            else os.getenv("GRID_INCLUDE_PLAYER_TEAM", "true").lower() == "true"
        )

        if central_data_url:
            os.environ.setdefault("GRID_CENTRAL_DATA_URL", central_data_url)
        if file_download_url:
            os.environ.setdefault("GRID_FILE_DOWNLOAD_URL", file_download_url)

    async def fetch_team_matches(
        self, team_name: str, limit: int = 50, game_title: str = "VALORANT"
    ) -> List[Dict]:
        if not self.force_live:
            cached = self._load_from_cache(team_name)
            if cached is not None:
                return cached

            if self.debug_mode:
                raise FileNotFoundError(
                    f"No cache found for {team_name}. Run seed_match_data.py first."
                )

        # For now, use the REST client since we don't have async GraphQL
        async with aiohttp.ClientSession() as session:
            central_client = AsyncCentralDataFeedClient(self.api_key, session=session)
            matches = await central_client.fetch_matches(team_name, limit, game_title)

        self._save_to_cache(team_name, matches)
        return matches

    async def fetch_series_events(self, series_id: str) -> List[Dict]:
        if self.debug_mode and not self.force_live:
            cached = self._load_events_from_cache(series_id)
            if cached is not None:
                return cached
        
        async with aiohttp.ClientSession() as session:
            file_download_client = AsyncFileDownloadClient(self.api_key, session=session)
            events = await file_download_client.fetch_series_events(series_id)
        
        if self.debug_mode:
            self._save_events_to_cache(series_id, events)
        return events

    def _load_from_cache(self, team_name: str) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"{self._slugify(team_name)}_matches.json"
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_to_cache(self, team_name: str, matches: List[Dict]) -> None:
        cache_file = self.cache_dir / f"{self._slugify(team_name)}_matches.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(matches, handle, indent=2)

    def _load_events_from_cache(self, series_id: str) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"events_{series_id}.json"
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_events_to_cache(self, series_id: str, events: List[Dict]) -> None:
        cache_file = self.cache_dir / f"events_{series_id}.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(events, handle)

    def _load_metrics_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load pre-computed metrics from cache."""
        cache_file = self.cache_dir / f"metrics_{self._slugify(cache_key)}.json"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_metrics_to_cache(self, cache_key: str, metrics: Dict) -> None:
        """Save computed metrics to cache."""
        cache_file = self.cache_dir / f"metrics_{self._slugify(cache_key)}.json"
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        return cleaned.strip("_")
