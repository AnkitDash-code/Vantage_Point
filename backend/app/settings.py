from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _list_env(name: str) -> List[str]:
    raw = os.getenv(name, "")
    return [value.strip() for value in raw.split(",") if value.strip()]


def _path_env(name: str) -> Optional[Path]:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


@dataclass(frozen=True)
class Settings:
    grid_api_key: str
    debug_mode: bool
    grid_force_live: bool
    grid_api_mode: str
    grid_fallback_to_rest: bool
    grid_graphql_url: str
    grid_central_data_url: str
    grid_stats_feed_url: str
    grid_live_feed_url: str
    grid_series_state_graphql_url: str
    grid_file_download_url: str
    grid_end_state_path: str
    grid_file_download_list_path: str
    grid_file_download_events_path: str
    grid_include_events: bool
    grid_events_max_series: int
    grid_series_page_size: int
    grid_series_max_pages: int
    grid_include_player_team: bool
    grid_graphql_query_path: Optional[Path]
    rag_use_web: bool
    rag_urls: List[str]
    use_precomputed: bool
    precomputed_dir: Optional[Path]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            grid_api_key=os.getenv("GRID_API_KEY", ""),
            debug_mode=_bool_env("DEBUG_MODE"),
            grid_force_live=_bool_env("GRID_FORCE_LIVE"),
            grid_api_mode=os.getenv("GRID_API_MODE", "auto").strip().lower(),
            grid_fallback_to_rest=_bool_env("GRID_FALLBACK_TO_REST", "false"),
            grid_graphql_url=os.getenv(
                "GRID_GRAPHQL_URL", "https://api-op.grid.gg/central-data/graphql"
            ),
            grid_central_data_url=os.getenv(
                "GRID_CENTRAL_DATA_URL", "https://api.grid.gg/central-data"
            ),
            grid_stats_feed_url=os.getenv(
                "GRID_STATS_FEED_URL", "https://api.grid.gg/stats"
            ),
            grid_live_feed_url=os.getenv(
                "GRID_LIVE_FEED_URL", "https://api.grid.gg/live"
            ),
            grid_series_state_graphql_url=os.getenv(
                "GRID_SERIES_STATE_GRAPHQL_URL",
                "https://api-op.grid.gg/live-data-feed/series-state/graphql",
            ),
            grid_file_download_url=os.getenv(
                "GRID_FILE_DOWNLOAD_URL", "https://api.grid.gg/file-download"
            ),
            grid_end_state_path=os.getenv(
                "GRID_END_STATE_PATH", "/end-state/grid/series/{series_id}"
            ),
            grid_file_download_list_path=os.getenv(
                "GRID_FILE_DOWNLOAD_LIST_PATH", "/list/{series_id}"
            ),
            grid_file_download_events_path=os.getenv(
                "GRID_FILE_DOWNLOAD_EVENTS_PATH", "/events/grid/series/{series_id}"
            ),
            grid_include_events=_bool_env("GRID_INCLUDE_EVENTS", "true"),
            grid_events_max_series=int(os.getenv("GRID_EVENTS_MAX_SERIES", "12")),
            grid_series_page_size=int(os.getenv("GRID_SERIES_PAGE_SIZE", "50")),
            grid_series_max_pages=int(os.getenv("GRID_SERIES_MAX_PAGES", "6")),
            grid_include_player_team=_bool_env("GRID_INCLUDE_PLAYER_TEAM", "true"),
            grid_graphql_query_path=_path_env("GRID_GRAPHQL_QUERY_PATH"),
            rag_use_web=_bool_env("RAG_USE_WEB"),
            rag_urls=_list_env("RAG_URLS"),
            use_precomputed=_bool_env("USE_PRECOMPUTED"),
            precomputed_dir=_path_env("PRECOMPUTED_DIR"),
        )
