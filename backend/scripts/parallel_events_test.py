import argparse
import asyncio
import time
from typing import List, Tuple

from app.env import load_env
from app.settings import Settings
from app.grid_client import GridClient, GridAPIError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test parallel GRID series event fetching."
    )
    parser.add_argument("team_name", help="Team name to fetch matches for")
    parser.add_argument("--match-limit", type=int, default=20)
    parser.add_argument("--max-series", type=int, default=9)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--game-title", default="VALORANT")
    return parser.parse_args()


async def _fetch_series_events(
    client: GridClient, series_id: str, semaphore: asyncio.Semaphore
) -> Tuple[str, float, int, str | None]:
    async with semaphore:
        start = time.perf_counter()
        try:
            events = await asyncio.to_thread(client.fetch_series_events, series_id)
            duration = time.perf_counter() - start
            return series_id, duration, len(events), None
        except GridAPIError as exc:
            duration = time.perf_counter() - start
            return series_id, duration, 0, str(exc)


async def main() -> None:
    args = _parse_args()
    load_env()
    settings = Settings.from_env()

    client = GridClient(
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

    start_total = time.perf_counter()
    matches = client.fetch_team_matches(
        args.team_name,
        limit=args.match_limit,
        game_title=args.game_title,
    )
    series_ids: List[str] = []
    for match in matches:
        series_id = str(match.get("series_id") or "").strip()
        if series_id:
            series_ids.append(series_id)
    unique_series = list(dict.fromkeys(series_ids))[: max(1, args.max_series)]

    print(
        f"Fetched {len(matches)} matches. Testing {len(unique_series)} series with concurrency={args.concurrency}."
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        _fetch_series_events(client, series_id, semaphore)
        for series_id in unique_series
    ]
    results = await asyncio.gather(*tasks)

    successes = 0
    total_events = 0
    for series_id, duration, event_count, error in results:
        status = "OK" if error is None else f"ERROR: {error}"
        print(
            f"series_id={series_id} | {duration:.2f}s | events={event_count} | {status}"
        )
        if error is None:
            successes += 1
            total_events += event_count

    print(
        f"Done. Successes={successes}/{len(results)} | total_events={total_events} | total_time={time.perf_counter() - start_total:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
