from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def normalize_end_state_series(
    payload: Dict[str, Any],
    series_id: str,
    team_name: Optional[str] = None,
) -> List[Dict]:
    if isinstance(payload, list):
        games = payload
    else:
        games = _find_list(
            payload, ("games", "maps", "matches", "seriesGames", "series_games")
        )
    if not games:
        games = [payload]

    matches: List[Dict] = []
    for index, game in enumerate(games):
        match = _normalize_game(game, series_id, index)
        if not match:
            continue
        if team_name and not _match_has_team(match, team_name):
            continue
        matches.append(match)
    return matches


def _normalize_game(game: Any, series_id: str, index: int) -> Optional[Dict]:
    if not isinstance(game, dict):
        return None

    map_name = _extract_map_name(game)
    segments = _extract_segments(game)
    players = _extract_players(game)
    teams = _extract_teams(game)

    match_id = str(game.get("id") or f"{series_id}_{index + 1}")
    return {
        "id": match_id,
        "series_id": series_id,
        "map": {"name": map_name} if map_name else {},
        "segments": segments,
        "players": players,
        "teams": teams,
        "source": "end_state",
    }


def _extract_map_name(game: Dict[str, Any]) -> Optional[str]:
    map_value = _find_first(game, ("mapName", "map_name", "map", "level", "arena"))
    if isinstance(map_value, dict):
        map_value = map_value.get("name") or map_value.get("id")
    if isinstance(map_value, str):
        return map_value
    return None


def _extract_segments(game: Dict[str, Any]) -> List[Dict]:
    rounds = _find_list(game, ("rounds", "segments", "roundResults", "round_results"))
    if not rounds:
        return []

    if any(isinstance(item, dict) and item.get("type") for item in rounds):
        rounds = [
            item
            for item in rounds
            if isinstance(item, dict) and str(item.get("type")).lower() == "round"
        ]

    segments: List[Dict] = []
    for idx, round_data in enumerate(rounds):
        if not isinstance(round_data, dict):
            continue
        winner = _find_first(round_data, ("winner", "winningTeam", "winnerTeam"))
        winner_entry = _normalize_winner(winner)
        end_reason = _find_first(
            round_data, ("endReason", "end_reason", "endType", "end_type")
        )

        if winner_entry is None:
            winner_entry, end_reason = _winner_from_round_teams(
                round_data.get("teams") or [], end_reason
            )
        segment_index = _find_first(
            round_data,
            ("sequenceNumber", "roundNumber", "roundIndex", "segmentIndex", "number"),
        )
        duration = _find_first(
            round_data, ("duration", "roundDuration", "time", "roundTime")
        )
        plant_location = _extract_plant_location(round_data)
        teams = _extract_round_teams(round_data)
        segments.append(
            {
                "segmentIndex": segment_index or (idx + 1),
                "winner": winner_entry,
                "endReason": end_reason,
                "duration": duration,
                "plantLocation": plant_location,
                "teams": teams,
            }
        )
    return segments


def _extract_players(game: Dict[str, Any]) -> List[Dict]:
    players = _find_list(
        game, ("players", "playerStats", "playerStatistics", "scoreboard")
    )
    if players and _looks_like_segment_players(players):
        players = _extract_players_from_teams(game)
    if not players:
        players = _extract_players_from_teams(game)
    if not players:
        return []

    normalized: List[Dict] = []
    for player in players:
        if not isinstance(player, dict):
            continue
        stats = player.get("playerStats") or player.get("stats") or player
        agent_value = player.get("agent") or player.get("character") or player.get("hero")
        assists = (
            stats.get("assists")
            if stats.get("assists") is not None
            else stats.get("killAssistsGiven")
        )
        economy = stats.get("economy")
        if economy is None:
            economy = stats.get("netWorth")
        if economy is None:
            economy = stats.get("money")
        if economy is None:
            economy = stats.get("loadoutValue")
        agent_entry = _normalize_agent(agent_value)
        normalized.append(
            {
                "id": player.get("id"),
                "inGameName": player.get("inGameName")
                or player.get("name")
                or player.get("playerName"),
                "team": _normalize_team_ref(
                    player.get("team") or player.get("teamRef") or player.get("teamId")
                ),
                "agent": agent_entry,
                "playerStats": {
                    "kills": stats.get("kills"),
                    "deaths": stats.get("deaths"),
                    "assists": assists,
                    "acs": stats.get("acs") or stats.get("averageCombatScore"),
                    "economy": economy or stats.get("credits"),
                },
            }
        )
    return normalized


def _extract_teams(game: Dict[str, Any]) -> List[Dict]:
    teams = _find_list(game, ("teams", "teamStats", "teamStatistics"))
    if not teams:
        return []
    normalized = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        normalized.append(
            {
                "id": team.get("id") or team.get("teamId"),
                "name": team.get("name") or team.get("teamName"),
            }
        )
    return normalized


def _normalize_winner(winner: Any) -> Optional[Dict]:
    if isinstance(winner, dict):
        return {"id": winner.get("id") or winner.get("teamId"), "name": winner.get("name")}
    if isinstance(winner, str):
        return {"name": winner}
    if isinstance(winner, int):
        return {"id": winner}
    return None


def _normalize_agent(agent_value: Any) -> Optional[Dict]:
    if isinstance(agent_value, dict):
        role = agent_value.get("role")
        roles = agent_value.get("roles")
        if not role and isinstance(roles, list) and roles:
            role = roles[0]
        return {
            "name": agent_value.get("name") or agent_value.get("id"),
            "role": role,
        }
    if isinstance(agent_value, str):
        return {"name": agent_value}
    return None


def _normalize_team_ref(team_value: Any) -> Optional[Dict]:
    if isinstance(team_value, dict):
        return {"id": team_value.get("id"), "name": team_value.get("name")}
    if isinstance(team_value, str):
        return {"id": team_value}
    return None


def _match_has_team(match: Dict[str, Any], team_name: str) -> bool:
    teams = match.get("teams") or []
    if not teams:
        return True
    for team in teams:
        name = (team.get("name") or "").strip().lower()
        if name == team_name:
            return True
    return False


def _extract_players_from_teams(game: Dict[str, Any]) -> List[Dict]:
    teams = game.get("teams") if isinstance(game.get("teams"), list) else None
    if teams is None:
        teams = _find_list(game, ("teams",)) or []
    if not teams:
        return []
    collected: List[Dict] = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        team_ref = {"id": team.get("id"), "name": team.get("name")}
        for player in team.get("players") or []:
            if not isinstance(player, dict):
                continue
            player = dict(player)
            player["team"] = team_ref
            collected.append(player)
    return collected


def _looks_like_segment_players(players: List[Dict[str, Any]]) -> bool:
    for player in players:
        if not isinstance(player, dict):
            continue
        typename = str(player.get("__typename") or "").lower()
        if "segment" in typename:
            return True
        if "currentarmor" in player or "currenthealth" in player:
            return True
    return False


def _winner_from_round_teams(
    teams: List[Dict[str, Any]], end_reason: Optional[str]
) -> tuple[Optional[Dict], Optional[str]]:
    winner_entry = None
    if teams:
        for team in teams:
            if not isinstance(team, dict):
                continue
            win_type = team.get("winType")
            if win_type:
                winner_entry = {"id": team.get("id"), "name": team.get("name")}
                end_reason = end_reason or win_type
                break
            if team.get("won") is True or team.get("isWinner") is True:
                winner_entry = {"id": team.get("id"), "name": team.get("name")}
                end_reason = end_reason or "won"
                break
        if winner_entry is None and len(teams) == 1:
            winner_entry = {"id": teams[0].get("id"), "name": teams[0].get("name")}
    return winner_entry, end_reason


def _extract_plant_location(round_data: Dict[str, Any]) -> Optional[str]:
    direct = _find_first(round_data, ("plantLocation", "plantSite", "site", "plant_site"))
    if direct:
        return str(direct)
    for team in round_data.get("teams") or []:
        objectives = team.get("objectives") or []
        for obj in objectives:
            if not isinstance(obj, dict):
                continue
            for key in ("site", "siteName", "plantSite", "plantLocation"):
                if obj.get(key):
                    return str(obj.get(key))
    return None


def _extract_round_teams(round_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    teams = round_data.get("teams") or []
    if not isinstance(teams, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        normalized.append(
            {
                "id": team.get("id"),
                "name": team.get("name"),
                "side": team.get("side"),
                "winType": team.get("winType"),
            }
        )
    return normalized


def _find_first(value: Any, keys: Iterable[str], depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return None
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for nested in value.values():
            found = _find_first(nested, keys, depth + 1, max_depth)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first(item, keys, depth + 1, max_depth)
            if found is not None:
                return found
    return None


def _find_list(value: Any, keys: Iterable[str], depth: int = 0, max_depth: int = 5) -> List[Any]:
    if depth > max_depth:
        return []
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, list):
                return found
        for nested in value.values():
            found = _find_list(nested, keys, depth + 1, max_depth)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_list(item, keys, depth + 1, max_depth)
            if found:
                return found
    return []
