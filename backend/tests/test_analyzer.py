from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.analyzer import ScoutingAnalyzer


def _build_player(
    player_id: str,
    name: str,
    team_id: str,
    team_name: str,
    agent_name: str,
    role: str,
    economy: Optional[int],
) -> dict:
    return {
        "id": player_id,
        "inGameName": name,
        "team": {"id": team_id, "name": team_name},
        "agent": {"name": agent_name, "role": role},
        "playerStats": {
            "kills": 10,
            "deaths": 5,
            "assists": 2,
            "acs": 200,
            "economy": economy,
        },
    }


def _build_match(
    match_id: str,
    map_name: str,
    segments: Optional[List[Dict]] = None,
    players: Optional[List[Dict]] = None,
    include_teams: bool = True,
) -> dict:
    match = {
        "id": match_id,
        "map": {"name": map_name},
        "segments": segments or [],
        "players": players or [],
    }
    if include_teams:
        match["teams"] = [
            {"id": "t1", "name": "Cloud9"},
            {"id": "t2", "name": "Sentinels"},
        ]
    return match


def test_win_rate_ignores_missing_winner_and_filters_map() -> None:
    segments = [
        {"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 30},
        {"segmentIndex": 2, "winner": {"id": "t2"}, "duration": 55},
        {"segmentIndex": 3, "winner": None, "duration": 60},
        {"segmentIndex": 4, "winner": {"id": "t1"}, "duration": 45},
    ]
    match_ascent = _build_match("m1", "Ascent", segments=segments)
    match_bind = _build_match(
        "m2",
        "Bind",
        segments=[{"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 40}],
    )

    analyzer = ScoutingAnalyzer([match_ascent, match_bind], team_name="Cloud9")
    assert analyzer.get_win_rate() == 75.0
    assert analyzer.get_win_rate(map_name="Bind") == 100.0


def test_aggression_index_uses_valid_durations() -> None:
    segments = [
        {"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 30},
        {"segmentIndex": 2, "winner": {"id": "t1"}, "duration": 70},
        {"segmentIndex": 3, "winner": {"id": "t2"}, "duration": None},
    ]
    match = _build_match("m1", "Ascent", segments=segments)
    analyzer = ScoutingAnalyzer([match], team_name="Cloud9")

    aggression = analyzer.get_aggression_index()
    assert aggression["style"] == "Default"
    assert aggression["avg_duration"] == 50.0
    assert aggression["rush_rate"] == 50.0


def test_agent_and_role_distribution_target_only() -> None:
    players = [
        _build_player("p1", "Alpha", "t1", "Cloud9", "Jett", "Duelist", 3000),
        _build_player("p2", "Bravo", "t1", "Cloud9", "Omen", "Controller", 2800),
        _build_player("p3", "Charlie", "t2", "Sentinels", "Sova", "Initiator", 3100),
        _build_player("p4", "Delta", "t2", "Sentinels", "Sage", "Sentinel", 3100),
    ]
    match = _build_match("m1", "Ascent", players=players)
    analyzer = ScoutingAnalyzer([match], team_name="Cloud9")

    composition = analyzer.get_agent_composition()
    agent_rates = {entry["agent"]: entry["pick_rate"] for entry in composition}
    assert agent_rates == {"Jett": 50.0, "Omen": 50.0}

    roles = analyzer.get_role_distribution()
    assert roles == {"Duelist": 50.0, "Controller": 50.0}


def test_economy_distribution_uses_median_and_filters_invalid() -> None:
    players_match_1 = [
        _build_player("p1", "Alpha", "t1", "Cloud9", "Jett", "Duelist", 1500),
        _build_player("p2", "Bravo", "t1", "Cloud9", "Omen", "Controller", 2500),
        _build_player("p3", "Other", "t2", "Sentinels", "Sova", "Initiator", 3000),
    ]
    players_match_2 = [
        _build_player("p4", "Echo", "t1", "Cloud9", "Raze", "Duelist", 4200),
        _build_player("p5", "Foxtrot", "t1", "Cloud9", "Viper", "Controller", 4000),
        _build_player("p6", "Other2", "t2", "Sentinels", "Sage", "Sentinel", None),
    ]
    players_match_3 = [
        _build_player("p7", "Ghost", "t1", "Cloud9", "Sova", "Initiator", -50),
    ]

    match_1 = _build_match("m1", "Ascent", players=players_match_1)
    match_2 = _build_match("m2", "Bind", players=players_match_2)
    match_3 = _build_match("m3", "Haven", players=players_match_3)

    analyzer = ScoutingAnalyzer([match_1, match_2, match_3], team_name="Cloud9")
    economy = analyzer.get_economy_distribution()
    assert economy == {"eco": 0.0, "force": 50.0, "full": 50.0}


def test_team_id_falls_back_to_player_team() -> None:
    segments = [{"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 35}]
    players = [
        _build_player("p1", "Alpha", "t1", "Cloud9", "Jett", "Duelist", 3000),
        _build_player("p2", "Bravo", "t2", "Sentinels", "Sova", "Initiator", 3000),
    ]
    match = _build_match(
        "m1", "Ascent", segments=segments, players=players, include_teams=False
    )
    analyzer = ScoutingAnalyzer([match], team_name="Cloud9")
    assert analyzer.get_win_rate() == 100.0


def test_site_preferences_prefers_event_data(monkeypatch) -> None:
    segments = [
        {"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 30, "plantLocation": "A"},
        {"segmentIndex": 2, "winner": {"id": "t1"}, "duration": 30, "plantLocation": "B"},
    ]
    match = _build_match("m1", "Ascent", segments=segments)
    analyzer = ScoutingAnalyzer([match], team_name="Cloud9")

    monkeypatch.setattr(
        analyzer, "_site_preferences_from_events", lambda: {"A": 60.0, "B": 40.0}
    )
    assert analyzer.get_site_preferences() == {"A": 60.0, "B": 40.0}


def test_pistol_site_preferences_filters_rounds() -> None:
    segments = [
        {"segmentIndex": 1, "winner": {"id": "t1"}, "duration": 30, "plantLocation": "A"},
        {"segmentIndex": 2, "winner": {"id": "t1"}, "duration": 30, "plantLocation": "B"},
        {"segmentIndex": 13, "winner": {"id": "t2"}, "duration": 50, "plantLocation": "B"},
    ]
    match = _build_match("m1", "Ascent", segments=segments)
    analyzer = ScoutingAnalyzer([match], team_name="Cloud9")

    assert analyzer.get_pistol_site_preferences() == {"A": 50.0, "B": 50.0}


def test_player_tendencies_reports_top_agent_and_kd() -> None:
    players_match_1 = [
        _build_player("p1", "Alpha", "t1", "Cloud9", "Jett", "Duelist", 3000),
        _build_player("p2", "Bravo", "t1", "Cloud9", "Omen", "Controller", 2800),
    ]
    players_match_2 = [
        _build_player("p1", "Alpha", "t1", "Cloud9", "Jett", "Duelist", 3200),
        _build_player("p2", "Bravo", "t1", "Cloud9", "Omen", "Controller", 2900),
    ]
    players_match_2[0]["playerStats"].update({"kills": 12, "deaths": 4, "assists": 3, "acs": 210})
    players_match_2[1]["playerStats"].update({"kills": 8, "deaths": 6, "assists": 4, "acs": 190})

    match_1 = _build_match("m1", "Ascent", players=players_match_1)
    match_2 = _build_match("m2", "Bind", players=players_match_2)
    analyzer = ScoutingAnalyzer([match_1, match_2], team_name="Cloud9")

    tendencies = analyzer.get_player_tendencies()
    alpha = next(entry for entry in tendencies if entry["player"] == "Alpha")
    assert alpha["matches_played"] == 2
    assert alpha["top_agent"] == "Jett"
    assert alpha["top_agent_rate"] == 100.0
    assert alpha["avg_kills"] == 11.0
    assert alpha["avg_deaths"] == 4.5
    assert alpha["kd_ratio"] == 2.44


def test_recent_compositions_summarizes_by_map() -> None:
    comp_a = ["Jett", "Raze", "Omen", "Sova", "Cypher"]
    comp_b = ["Jett", "Omen", "Viper", "Sova", "Killjoy"]

    players_1 = [
        _build_player("p1", "A1", "t1", "Cloud9", comp_a[0], "Duelist", 3000),
        _build_player("p2", "A2", "t1", "Cloud9", comp_a[1], "Duelist", 3000),
        _build_player("p3", "A3", "t1", "Cloud9", comp_a[2], "Controller", 3000),
        _build_player("p4", "A4", "t1", "Cloud9", comp_a[3], "Initiator", 3000),
        _build_player("p5", "A5", "t1", "Cloud9", comp_a[4], "Sentinel", 3000),
    ]
    players_2 = [
        _build_player("p1", "A1", "t1", "Cloud9", comp_a[0], "Duelist", 3000),
        _build_player("p2", "A2", "t1", "Cloud9", comp_a[1], "Duelist", 3000),
        _build_player("p3", "A3", "t1", "Cloud9", comp_a[2], "Controller", 3000),
        _build_player("p4", "A4", "t1", "Cloud9", comp_a[3], "Initiator", 3000),
        _build_player("p5", "A5", "t1", "Cloud9", comp_a[4], "Sentinel", 3000),
    ]
    players_3 = [
        _build_player("p1", "A1", "t1", "Cloud9", comp_b[0], "Duelist", 3000),
        _build_player("p2", "A2", "t1", "Cloud9", comp_b[1], "Controller", 3000),
        _build_player("p3", "A3", "t1", "Cloud9", comp_b[2], "Controller", 3000),
        _build_player("p4", "A4", "t1", "Cloud9", comp_b[3], "Initiator", 3000),
        _build_player("p5", "A5", "t1", "Cloud9", comp_b[4], "Sentinel", 3000),
    ]

    match_1 = _build_match("m1", "Ascent", players=players_1)
    match_2 = _build_match("m2", "Ascent", players=players_2)
    match_3 = _build_match("m3", "Bind", players=players_3)
    analyzer = ScoutingAnalyzer([match_1, match_2, match_3], team_name="Cloud9")

    compositions = analyzer.get_recent_compositions()
    overall = compositions["overall"][0]
    assert overall["composition"] == sorted(comp_a)
    assert overall["pick_rate"] == 66.7
    assert compositions["by_map"]["Ascent"][0]["pick_rate"] == 100.0
    assert compositions["most_recent"] == sorted(comp_a)
