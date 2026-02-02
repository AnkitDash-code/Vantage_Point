from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

from datetime import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _convert_to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


AGENT_ROLE_MAP = {
    "astra": "Controller",
    "breach": "Initiator",
    "brimstone": "Controller",
    "chamber": "Sentinel",
    "clove": "Controller",
    "cypher": "Sentinel",
    "deadlock": "Sentinel",
    "fade": "Initiator",
    "gekko": "Initiator",
    "harbor": "Controller",
    "iso": "Duelist",
    "jett": "Duelist",
    "kay/o": "Initiator",
    "killjoy": "Sentinel",
    "neon": "Duelist",
    "omen": "Controller",
    "phoenix": "Duelist",
    "raze": "Duelist",
    "reyna": "Duelist",
    "sage": "Sentinel",
    "skye": "Initiator",
    "sova": "Initiator",
    "tejo": "Initiator",
    "veto": "Sentinel",
    "viper": "Controller",
    "vyse": "Sentinel",
    "waylay": "Duelist",
    "yoru": "Duelist",
}


class ScoutingAnalyzer:
    """Analyze match data to produce scouting metrics."""

    def __init__(
        self,
        matches: List[Dict],
        team_name: str,
        events_by_series: Optional[Dict[str, List[Dict]]] = None,
        cached_event_metrics: Optional[Dict] = None,
        cached_full_metrics: Optional[Dict] = None,
    ) -> None:
        self.matches = matches
        self.team_name = team_name
        self.team_name_lower = team_name.strip().lower()
        self.df_rounds = pd.DataFrame()
        self.df_players = pd.DataFrame()
        self.events_by_series = events_by_series or {}
        self._target_players_cache: Optional[pd.DataFrame] = None
        self._event_metrics_cache: Optional[Dict] = cached_event_metrics
        self._full_metrics_cache: Optional[Dict] = cached_full_metrics
        # Cache for intermediate event parsing results
        self._parsed_events_cache: Optional[Tuple] = None
        self._round_metadata_cache: Optional[Tuple] = None
        self._player_names_cache: Optional[Dict[str, str]] = None
        self._team_id_cache: Optional[str] = None
        # Only normalize if we don't have full cached metrics (skip heavy DataFrame ops)
        if cached_full_metrics is None:
            self._normalize_data()
        # Event metrics are now lazy-loaded via property

    @property
    def event_metrics(self) -> Dict:
        """Lazy load event metrics - only compute when accessed."""
        if self._event_metrics_cache is None:
            t0 = time.perf_counter()
            self._event_metrics_cache = self._compute_event_metrics()
            logger.info(f"[ANALYZER TIMING] _compute_event_metrics: {time.perf_counter() - t0:.2f}s")
        return self._event_metrics_cache

    def _normalize_data(self) -> None:
        t0 = time.perf_counter()
        rounds_data: List[Dict] = []
        players_data: List[Dict] = []

        for match in self.matches:
            match_id = match.get("id")
            if match_id is not None:
                match_id = str(match_id)
            map_name = (match.get("map") or {}).get("name")
            if map_name is not None:
                map_name = str(map_name)
            target_team_id = self._find_team_id(match)

            for segment in match.get("segments", []) or []:
                winner = segment.get("winner") or {}
                winner_id = winner.get("id") if isinstance(winner, dict) else winner
                if winner_id is not None:
                    winner_id = str(winner_id)
                rounds_data.append(
                    {
                        "match_id": match_id,
                        "map_name": map_name,
                        "round_number": segment.get("segmentIndex"),
                        "winning_team_id": winner_id,
                        "target_team_id": target_team_id,
                        "end_reason": segment.get("endReason"),
                        "duration": segment.get("duration"),
                        "plant_location": segment.get("plantLocation"),
                    }
                )

            for player in match.get("players", []) or []:
                team = player.get("team") or {}
                team_id = team.get("id")
                if team_id is not None:
                    team_id = str(team_id)
                team_name = team.get("name") or team.get("teamName")
                is_target: Optional[bool] = None
                if team_id is not None and target_team_id:
                    is_target = str(team_id) == str(target_team_id)
                if is_target is None and team_name:
                    is_target = (
                        str(team_name).strip().lower() == self.team_name_lower
                    )
                stats = player.get("playerStats") or {}
                agent = player.get("agent") or {}
                agent_name = agent.get("name")
                agent_role = agent.get("role")
                if not agent_role and agent_name:
                    agent_role = AGENT_ROLE_MAP.get(str(agent_name).strip().lower())
                players_data.append(
                    {
                        "match_id": match_id,
                        "map_name": map_name,
                        "player_id": player.get("id"),
                        "player_name": player.get("inGameName"),
                        "team_id": team_id,
                        "team_name": team_name,
                        "is_target": is_target,
                        "agent": agent_name,
                        "agent_role": agent_role,
                        "kills": stats.get("kills"),
                        "deaths": stats.get("deaths"),
                        "assists": stats.get("assists"),
                        "acs": stats.get("acs"),
                        "economy": stats.get("economy"),
                    }
                )

        self.df_rounds = pd.DataFrame(rounds_data)
        self.df_players = pd.DataFrame(players_data)

        if not self.df_rounds.empty:
            self.df_rounds["duration"] = self.df_rounds["duration"].apply(
                _parse_duration_seconds
            )

        if not self.df_players.empty:
            self.df_players["economy"] = pd.to_numeric(
                self.df_players["economy"], errors="coerce"
            )
        logger.info(f"[ANALYZER TIMING] _normalize_data: {time.perf_counter() - t0:.2f}s ({len(rounds_data)} rounds, {len(players_data)} players)")

    def _find_team_id(self, match: Dict) -> Optional[str]:
        for team in match.get("teams", []) or []:
            base_info = team.get("baseInfo") or {}
            base_name = base_info.get("name") if isinstance(base_info, dict) else None
            name = str(team.get("name") or base_name or team.get("teamName") or "").strip().lower()
            if name == self.team_name_lower:
                return str(team.get("id") or team.get("name"))
        for player in match.get("players", []) or []:
            team = player.get("team") or {}
            base_info = team.get("baseInfo") or {}
            base_name = base_info.get("name") if isinstance(base_info, dict) else None
            name = str(team.get("name") or base_name or team.get("teamName") or "").strip().lower()
            if name == self.team_name_lower:
                team_id = team.get("id") or team.get("name")
                if team_id is not None:
                    return str(team_id)
        return None

    def _resolve_team_id(self) -> Optional[str]:
        """Get target team ID with caching."""
        if self._team_id_cache is not None:
            return self._team_id_cache
        for match in self.matches:
            team_id = self._find_team_id(match)
            if team_id:
                self._team_id_cache = str(team_id)
                return self._team_id_cache
        return None

    def _target_players(self) -> pd.DataFrame:
        if self._target_players_cache is not None:
            return self._target_players_cache
        if "is_target" in self.df_players.columns:
            target_df = self.df_players[self.df_players["is_target"] == True]
            if not target_df.empty:
                self._target_players_cache = target_df
                return target_df
            if self.df_players["is_target"].notna().any():
                self._target_players_cache = target_df
                return target_df
        self._target_players_cache = self.df_players
        return self.df_players

    def get_win_rate(self, map_name: Optional[str] = None) -> float:
        df = self.df_rounds
        if df.empty:
            return 0.0

        if map_name:
            map_series = _clean_string_series(df["map_name"])
            map_filter = map_name.strip().lower()
            df = df[map_series.notna()]
            df = df[map_series.str.lower() == map_filter]

        df = df[df["target_team_id"].notna()]
        df = df[df["winning_team_id"].notna()]
        if df.empty:
            return 0.0

        wins = (df["winning_team_id"] == df["target_team_id"]).sum()
        return round((wins / len(df)) * 100, 1)

    def get_site_preferences(self) -> Dict[str, float]:
        event_sites = self._site_preferences_from_events()
        if event_sites:
            return event_sites

        df = self.df_rounds
        if df.empty:
            return {}

        site_series = _clean_string_series(df["plant_location"]).dropna()
        if site_series.empty:
            return {}

        site_counts = site_series.value_counts()
        total = site_counts.sum()
        return {
            str(site): round((count / total) * 100, 1)
            for site, count in site_counts.items()
        }

    def get_pistol_site_preferences(self) -> Dict[str, float]:
        df = self.df_rounds
        if df.empty:
            return {}

        round_numbers = pd.to_numeric(df["round_number"], errors="coerce")
        pistol_mask = round_numbers.isin([1, 13])
        if not pistol_mask.any():
            return {}

        site_series = _clean_string_series(df.loc[pistol_mask, "plant_location"]).dropna()
        if site_series.empty:
            return {}

        site_counts = site_series.value_counts()
        total = site_counts.sum()
        return {
            str(site): round((count / total) * 100, 1)
            for site, count in site_counts.items()
        }

    def get_aggression_index(self) -> Dict[str, float]:
        df = self.df_rounds
        if df.empty:
            return {"style": "Unknown", "avg_duration": 0.0, "rush_rate": 0.0}

        duration_series = df["duration"].dropna()
        if duration_series.empty:
            # Fallback: use avg first kill time from event metrics if available
            first_duel = self.event_metrics.get("first_duel") or {}
            avg_time = first_duel.get("avg_time_to_first_kill")
            if avg_time is not None:
                if avg_time < 25:
                    style = "Rush"
                elif avg_time < 50:
                    style = "Default"
                else:
                    style = "Slow/Default"
                return {
                    "style": style,
                    "avg_duration": round(avg_time, 1),
                    "rush_rate": 0.0,  # Can't compute without round duration
                }
            return {"style": "Unknown", "avg_duration": 0.0, "rush_rate": 0.0}

        avg_duration = duration_series.mean()
        if avg_duration < 40:
            style = "Rush"
        elif avg_duration < 80:
            style = "Default"
        else:
            style = "Slow/Default"

        rush_rate = (duration_series < 40).sum() / len(duration_series) * 100
        return {
            "style": style,
            "avg_duration": round(avg_duration, 1),
            "rush_rate": round(rush_rate, 1),
        }

    def get_agent_composition(self) -> List[Dict[str, float]]:
        df = self._target_players()
        if df.empty:
            return []

        agent_series = _clean_string_series(df["agent"]).dropna()
        if agent_series.empty:
            return []

        agent_counts = agent_series.value_counts().head(5)
        total_picks = len(agent_series)
        return [
            {
                "agent": str(agent),
                "pick_count": int(count),
                "pick_rate": round((count / total_picks) * 100, 1),
            }
            for agent, count in agent_counts.items()
        ]

    def get_player_tendencies(self, include_first_kill: bool = True) -> List[Dict[str, object]]:
        """Get player tendency stats. Set include_first_kill=False for fast mode."""
        df = self._target_players()
        if df.empty:
            return []

        name_series = _clean_string_series(df["player_name"]).dropna()
        if name_series.empty:
            return []

        df = df.copy()
        df["player_name"] = name_series
        df["agent"] = _clean_string_series(df["agent"])
        df["kills"] = df["kills"].fillna(0)
        df["deaths"] = df["deaths"].fillna(0)
        df["assists"] = df["assists"].fillna(0)

        # Get first kill rates only if requested (triggers event parsing)
        first_kill_rate = {}
        if include_first_kill:
            first_duel = self.event_metrics.get("first_duel") or {}
            top_first_kill_players = first_duel.get("top_first_kill_players") or []
            first_kill_rate = {
                entry.get("player"): entry.get("first_kill_rate")
                for entry in top_first_kill_players
                if entry.get("player")
            }

        # Optimized pandas aggregation instead of loop
        agg_df = df.groupby("player_name").agg(
            match_count=("match_id", "nunique"),
            total_kills=("kills", "sum"),
            total_deaths=("deaths", "sum"),
            total_assists=("assists", "sum"),
            avg_kills=("kills", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_assists=("assists", "mean"),
            avg_acs=("acs", "mean"),
            top_agent=("agent", lambda x: x.dropna().value_counts().index[0] if not x.dropna().empty else None),
            top_agent_count=("agent", lambda x: x.dropna().value_counts().iloc[0] if not x.dropna().empty else 0),
        ).reset_index()

        records: List[Dict[str, object]] = []
        for _, row in agg_df.iterrows():
            player_name = str(row["player_name"])
            match_count = int(row["match_count"]) if row["match_count"] > 0 else 1
            total_deaths = row["total_deaths"]
            kd_ratio = round(row["total_kills"] / total_deaths, 2) if total_deaths > 0 else None
            avg_acs = round(row["avg_acs"], 1) if pd.notna(row["avg_acs"]) else None
            top_agent_rate = round((row["top_agent_count"] / match_count) * 100, 1) if row["top_agent"] else 0.0

            records.append({
                "player": player_name,
                "matches_played": match_count,
                "top_agent": str(row["top_agent"]) if row["top_agent"] else None,
                "top_agent_rate": top_agent_rate,
                "avg_kills": round(row["avg_kills"], 1),
                "avg_deaths": round(row["avg_deaths"], 1),
                "avg_assists": round(row["avg_assists"], 1),
                "kd_ratio": kd_ratio,
                "avg_acs": avg_acs,
                "first_kill_rate": first_kill_rate.get(player_name),
            })

        records.sort(key=lambda item: (-item["matches_played"], item["player"]))
        return records

    def get_role_distribution(self) -> Dict[str, float]:
        df = self._target_players()
        if df.empty:
            return {}

        role_series = _clean_string_series(df["agent_role"]).dropna()
        if role_series.empty:
            return {}

        role_counts = role_series.value_counts()
        total = role_counts.sum()
        return {
            str(role): round((count / total) * 100, 1)
            for role, count in role_counts.items()
        }

    def get_recent_compositions(self) -> Dict[str, object]:
        df = self._target_players()
        if df.empty:
            return {"overall": [], "by_map": {}, "most_recent": []}

        df = df.copy()
        df["agent"] = _clean_string_series(df["agent"])
        df["map_name"] = _clean_string_series(df["map_name"]).fillna("Unknown")
        df = df[df["agent"].notna()]
        df = df[df["match_id"].notna()]
        if df.empty:
            return {"overall": [], "by_map": {}, "most_recent": []}

        compositions: List[Dict[str, object]] = []
        for (match_id, map_name), group in df.groupby(["match_id", "map_name"]):
            agents = sorted(set(group["agent"].dropna().tolist()))
            if not agents:
                continue
            comp_key = "|".join(agents)
            compositions.append(
                {
                    "match_id": str(match_id),
                    "map_name": str(map_name),
                    "composition": agents,
                    "composition_key": comp_key,
                }
            )

        if not compositions:
            return {"overall": [], "by_map": {}, "most_recent": []}

        comp_df = pd.DataFrame(compositions)
        overall_counts = comp_df["composition_key"].value_counts()
        overall_total = overall_counts.sum()
        overall: List[Dict[str, object]] = []
        for comp_key, count in overall_counts.head(3).items():
            comp_agents = (
                comp_df[comp_df["composition_key"] == comp_key]["composition"].iloc[0]
            )
            overall.append(
                {
                    "composition": comp_agents,
                    "matches": int(count),
                    "pick_rate": round((count / overall_total) * 100, 1),
                }
            )

        by_map: Dict[str, List[Dict[str, object]]] = {}
        for map_name, group in comp_df.groupby("map_name"):
            counts = group["composition_key"].value_counts()
            total = counts.sum()
            entries: List[Dict[str, object]] = []
            for comp_key, count in counts.head(3).items():
                comp_agents = (
                    group[group["composition_key"] == comp_key]["composition"].iloc[0]
                )
                entries.append(
                    {
                        "composition": comp_agents,
                        "matches": int(count),
                        "pick_rate": round((count / total) * 100, 1),
                    }
                )
            by_map[str(map_name)] = entries

        most_recent: List[str] = []
        for match in self.matches:
            match_id = match.get("id")
            if match_id is None:
                continue
            row = comp_df[comp_df["match_id"] == str(match_id)].head(1)
            if not row.empty:
                most_recent = row.iloc[0]["composition"]
                break

        return {"overall": overall, "by_map": by_map, "most_recent": most_recent}

    def get_win_rate_by_map(self) -> Dict[str, float]:
        df = self.df_rounds
        if df.empty:
            return {}

        df = df[df["target_team_id"].notna()]
        df = df[df["winning_team_id"].notna()]
        if df.empty:
            return {}

        results = {}
        map_series = _clean_string_series(df["map_name"])
        df = df[map_series.notna()]
        if df.empty:
            return {}
        map_series = map_series[map_series.notna()]
        for map_name, group in df.groupby(map_series):
            wins = (group["winning_team_id"] == group["target_team_id"]).sum()
            # Use title case for consistent map names
            results[str(map_name).title()] = round((wins / len(group)) * 100, 1)
        return results

    def get_economy_distribution(self) -> Dict[str, float]:
        df = self._target_players()
        if df.empty:
            return {"eco": 0.0, "force": 0.0, "full": 0.0}

        economy_df = df[df["economy"].notna()]
        economy_df = economy_df[economy_df["economy"] >= 0]
        economy_df = economy_df[economy_df["match_id"].notna()]
        if economy_df.empty:
            return {"eco": 0.0, "force": 0.0, "full": 0.0}

        match_economy = economy_df.groupby("match_id")["economy"].median()
        total = len(match_economy)
        if total == 0:
            return {"eco": 0.0, "force": 0.0, "full": 0.0}

        eco = (match_economy < 2000).sum()
        force = ((match_economy >= 2000) & (match_economy < 3900)).sum()
        full = (match_economy >= 3900).sum()

        return {
            "eco": round((eco / total) * 100, 1),
            "force": round((force / total) * 100, 1),
            "full": round((full / total) * 100, 1),
        }

    def generate_fast_metrics(self) -> Dict:
        """Return quick metrics that don't require event parsing - shows data instantly."""
        return _convert_to_serializable({
            "win_rate": self.get_win_rate(),
            "win_rate_by_map": self.get_win_rate_by_map(),
            "agent_composition": self.get_agent_composition(),
            "role_distribution": self.get_role_distribution(),
            "economy": self.get_economy_distribution(),
            "opponent_stats": self.get_opponent_stats(),
            "round_type_performance": self.get_round_type_performance(),
            # Fast player tendencies without first_kill_rate (no event parsing)
            "player_tendencies": self.get_player_tendencies(include_first_kill=False),
        })

    def generate_detailed_metrics(self) -> Dict:
        """Return detailed metrics that require event parsing."""
        detailed = {
            "site_preferences": self.get_site_preferences(),
            "pistol_site_preferences": self.get_pistol_site_preferences(),
            "aggression": self.get_aggression_index(),
            # Full player tendencies with first_kill_rate (triggers event parsing)
            "player_tendencies": self.get_player_tendencies(include_first_kill=True),
            "recent_compositions": self.get_recent_compositions(),
            "map_detailed": self.get_map_detailed_stats(),
        }
        # Add event-based metrics (triggers lazy loading)
        detailed.update(self.event_metrics)
        return _convert_to_serializable(detailed)

    def generate_metrics_summary(self) -> Dict:
        # If we have fully cached metrics, return them directly (massive speedup)
        if self._full_metrics_cache is not None:
            return self._full_metrics_cache
            
        summary = {
            "win_rate": self.get_win_rate(),
            "win_rate_by_map": self.get_win_rate_by_map(),
            "site_preferences": self.get_site_preferences(),
            "pistol_site_preferences": self.get_pistol_site_preferences(),
            "aggression": self.get_aggression_index(),
            "agent_composition": self.get_agent_composition(),
            "player_tendencies": self.get_player_tendencies(),
            "role_distribution": self.get_role_distribution(),
            "recent_compositions": self.get_recent_compositions(),
            "economy": self.get_economy_distribution(),
            "opponent_stats": self.get_opponent_stats(),
            "map_detailed": self.get_map_detailed_stats(),
            "round_type_performance": self.get_round_type_performance(),
        }
        summary.update(self.event_metrics)
        
        # Map event_metrics keys to frontend-expected names
        side_performance = self.event_metrics.get("side_performance", {})
        if side_performance:
            attack_data = side_performance.get("attack", {})
            defense_data = side_performance.get("defense", {})
            summary["side_metrics"] = {
                "attack_rounds": attack_data.get("rounds", 0),
                "attack_wins": attack_data.get("wins", 0),
                "attack_win_rate": attack_data.get("win_rate", 0.0),
                "attack_kills": attack_data.get("kills", 0),
                "attack_deaths": attack_data.get("deaths", 0),
                "attack_kd": attack_data.get("kd_ratio", 0.0),
                "defense_rounds": defense_data.get("rounds", 0),
                "defense_wins": defense_data.get("wins", 0),
                "defense_win_rate": defense_data.get("win_rate", 0.0),
                "defense_kills": defense_data.get("kills", 0),
                "defense_deaths": defense_data.get("deaths", 0),
                "defense_kd": defense_data.get("kd_ratio", 0.0),
            }
        
        # Map combat to combat_metrics for frontend
        combat_data = self.event_metrics.get("combat", {})
        if combat_data:
            summary["combat_metrics"] = combat_data
        
        # Map new advanced metrics for frontend
        pace_data = self.event_metrics.get("pace", {})
        if pace_data:
            summary["pace_metrics"] = pace_data
        
        site_bias_data = self.event_metrics.get("site_bias", {})
        if site_bias_data:
            summary["site_bias"] = site_bias_data
        
        first_death_data = self.event_metrics.get("first_death_context", {})
        if first_death_data:
            summary["first_death_context"] = first_death_data
        
        ult_data = self.event_metrics.get("ultimate_impact", {})
        if ult_data:
            summary["ultimate_impact"] = ult_data
        
        man_adv_data = self.event_metrics.get("man_advantage", {})
        if man_adv_data:
            summary["man_advantage"] = man_adv_data
        
        discipline_data = self.event_metrics.get("discipline", {})
        if discipline_data:
            summary["discipline"] = discipline_data
        
        # Convert all numpy/pandas types to native Python for JSON serialization
        return _convert_to_serializable(summary)

    def get_opponent_stats(self) -> List[Dict]:
        """Calculate stats grouped by opponent team."""
        if self.df_rounds.empty:
            return []
        
        opponent_stats: Dict[str, Dict] = {}
        
        df = self.df_rounds
        df = df[df["match_id"].notna()]
        df = df[df["target_team_id"].notna() & df["winning_team_id"].notna()]
        if df.empty:
            return []
        
        total_rounds_by_match = df.groupby("match_id").size().to_dict()
        wins_by_match = (
            (df["winning_team_id"] == df["target_team_id"]).groupby(df["match_id"]).sum().to_dict()
        )
        
        for match in self.matches:
            match_id = str(match.get("id") or "")
            if match_id not in total_rounds_by_match:
                continue
            target_team_id = self._find_team_id(match)
            
            # Find opponent team
            opponent_name = None
            opponent_id = None
            for team in match.get("teams", []) or []:
                team_id = str(team.get("id") or "")
                if team_id != target_team_id:
                    opponent_name = team.get("name")
                    opponent_id = team_id
                    break
            
            if not opponent_name:
                continue
            
            total_rounds = total_rounds_by_match.get(match_id, 0)
            wins = wins_by_match.get(match_id, 0)
            
            if opponent_name not in opponent_stats:
                opponent_stats[opponent_name] = {
                    "matches": 0,
                    "rounds_played": 0,
                    "rounds_won": 0,
                }
            
            opponent_stats[opponent_name]["matches"] += 1
            opponent_stats[opponent_name]["rounds_played"] += total_rounds
            opponent_stats[opponent_name]["rounds_won"] += wins
        
        # Calculate win rates
        result = []
        for opponent, stats in opponent_stats.items():
            rounds_played = stats["rounds_played"]
            rounds_won = stats["rounds_won"]
            result.append({
                "opponent": opponent,
                "matches": stats["matches"],
                "rounds_played": rounds_played,
                "rounds_won": rounds_won,
                "win_rate": round((rounds_won / rounds_played) * 100, 1) if rounds_played > 0 else 0.0,
            })
        
        return sorted(result, key=lambda x: x["matches"], reverse=True)

    def get_map_detailed_stats(self) -> Dict:
        """Get detailed stats broken down by map."""
        if self.df_rounds.empty or self.df_players.empty:
            return {}
        
        map_stats: Dict[str, Dict] = {}
        target_df = self._target_players()
        
        for map_name in self.df_rounds["map_name"].dropna().unique():
            map_name_clean = str(map_name).strip()
            if not map_name_clean or map_name_clean.lower() in ("none", "nan"):
                continue
            
            # Rounds on this map (case-insensitive)
            map_name_lower = map_name_clean.lower()
            map_rounds = self.df_rounds[
                self.df_rounds["map_name"].str.lower().str.strip() == map_name_lower
            ]
            total_rounds = len(map_rounds)
            if total_rounds == 0:
                continue
            
            wins = (map_rounds["winning_team_id"] == map_rounds["target_team_id"]).sum()
            
            # Attack vs defense on this map
            attack_rounds = 0
            attack_wins = 0
            defense_rounds = 0
            defense_wins = 0
            
            for match in self.matches:
                match_map = (match.get("map") or {}).get("name") or ""
                if match_map.lower().strip() != map_name_lower:
                    continue
                match_id = str(match.get("id") or "")
                team_id = self._find_team_id(match)
                
                for segment in match.get("segments", []) or []:
                    round_number = segment.get("segmentIndex")
                    winner = segment.get("winner") or {}
                    winner_id = str(winner.get("id")) if winner.get("id") else None
                    
                    for team in segment.get("teams", []) or []:
                        if str(team.get("id")) == team_id:
                            side = team.get("side")
                            if side == "attacker":
                                attack_rounds += 1
                                if winner_id == team_id:
                                    attack_wins += 1
                            elif side == "defender":
                                defense_rounds += 1
                                if winner_id == team_id:
                                    defense_wins += 1
            
            # Players on this map (case-insensitive)
            map_players = target_df[
                target_df["map_name"].str.lower().str.strip() == map_name_lower
            ]
            
            # Top agent on this map
            top_agent = None
            if not map_players.empty:
                agent_counts = map_players["agent"].dropna().value_counts()
                if len(agent_counts) > 0:
                    top_agent = str(agent_counts.index[0]).title()  # Capitalize nicely
            
            # Use title case for map name output
            map_stats[map_name_clean.title()] = {
                "rounds_played": total_rounds,
                "rounds_won": int(wins),
                "win_rate": round((wins / total_rounds) * 100, 1),
                "attack_rounds": attack_rounds,
                "attack_wins": attack_wins,
                "attack_win_rate": round((attack_wins / attack_rounds) * 100, 1) if attack_rounds > 0 else 0.0,
                "defense_rounds": defense_rounds,
                "defense_wins": defense_wins,
                "defense_win_rate": round((defense_wins / defense_rounds) * 100, 1) if defense_rounds > 0 else 0.0,
                "top_agent": top_agent,
            }
        
        return map_stats

    def get_round_type_performance(self) -> Dict:
        """Calculate win rates by round type (pistol, eco, force, full buy)."""
        if self.df_rounds.empty:
            return {}
        
        # Pistol rounds (1 and 13)
        pistol_mask = self.df_rounds["round_number"].isin([1, 13])
        pistol_rounds = self.df_rounds[pistol_mask]
        pistol_total = len(pistol_rounds)
        pistol_wins = (pistol_rounds["winning_team_id"] == pistol_rounds["target_team_id"]).sum() if pistol_total > 0 else 0
        
        # Eco rounds (2-3 after loss, 14-15 after loss) - simplified as rounds 2,3,14,15
        eco_mask = self.df_rounds["round_number"].isin([2, 3, 14, 15])
        eco_rounds = self.df_rounds[eco_mask]
        eco_total = len(eco_rounds)
        eco_wins = (eco_rounds["winning_team_id"] == eco_rounds["target_team_id"]).sum() if eco_total > 0 else 0
        
        # Force buy rounds (4-6 after loss, 16-18 after loss) - simplified
        force_mask = self.df_rounds["round_number"].isin([4, 5, 6, 16, 17, 18])
        force_rounds = self.df_rounds[force_mask]
        force_total = len(force_rounds)
        force_wins = (force_rounds["winning_team_id"] == force_rounds["target_team_id"]).sum() if force_total > 0 else 0
        
        # Full buy rounds (everything else)
        full_mask = ~(pistol_mask | eco_mask | force_mask)
        full_rounds = self.df_rounds[full_mask]
        full_total = len(full_rounds)
        full_wins = (full_rounds["winning_team_id"] == full_rounds["target_team_id"]).sum() if full_total > 0 else 0
        
        return {
            "pistol": {
                "rounds": pistol_total,
                "wins": int(pistol_wins),
                "win_rate": round((pistol_wins / pistol_total) * 100, 1) if pistol_total > 0 else 0.0,
            },
            "eco": {
                "rounds": eco_total,
                "wins": int(eco_wins),
                "win_rate": round((eco_wins / eco_total) * 100, 1) if eco_total > 0 else 0.0,
            },
            "force": {
                "rounds": force_total,
                "wins": int(force_wins),
                "win_rate": round((force_wins / force_total) * 100, 1) if force_total > 0 else 0.0,
            },
            "full_buy": {
                "rounds": full_total,
                "wins": int(full_wins),
                "win_rate": round((full_wins / full_total) * 100, 1) if full_total > 0 else 0.0,
            },
        }

    def _compute_event_metrics(self) -> Dict:
        if not self.events_by_series:
            return {}

        round_winner, round_team_side = self._build_round_metadata()
        allowed_match_ids = {
            str(match.get("id"))
            for match in self.matches
            if match.get("id") is not None
        }
        first_kills, round_start_times, event_rounds = self._parse_first_kill_events(
            allowed_match_ids
        )
        if not first_kills:
            return {}

        team_id = self._resolve_team_id()
        if not team_id:
            return {}

        total_rounds = len(event_rounds) or len(round_winner)
        if total_rounds == 0:
            return {}

        team_first_kills = [
            fk for fk in first_kills if fk.get("team_id") == team_id
        ]
        team_first_kill_rate = round(
            (len(team_first_kills) / total_rounds) * 100, 1
        )

        first_kill_wins = 0
        for fk in team_first_kills:
            key = (fk.get("match_id"), fk.get("round_number"))
            if round_winner.get(key) == team_id:
                first_kill_wins += 1
        first_kill_conversion_rate = (
            round((first_kill_wins / len(team_first_kills)) * 100, 1)
            if team_first_kills
            else 0.0
        )

        time_samples = [
            fk["time_to_first_kill"]
            for fk in team_first_kills
            if fk.get("time_to_first_kill") is not None
        ]
        avg_time_to_first_kill = (
            round(sum(time_samples) / len(time_samples), 1)
            if time_samples
            else None
        )

        player_names = self._player_name_lookup()
        player_counts: Dict[str, int] = {}
        for fk in team_first_kills:
            player_id = fk.get("player_id")
            if not player_id:
                continue
            player_counts[player_id] = player_counts.get(player_id, 0) + 1

        top_players = sorted(
            player_counts.items(), key=lambda item: item[1], reverse=True
        )[:5]
        top_first_kill_players = [
            {
                "player": player_names.get(pid, pid),
                "first_kills": count,
                "first_kill_rate": round((count / total_rounds) * 100, 1),
            }
            for pid, count in top_players
        ]

        weapon_counts: Dict[str, int] = {}
        for fk in team_first_kills:
            weapon = fk.get("weapon")
            if not weapon:
                continue
            weapon_counts[weapon] = weapon_counts.get(weapon, 0) + 1

        top_weapons = sorted(
            weapon_counts.items(), key=lambda item: item[1], reverse=True
        )[:3]
        top_first_kill_weapons = [
            {
                "weapon": weapon,
                "first_kills": count,
                "first_kill_rate": round((count / total_rounds) * 100, 1),
            }
            for weapon, count in top_weapons
        ]

        pistol_rounds = self._compute_pistol_round_metrics(
            team_id, round_winner, round_team_side, team_first_kills, event_rounds
        )

        # Parse ALL events in a single pass (kills, bombs, abilities)
        all_kills, bomb_events, ability_events = self._parse_all_events_optimized(allowed_match_ids, round_start_times)
        
        # Calculate advanced combat metrics
        combat_metrics = self._compute_combat_metrics(
            all_kills, team_id, total_rounds, round_winner, round_start_times, player_names
        )
        
        # Calculate economy metrics from round data
        economy_metrics = self._compute_economy_metrics(round_winner, round_team_side, team_id)
        
        # Calculate map-specific metrics
        map_metrics = self._compute_map_metrics(round_winner, round_team_side, bomb_events, team_id)
        
        # Calculate side-specific performance
        side_metrics = self._compute_side_metrics(round_winner, round_team_side, team_id, all_kills)
        
        # Calculate advanced metrics
        pace_metrics = self._compute_pace_metrics(bomb_events, all_kills, round_start_times, team_id, round_team_side)
        site_bias_metrics = self._compute_site_bias_metrics(bomb_events, round_winner, team_id)
        first_death_context = self._compute_first_death_context(all_kills, team_id, player_names, round_team_side)
        ult_metrics = self._compute_ultimate_metrics(ability_events, round_winner, team_id, player_names)
        man_advantage_metrics = self._compute_man_advantage_metrics(all_kills, round_winner, team_id)
        discipline_metrics = self._compute_discipline_metrics(all_kills, round_winner, round_team_side, team_id)

        return {
            "first_duel": {
                "team_first_kill_rate": team_first_kill_rate,
                "first_kill_conversion_rate": first_kill_conversion_rate,
                "avg_time_to_first_kill": avg_time_to_first_kill,
                "top_first_kill_players": top_first_kill_players,
                "top_first_kill_weapons": top_first_kill_weapons,
            },
            "pistol_rounds": pistol_rounds,
            "combat": combat_metrics,
            "economy_advanced": economy_metrics,
            "map_control": map_metrics,
            "side_performance": side_metrics,
            "pace": pace_metrics,
            "site_bias": site_bias_metrics,
            "first_death_context": first_death_context,
            "ultimate_impact": ult_metrics,
            "man_advantage": man_advantage_metrics,
            "discipline": discipline_metrics,
        }

    def _parse_all_events_optimized(
        self, allowed_match_ids: set[str], round_start_times: Dict[Tuple[str, int], str]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Parse ALL events (kills, bombs, abilities) in a single pass for performance."""
        all_kills: List[Dict] = []
        bomb_events: List[Dict] = []
        ability_events: List[Dict] = []
        current_round: Dict[str, int] = {}
        
        # Define ultimate abilities by agent
        ULTIMATE_ABILITIES = {
            "hunter's-fury": "sova",
            "orbital-strike": "brimstone",
            "showstopper": "raze",
            "lockdown": "killjoy",
            "from-the-shadows": "omen",
            "viper's-pit": "viper",
            "seekers": "skye",
            "annihilation": "neon",
            "steel-garden": "deadlock",
        }

        for events in self.events_by_series.values():
            for envelope in events:
                occurred_at = envelope.get("occurredAt")
                for event in envelope.get("events") or []:
                    game_id, round_number, started_at = _event_round_context(event)
                    
                    # Fallback: extract game_id from seriesStateDelta for events without segments
                    if not game_id:
                        series_delta = event.get("seriesStateDelta") or {}
                        games = series_delta.get("games") or []
                        if games:
                            game_id = str(games[0].get("id") or "")
                    
                    if game_id and allowed_match_ids and game_id not in allowed_match_ids:
                        continue

                    if game_id and round_number:
                        current_round[game_id] = round_number
                        if started_at and (game_id, round_number) not in round_start_times:
                            round_start_times[(game_id, round_number)] = started_at

                    event_type = (event.get("type") or "").lower()
                    
                    # Parse kill events
                    if event_type == "player-killed-player":
                        if not round_number and game_id and game_id in current_round:
                            round_number = current_round[game_id]
                        if not round_number:
                            continue

                        actor = event.get("actor") or {}
                        target = event.get("target") or {}
                        actor_state = actor.get("state") or {}
                        target_state = target.get("state") or {}
                        
                        killer_team_id = actor_state.get("teamId")
                        victim_team_id = target_state.get("teamId")
                        
                        position = actor.get("position") or {}
                        is_first = _is_first_kill(event)
                        
                        time_in_round = _compute_time_delta_seconds(
                            occurred_at,
                            round_start_times.get((game_id, round_number)),
                        )

                        all_kills.append({
                            "match_id": game_id,
                            "round_number": round_number,
                            "killer_id": str(actor.get("id")) if actor.get("id") else None,
                            "killer_team_id": str(killer_team_id) if killer_team_id else None,
                            "victim_id": str(target.get("id")) if target.get("id") else None,
                            "victim_team_id": str(victim_team_id) if victim_team_id else None,
                            "weapon": _extract_weapon_from_event(event),
                            "is_first_kill": is_first,
                            "occurred_at": occurred_at,
                            "time_in_round": time_in_round,
                            "position": position,
                        })
                    
                    # Parse bomb events
                    elif event_type in ("bomb-planted", "spike-planted", "plant-bomb", "player-completed-plantbomb"):
                        if not round_number and game_id and game_id in current_round:
                            round_number = current_round[game_id]
                        if not round_number:
                            continue
                            
                        actor = event.get("actor") or {}
                        actor_state = actor.get("state") or {}
                        game_state = actor_state.get("game") or {}
                        position = game_state.get("position") or event.get("position") or actor.get("position") or {}
                        
                        plant_time = _compute_time_delta_seconds(
                            occurred_at,
                            round_start_times.get((game_id, round_number)),
                        )
                        
                        map_name = _map_name_from_event(event, game_id)
                        site = _infer_site_from_position(map_name, position)

                        bomb_events.append({
                            "match_id": game_id,
                            "round_number": round_number,
                            "event_type": "planted",
                            "player_id": str(actor.get("id")) if actor.get("id") else None,
                            "team_id": str(actor_state.get("teamId")) if actor_state.get("teamId") else None,
                            "occurred_at": occurred_at,
                            "time_in_round": plant_time,
                            "position": position,
                            "site": site,
                            "map_name": map_name,
                        })
                    
                    elif event_type in ("bomb-defused", "spike-defused", "defuse-bomb", "player-completed-defusebomb"):
                        if not round_number and game_id and game_id in current_round:
                            round_number = current_round[game_id]
                        if not round_number:
                            continue
                            
                        actor = event.get("actor") or {}
                        actor_state = actor.get("state") or {}

                        bomb_events.append({
                            "match_id": game_id,
                            "round_number": round_number,
                            "event_type": "defused",
                            "player_id": str(actor.get("id")) if actor.get("id") else None,
                            "team_id": str(actor_state.get("teamId")) if actor_state.get("teamId") else None,
                            "occurred_at": occurred_at,
                        })
                    
                    elif event_type in ("bomb-exploded", "spike-exploded", "player-completed-explodebomb"):
                        if not round_number and game_id and game_id in current_round:
                            round_number = current_round[game_id]
                        if not round_number:
                            continue

                        bomb_events.append({
                            "match_id": game_id,
                            "round_number": round_number,
                            "event_type": "exploded",
                            "occurred_at": occurred_at,
                        })
                    
                    # Parse ability events
                    elif event_type == "player-used-ability":
                        if not round_number and game_id and game_id in current_round:
                            round_number = current_round[game_id]
                        if not round_number:
                            continue
                        
                        target = event.get("target") or {}
                        ability_name = target.get("name") or target.get("id") or ""
                        
                        actor = event.get("actor") or {}
                        actor_state = actor.get("state") or {}
                        game_state = actor_state.get("game") or {}
                        character = game_state.get("character") or {}
                        
                        is_ultimate = ability_name.lower() in ULTIMATE_ABILITIES
                        
                        # Get agent name - from character data or from ultimate mapping
                        agent_name = character.get("name") or character.get("id")
                        if not agent_name and is_ultimate:
                            agent_name = ULTIMATE_ABILITIES.get(ability_name.lower())
                        
                        ability_events.append({
                            "match_id": game_id,
                            "round_number": round_number,
                            "player_id": str(actor.get("id")) if actor.get("id") else None,
                            "player_name": actor_state.get("name"),
                            "team_id": str(actor_state.get("teamId")) if actor_state.get("teamId") else None,
                            "ability": ability_name,
                            "agent": agent_name,
                            "is_ultimate": is_ultimate,
                            "occurred_at": occurred_at,
                        })

        return all_kills, bomb_events, ability_events

    def _parse_all_events(
        self, allowed_match_ids: set[str], round_start_times: Dict[Tuple[str, int], str]
    ) -> Tuple[List[Dict], List[Dict]]:
        """DEPRECATED: Use _parse_all_events_optimized instead. Kept for compatibility."""
        all_kills, bomb_events, _ = self._parse_all_events_optimized(allowed_match_ids, round_start_times)
        return all_kills, bomb_events

    def _compute_combat_metrics(
        self,
        all_kills: List[Dict],
        team_id: str,
        total_rounds: int,
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_start_times: Dict[Tuple[str, int], str],
        player_names: Dict[str, str],
    ) -> Dict:
        """Calculate advanced combat metrics from all kill events."""
        if not all_kills or total_rounds == 0:
            return {}

        # Group kills by round
        kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}
        for kill in all_kills:
            key = (kill.get("match_id"), kill.get("round_number"))
            kills_by_round.setdefault(key, []).append(kill)
        
        # Sort kills within each round by timestamp
        for key in kills_by_round:
            kills_by_round[key].sort(key=lambda k: k.get("occurred_at") or "")

        # Trade kill detection (kill within 5s of teammate death)
        trade_kills = 0
        trade_opportunities = 0
        player_trades: Dict[str, Dict[str, int]] = {}  # player_id -> {trades: X, opportunities: Y}
        
        for round_key, round_kills in kills_by_round.items():
            for i, kill in enumerate(round_kills):
                victim_team = kill.get("victim_team_id")
                if victim_team != team_id:
                    continue  # Our team member died
                
                trade_opportunities += 1
                victim_id = kill.get("victim_id")
                kill_time = _parse_iso_timestamp(kill.get("occurred_at"))
                if not kill_time:
                    continue
                
                # Check if a teammate got a trade kill within 5 seconds
                for j in range(i + 1, min(i + 5, len(round_kills))):
                    next_kill = round_kills[j]
                    if next_kill.get("killer_team_id") == team_id:
                        next_time = _parse_iso_timestamp(next_kill.get("occurred_at"))
                        if next_time and (next_time - kill_time).total_seconds() <= 5.0:
                            trade_kills += 1
                            trader_id = next_kill.get("killer_id")
                            if trader_id:
                                player_trades.setdefault(trader_id, {"trades": 0, "opportunities": 0})
                                player_trades[trader_id]["trades"] += 1
                            break
                
                # Track player trade opportunities
                if victim_id:
                    player_trades.setdefault(victim_id, {"trades": 0, "opportunities": 0})
        
        trade_efficiency = round((trade_kills / trade_opportunities) * 100, 1) if trade_opportunities > 0 else 0.0

        # Multi-kill detection (2K, 3K, 4K, 5K in a round)
        multi_kills: Dict[str, Dict[str, int]] = {}  # player_id -> {2k: X, 3k: Y, 4k: Z, 5k: W}
        
        for round_key, round_kills in kills_by_round.items():
            player_round_kills: Dict[str, int] = {}
            for kill in round_kills:
                if kill.get("killer_team_id") == team_id:
                    killer_id = kill.get("killer_id")
                    if killer_id:
                        player_round_kills[killer_id] = player_round_kills.get(killer_id, 0) + 1
            
            for player_id, kills_count in player_round_kills.items():
                if kills_count >= 2:
                    multi_kills.setdefault(player_id, {"2k": 0, "3k": 0, "4k": 0, "5k": 0})
                    if kills_count >= 5:
                        multi_kills[player_id]["5k"] += 1
                    elif kills_count >= 4:
                        multi_kills[player_id]["4k"] += 1
                    elif kills_count >= 3:
                        multi_kills[player_id]["3k"] += 1
                    else:
                        multi_kills[player_id]["2k"] += 1

        # Clutch detection (1vX situations)
        clutch_situations: Dict[str, Dict[str, int]] = {}  # player_id -> {faced: X, won: Y}
        
        for round_key, round_kills in kills_by_round.items():
            # Track alive players through the round
            team_alive = set()
            enemy_alive = set()
            
            # Initialize with all players (simplified - assume 5v5)
            for kill in round_kills:
                if kill.get("killer_team_id") == team_id:
                    team_alive.add(kill.get("killer_id"))
                else:
                    enemy_alive.add(kill.get("killer_id"))
                if kill.get("victim_team_id") == team_id:
                    team_alive.add(kill.get("victim_id"))
                else:
                    enemy_alive.add(kill.get("victim_id"))
            
            # Simulate kills to find 1vX scenarios
            team_alive_count = 5
            enemy_alive_count = 5
            last_team_player = None
            clutch_started = False
            enemies_at_clutch_start = 0
            
            for kill in round_kills:
                if kill.get("victim_team_id") == team_id:
                    team_alive_count -= 1
                    if team_alive_count == 1 and not clutch_started:
                        # Find who's the last player
                        for k in round_kills:
                            if k.get("killer_team_id") == team_id:
                                last_team_player = k.get("killer_id")
                        if last_team_player:
                            clutch_started = True
                            enemies_at_clutch_start = enemy_alive_count
                else:
                    enemy_alive_count -= 1
            
            if clutch_started and last_team_player and enemies_at_clutch_start >= 1:
                clutch_situations.setdefault(last_team_player, {"faced": 0, "won": 0})
                clutch_situations[last_team_player]["faced"] += 1
                if round_winner.get(round_key) == team_id:
                    clutch_situations[last_team_player]["won"] += 1

        # Weapon effectiveness
        weapon_kills: Dict[str, int] = {}
        weapon_deaths: Dict[str, int] = {}
        
        for kill in all_kills:
            weapon = kill.get("weapon")
            if not weapon:
                continue
            if kill.get("killer_team_id") == team_id:
                weapon_kills[weapon] = weapon_kills.get(weapon, 0) + 1
            if kill.get("victim_team_id") == team_id:
                weapon_deaths[weapon] = weapon_deaths.get(weapon, 0) + 1

        # Opening duel stats by player
        opening_duels: Dict[str, Dict[str, int]] = {}  # player_id -> {wins: X, losses: Y}
        
        for round_key, round_kills in kills_by_round.items():
            if not round_kills:
                continue
            first_kill = round_kills[0]
            killer_id = first_kill.get("killer_id")
            victim_id = first_kill.get("victim_id")
            
            if first_kill.get("killer_team_id") == team_id and killer_id:
                opening_duels.setdefault(killer_id, {"wins": 0, "losses": 0})
                opening_duels[killer_id]["wins"] += 1
            if first_kill.get("victim_team_id") == team_id and victim_id:
                opening_duels.setdefault(victim_id, {"wins": 0, "losses": 0})
                opening_duels[victim_id]["losses"] += 1

        # Build top multi-killers
        top_multi_killers = []
        for player_id, counts in sorted(multi_kills.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
            top_multi_killers.append({
                "player": player_names.get(player_id, player_id),
                "2k": counts.get("2k", 0),
                "3k": counts.get("3k", 0),
                "4k": counts.get("4k", 0),
                "5k": counts.get("5k", 0),
                "total": sum(counts.values()),
            })

        # Build clutch performers
        clutch_performers = []
        for player_id, stats in sorted(clutch_situations.items(), key=lambda x: x[1].get("won", 0), reverse=True)[:5]:
            faced = stats.get("faced", 0)
            won = stats.get("won", 0)
            clutch_performers.append({
                "player": player_names.get(player_id, player_id),
                "clutches_faced": faced,
                "clutches_won": won,
                "clutch_rate": round((won / faced) * 100, 1) if faced > 0 else 0.0,
            })

        # Build opening duel stats
        opening_duel_stats = []
        for player_id, stats in sorted(opening_duels.items(), key=lambda x: x[1].get("wins", 0), reverse=True)[:5]:
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            total = wins + losses
            opening_duel_stats.append({
                "player": player_names.get(player_id, player_id),
                "opening_wins": wins,
                "opening_losses": losses,
                "opening_duel_rate": round((wins / total) * 100, 1) if total > 0 else 0.0,
            })

        # Weapon effectiveness summary
        top_weapons = []
        for weapon, kills in sorted(weapon_kills.items(), key=lambda x: x[1], reverse=True)[:10]:
            deaths = weapon_deaths.get(weapon, 0)
            top_weapons.append({
                "weapon": weapon,
                "kills": kills,
                "deaths": deaths,
                "kd_ratio": round(kills / deaths, 2) if deaths > 0 else kills,
            })

        return {
            "trade_efficiency": trade_efficiency,
            "trade_kills": trade_kills,
            "trade_opportunities": trade_opportunities,
            "multi_killers": top_multi_killers,
            "clutch_performers": clutch_performers,
            "opening_duels": opening_duel_stats,
            "weapon_effectiveness": top_weapons,
            "total_kills_analyzed": len(all_kills),
        }

    def _compute_economy_metrics(
        self,
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_team_side: Dict[Tuple[str, int, str], str],
        team_id: str,
    ) -> Dict:
        """Calculate advanced economy metrics from round and player data."""
        if self.df_players.empty:
            return {}
        
        # Get team players' economy data
        target_df = self._target_players()
        if target_df.empty:
            return {}
        
        # Economy by match
        economy_by_match: Dict[str, List[float]] = {}
        for _, row in target_df.iterrows():
            match_id = row.get("match_id")
            economy = row.get("economy")
            if match_id and pd.notna(economy):
                economy_by_match.setdefault(str(match_id), []).append(float(economy))
        
        # Calculate average team economy per match
        match_avg_economies = []
        for match_id, economies in economy_by_match.items():
            if economies:
                match_avg_economies.append(sum(economies) / len(economies))
        
        overall_avg_economy = round(sum(match_avg_economies) / len(match_avg_economies), 0) if match_avg_economies else 0
        
        # Classify matches by economy state
        eco_matches = sum(1 for e in match_avg_economies if e < 2000)
        force_matches = sum(1 for e in match_avg_economies if 2000 <= e < 3900)
        full_buy_matches = sum(1 for e in match_avg_economies if e >= 3900)
        
        total_matches = len(match_avg_economies) or 1
        
        return {
            "average_economy": overall_avg_economy,
            "eco_round_percentage": round((eco_matches / total_matches) * 100, 1),
            "force_round_percentage": round((force_matches / total_matches) * 100, 1),
            "full_buy_percentage": round((full_buy_matches / total_matches) * 100, 1),
            "economy_trend": "stable",  # Could be enhanced with more match data
        }

    def _compute_map_metrics(
        self,
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_team_side: Dict[Tuple[str, int, str], str],
        bomb_events: List[Dict],
        team_id: str,
    ) -> Dict:
        """Calculate map-specific metrics including site control."""
        # Site plant success by site and map
        site_plants: Dict[str, Dict[str, int]] = {}  # map -> site -> count
        site_wins: Dict[str, Dict[str, int]] = {}  # map -> site -> wins
        
        for event in bomb_events:
            if event.get("event_type") != "planted":
                continue
            if event.get("team_id") != team_id:
                continue
            
            map_name = event.get("map_name") or "unknown"
            site = event.get("site") or "unknown"
            round_key = (event.get("match_id"), event.get("round_number"))
            
            site_plants.setdefault(map_name, {}).setdefault(site, 0)
            site_plants[map_name][site] += 1
            
            if round_winner.get(round_key) == team_id:
                site_wins.setdefault(map_name, {}).setdefault(site, 0)
                site_wins[map_name][site] += 1
        
        # Calculate site success rates
        site_success_rates: Dict[str, Dict[str, float]] = {}
        for map_name, sites in site_plants.items():
            site_success_rates[map_name] = {}
            for site, plants in sites.items():
                wins = site_wins.get(map_name, {}).get(site, 0)
                site_success_rates[map_name][site] = round((wins / plants) * 100, 1) if plants > 0 else 0.0
        
        # Plant timing analysis
        plant_times: List[float] = []
        for event in bomb_events:
            if event.get("event_type") == "planted" and event.get("team_id") == team_id:
                time_in_round = event.get("time_in_round")
                if time_in_round is not None:
                    plant_times.append(time_in_round)
        
        avg_plant_time = round(sum(plant_times) / len(plant_times), 1) if plant_times else None
        
        # Classify plant speed
        if avg_plant_time is not None:
            if avg_plant_time < 30:
                plant_style = "Rush"
            elif avg_plant_time < 50:
                plant_style = "Fast Default"
            elif avg_plant_time < 70:
                plant_style = "Slow Default"
            else:
                plant_style = "Late Execute"
        else:
            plant_style = "Unknown"

        return {
            "site_success_rates": site_success_rates,
            "avg_plant_time": avg_plant_time,
            "plant_style": plant_style,
            "total_plants": len([e for e in bomb_events if e.get("event_type") == "planted" and e.get("team_id") == team_id]),
        }

    def _compute_side_metrics(
        self,
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_team_side: Dict[Tuple[str, int, str], str],
        team_id: str,
        all_kills: List[Dict],
    ) -> Dict:
        """Calculate attack vs defense performance splits."""
        attack_rounds = 0
        attack_wins = 0
        defense_rounds = 0
        defense_wins = 0
        
        for (match_id, round_number), winner_id in round_winner.items():
            side = round_team_side.get((match_id, round_number, team_id))
            if side == "attacker":
                attack_rounds += 1
                if winner_id == team_id:
                    attack_wins += 1
            elif side == "defender":
                defense_rounds += 1
                if winner_id == team_id:
                    defense_wins += 1
        
        # K/D by side
        attack_kills = 0
        attack_deaths = 0
        defense_kills = 0
        defense_deaths = 0
        
        for kill in all_kills:
            round_key = (kill.get("match_id"), kill.get("round_number"))
            side = round_team_side.get((round_key[0], round_key[1], team_id))
            
            if kill.get("killer_team_id") == team_id:
                if side == "attacker":
                    attack_kills += 1
                elif side == "defender":
                    defense_kills += 1
            
            if kill.get("victim_team_id") == team_id:
                if side == "attacker":
                    attack_deaths += 1
                elif side == "defender":
                    defense_deaths += 1

        return {
            "attack": {
                "rounds": attack_rounds,
                "wins": attack_wins,
                "win_rate": round((attack_wins / attack_rounds) * 100, 1) if attack_rounds > 0 else 0.0,
                "kills": attack_kills,
                "deaths": attack_deaths,
                "kd_ratio": round(attack_kills / attack_deaths, 2) if attack_deaths > 0 else attack_kills,
            },
            "defense": {
                "rounds": defense_rounds,
                "wins": defense_wins,
                "win_rate": round((defense_wins / defense_rounds) * 100, 1) if defense_rounds > 0 else 0.0,
                "kills": defense_kills,
                "deaths": defense_deaths,
                "kd_ratio": round(defense_kills / defense_deaths, 2) if defense_deaths > 0 else defense_kills,
            },
        }

    def _parse_ability_events(self, allowed_match_ids: set[str]) -> List[Dict]:
        """DEPRECATED: Use _parse_all_events_optimized instead. Kept for compatibility."""
        _, _, ability_events = self._parse_all_events_optimized(allowed_match_ids, {})
        return ability_events

    def _compute_pace_metrics(
        self,
        bomb_events: List[Dict],
        all_kills: List[Dict],
        round_start_times: Dict[Tuple[str, int], str],
        team_id: str,
        round_team_side: Dict[Tuple[str, int, str], str],
    ) -> Dict:
        """Calculate pace of play metrics - time to plant, time to first damage."""
        
        # Plant timing histogram
        plant_times: List[float] = []
        plant_times_by_map: Dict[str, List[float]] = {}
        
        for event in bomb_events:
            if event.get("event_type") != "planted":
                continue
            if event.get("team_id") != team_id:
                continue
            
            time_in_round = event.get("time_in_round")
            if time_in_round is not None and time_in_round > 0:
                plant_times.append(time_in_round)
                map_name = event.get("map_name") or "unknown"
                plant_times_by_map.setdefault(map_name, []).append(time_in_round)
        
        # Classify plant times
        total_plants = len(plant_times)
        if total_plants > 0:
            rush_plants = sum(1 for t in plant_times if t < 30)
            default_plants = sum(1 for t in plant_times if 30 <= t < 60)
            late_plants = sum(1 for t in plant_times if t >= 60)
            
            pace_histogram = {
                "rush": {"count": rush_plants, "percent": round((rush_plants / total_plants) * 100, 1)},
                "default": {"count": default_plants, "percent": round((default_plants / total_plants) * 100, 1)},
                "late": {"count": late_plants, "percent": round((late_plants / total_plants) * 100, 1)},
            }
            avg_plant_time = round(sum(plant_times) / total_plants, 1)
        else:
            pace_histogram = {"rush": {"count": 0, "percent": 0}, "default": {"count": 0, "percent": 0}, "late": {"count": 0, "percent": 0}}
            avg_plant_time = None
        
        # Calculate by map
        pace_by_map: Dict[str, Dict] = {}
        for map_name, times in plant_times_by_map.items():
            if not times:
                continue
            total = len(times)
            rush = sum(1 for t in times if t < 30)
            default = sum(1 for t in times if 30 <= t < 60)
            late = sum(1 for t in times if t >= 60)
            pace_by_map[map_name] = {
                "avg_time": round(sum(times) / total, 1),
                "rush_percent": round((rush / total) * 100, 1),
                "default_percent": round((default / total) * 100, 1),
                "late_percent": round((late / total) * 100, 1),
                "total_plants": total,
            }
        
        # Time to first damage by side
        first_damage_times_attack: List[float] = []
        first_damage_times_defense: List[float] = []
        
        # Group kills by round to find first damage
        kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}
        for kill in all_kills:
            key = (kill.get("match_id"), kill.get("round_number"))
            kills_by_round.setdefault(key, []).append(kill)
        
        for round_key, round_kills in kills_by_round.items():
            if not round_kills:
                continue
            
            # Sort by time
            sorted_kills = sorted(round_kills, key=lambda k: k.get("occurred_at") or "")
            first_kill = sorted_kills[0]
            time_in_round = first_kill.get("time_in_round")
            
            if time_in_round is None:
                continue
            
            # Determine our side this round
            side = round_team_side.get((round_key[0], round_key[1], team_id))
            
            # Track if we were involved in first damage
            our_kill = first_kill.get("killer_team_id") == team_id
            our_death = first_kill.get("victim_team_id") == team_id
            
            if our_kill or our_death:
                if side == "attacker":
                    first_damage_times_attack.append(time_in_round)
                elif side == "defender":
                    first_damage_times_defense.append(time_in_round)
        
        attack_first_damage_avg = round(sum(first_damage_times_attack) / len(first_damage_times_attack), 1) if first_damage_times_attack else None
        defense_first_damage_avg = round(sum(first_damage_times_defense) / len(first_damage_times_defense), 1) if first_damage_times_defense else None
        
        # Determine overall pace style
        if avg_plant_time is not None:
            if avg_plant_time < 30:
                pace_style = "Rush"
            elif avg_plant_time < 45:
                pace_style = "Fast Default"
            elif avg_plant_time < 60:
                pace_style = "Default"
            else:
                pace_style = "Slow/Late Execute"
        else:
            pace_style = "Unknown"
        
        return {
            "style": pace_style,
            "avg_plant_time": avg_plant_time,
            "histogram": pace_histogram,
            "by_map": pace_by_map,
            "attack_first_damage_avg": attack_first_damage_avg,
            "defense_first_damage_avg": defense_first_damage_avg,
            "total_plants_analyzed": total_plants,
        }

    def _compute_site_bias_metrics(
        self,
        bomb_events: List[Dict],
        round_winner: Dict[Tuple[str, int], Optional[str]],
        team_id: str,
    ) -> Dict:
        """Calculate site preference and win rate per site per map."""
        
        # Track plants and wins by map and site
        site_data: Dict[str, Dict[str, Dict[str, int]]] = {}  # map -> site -> {attempts, wins}
        
        for event in bomb_events:
            if event.get("event_type") != "planted":
                continue
            if event.get("team_id") != team_id:
                continue
            
            map_name = (event.get("map_name") or "unknown").lower()
            site = event.get("site") or "unknown"
            round_key = (event.get("match_id"), event.get("round_number"))
            
            site_data.setdefault(map_name, {}).setdefault(site, {"attempts": 0, "wins": 0})
            site_data[map_name][site]["attempts"] += 1
            
            if round_winner.get(round_key) == team_id:
                site_data[map_name][site]["wins"] += 1
        
        # Calculate win rates and format output
        site_bias: Dict[str, Dict[str, Dict]] = {}
        for map_name, sites in site_data.items():
            site_bias[map_name.title()] = {}
            total_plants = sum(s["attempts"] for s in sites.values())
            
            for site, stats in sites.items():
                attempts = stats["attempts"]
                wins = stats["wins"]
                site_bias[map_name.title()][site] = {
                    "attempts": attempts,
                    "wins": wins,
                    "win_rate": round((wins / attempts) * 100, 1) if attempts > 0 else 0.0,
                    "preference": round((attempts / total_plants) * 100, 1) if total_plants > 0 else 0.0,
                }
        
        # Generate insights
        insights: List[Dict] = []
        for map_name, sites in site_bias.items():
            sorted_sites = sorted(sites.items(), key=lambda x: x[1]["preference"], reverse=True)
            if len(sorted_sites) >= 2:
                top_site, top_data = sorted_sites[0]
                second_site, second_data = sorted_sites[1]
                
                # Check for actionable insight
                if top_data["win_rate"] < 40 and second_data["win_rate"] > 60:
                    insights.append({
                        "map": map_name,
                        "insight": f"On {map_name}, they plant {top_site} {top_data['preference']}% but only win {top_data['win_rate']}%. They plant {second_site} {second_data['preference']}% with {second_data['win_rate']}% WR. Let them go {top_site}, defend {second_site}.",
                        "type": "site_weakness",
                    })
        
        return {
            "by_map": site_bias,
            "insights": insights,
        }

    def _compute_first_death_context(
        self,
        all_kills: List[Dict],
        team_id: str,
        player_names: Dict[str, str],
        round_team_side: Dict[Tuple[str, int, str], str],
    ) -> Dict:
        """Analyze who dies first on the team, with agent/role and side context."""
        
        # Group kills by round
        kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}
        for kill in all_kills:
            key = (kill.get("match_id"), kill.get("round_number"))
            kills_by_round.setdefault(key, []).append(kill)
        
        # Track first deaths
        first_death_stats: Dict[str, Dict] = {}  # player_id -> stats
        
        for round_key, round_kills in kills_by_round.items():
            if not round_kills:
                continue
            
            sorted_kills = sorted(round_kills, key=lambda k: k.get("occurred_at") or "")
            
            # Find the first death on our team
            for kill in sorted_kills:
                if kill.get("victim_team_id") == team_id:
                    victim_id = kill.get("victim_id")
                    if not victim_id:
                        break
                    
                    side = round_team_side.get((round_key[0], round_key[1], team_id))
                    weapon = kill.get("weapon") or "unknown"
                    
                    first_death_stats.setdefault(victim_id, {
                        "total": 0,
                        "attack": 0,
                        "defense": 0,
                        "weapons": {},
                    })
                    first_death_stats[victim_id]["total"] += 1
                    if side == "attacker":
                        first_death_stats[victim_id]["attack"] += 1
                    elif side == "defender":
                        first_death_stats[victim_id]["defense"] += 1
                    
                    first_death_stats[victim_id]["weapons"][weapon] = first_death_stats[victim_id]["weapons"].get(weapon, 0) + 1
                    break
        
        # Get player agent info
        player_agents: Dict[str, str] = {}
        for match in self.matches:
            target_team_id = self._find_team_id(match)
            for player in match.get("players", []) or []:
                team = player.get("team") or {}
                if str(team.get("id")) == target_team_id:
                    pid = str(player.get("id"))
                    agent = (player.get("agent") or {}).get("name")
                    if agent:
                        player_agents[pid] = agent
        
        # Build output
        total_rounds = len(kills_by_round)
        first_death_breakdown = []
        
        for player_id, stats in sorted(first_death_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            player_name = player_names.get(player_id, player_id)
            agent = player_agents.get(player_id, "Unknown")
            role = AGENT_ROLE_MAP.get(agent.lower() if agent else "", "Unknown")
            
            top_weapon = max(stats["weapons"].items(), key=lambda x: x[1])[0] if stats["weapons"] else "unknown"
            
            first_death_breakdown.append({
                "player": player_name,
                "agent": agent,
                "role": role,
                "total": stats["total"],
                "rate": round((stats["total"] / total_rounds) * 100, 1) if total_rounds > 0 else 0.0,
                "attack_deaths": stats["attack"],
                "defense_deaths": stats["defense"],
                "top_weapon_died_to": top_weapon,
            })
        
        # Identify red flags (Sentinel dying first on defense is bad)
        red_flags: List[Dict] = []
        for entry in first_death_breakdown:
            if entry["role"] == "Sentinel" and entry["defense_deaths"] > 3:
                red_flags.append({
                    "player": entry["player"],
                    "issue": f"{entry['player']} ({entry['agent']}) dies first {entry['defense_deaths']} times on Defense. Over-aggressive peeking?",
                    "severity": "high" if entry["rate"] > 20 else "medium",
                })
            elif entry["role"] == "Controller" and entry["total"] > 5:
                red_flags.append({
                    "player": entry["player"],
                    "issue": f"{entry['player']} ({entry['agent']}) dying first often ({entry['total']} times). Controllers should survive longer.",
                    "severity": "medium",
                })
        
        return {
            "breakdown": first_death_breakdown[:10],  # Top 10
            "red_flags": red_flags,
            "total_rounds": total_rounds,
        }

    def _compute_ultimate_metrics(
        self,
        ability_events: List[Dict],
        round_winner: Dict[Tuple[str, int], Optional[str]],
        team_id: str,
        player_names: Dict[str, str],
    ) -> Dict:
        """Calculate ultimate usage and conversion rates."""
        
        # Filter to our team's ultimates only
        ult_events = [e for e in ability_events if e.get("is_ultimate") and e.get("team_id") == team_id]
        
        if not ult_events:
            return {"by_agent": [], "total_ults": 0, "overall_conversion": 0.0}
        
        # Track by agent
        agent_ult_stats: Dict[str, Dict] = {}  # agent -> {uses, wins}
        
        for event in ult_events:
            agent = event.get("agent") or "unknown"
            round_key = (event.get("match_id"), event.get("round_number"))
            
            agent_ult_stats.setdefault(agent, {"uses": 0, "wins": 0, "players": set()})
            agent_ult_stats[agent]["uses"] += 1
            
            player_name = event.get("player_name") or player_names.get(event.get("player_id"), "Unknown")
            agent_ult_stats[agent]["players"].add(player_name)
            
            if round_winner.get(round_key) == team_id:
                agent_ult_stats[agent]["wins"] += 1
        
        # Build output
        by_agent = []
        total_uses = 0
        total_wins = 0
        
        for agent, stats in sorted(agent_ult_stats.items(), key=lambda x: x[1]["uses"], reverse=True):
            uses = stats["uses"]
            wins = stats["wins"]
            total_uses += uses
            total_wins += wins
            
            by_agent.append({
                "agent": agent.title() if agent else "Unknown",
                "uses": uses,
                "wins": wins,
                "conversion_rate": round((wins / uses) * 100, 1) if uses > 0 else 0.0,
                "players": list(stats["players"]),
            })
        
        overall_conversion = round((total_wins / total_uses) * 100, 1) if total_uses > 0 else 0.0
        
        # Insights
        insights: List[Dict] = []
        for entry in by_agent:
            if entry["conversion_rate"] < 40 and entry["uses"] >= 3:
                insights.append({
                    "agent": entry["agent"],
                    "insight": f"When {entry['agent']} uses ultimate, team only wins {entry['conversion_rate']}% of rounds. Consider saving ult for better situations.",
                    "type": "low_conversion",
                })
            elif entry["conversion_rate"] > 70 and entry["uses"] >= 3:
                insights.append({
                    "agent": entry["agent"],
                    "insight": f"{entry['agent']}'s ult has {entry['conversion_rate']}% conversion rate. High-impact ult usage.",
                    "type": "high_impact",
                })
        
        return {
            "by_agent": by_agent,
            "total_ults": total_uses,
            "overall_conversion": overall_conversion,
            "insights": insights,
        }

    def _compute_man_advantage_metrics(
        self,
        all_kills: List[Dict],
        round_winner: Dict[Tuple[str, int], Optional[str]],
        team_id: str,
    ) -> Dict:
        """Calculate man-advantage (5v4) conversion rate."""
        
        # Group kills by round
        kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}
        for kill in all_kills:
            key = (kill.get("match_id"), kill.get("round_number"))
            kills_by_round.setdefault(key, []).append(kill)
        
        # Track 5v4 situations
        man_advantage_rounds = 0
        man_advantage_wins = 0
        
        for round_key, round_kills in kills_by_round.items():
            if not round_kills:
                continue
            
            sorted_kills = sorted(round_kills, key=lambda k: k.get("occurred_at") or "")
            
            # Check first kill
            first_kill = sorted_kills[0]
            
            # Did we get the first kill?
            if first_kill.get("killer_team_id") == team_id:
                man_advantage_rounds += 1
                if round_winner.get(round_key) == team_id:
                    man_advantage_wins += 1
        
        conversion_rate = round((man_advantage_wins / man_advantage_rounds) * 100, 1) if man_advantage_rounds > 0 else 0.0
        
        return {
            "situations": man_advantage_rounds,
            "wins": man_advantage_wins,
            "conversion_rate": conversion_rate,
            "is_strong": conversion_rate >= 70,
            "insight": f"{'Strong closer' if conversion_rate >= 70 else 'Struggles to close'}: {conversion_rate}% conversion when getting first kill (5v4). {'Team capitalizes well on advantages.' if conversion_rate >= 70 else 'They may over-peek after getting advantage.'}"
        }

    def _compute_discipline_metrics(
        self,
        all_kills: List[Dict],
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_team_side: Dict[Tuple[str, int, str], str],
        team_id: str,
    ) -> Dict:
        """Calculate discipline metrics: untraded deaths, eco threat, bonus round conversion."""
        
        # Group kills by round
        kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}
        for kill in all_kills:
            key = (kill.get("match_id"), kill.get("round_number"))
            kills_by_round.setdefault(key, []).append(kill)
        
        # Track untraded deaths
        total_deaths = 0
        untraded_deaths = 0
        
        for round_key, round_kills in kills_by_round.items():
            sorted_kills = sorted(round_kills, key=lambda k: k.get("occurred_at") or "")
            
            for i, kill in enumerate(sorted_kills):
                if kill.get("victim_team_id") != team_id:
                    continue
                
                total_deaths += 1
                death_time = _parse_iso_timestamp(kill.get("occurred_at"))
                if not death_time:
                    continue
                
                # Check if traded within 5 seconds
                traded = False
                for j in range(i + 1, min(i + 5, len(sorted_kills))):
                    next_kill = sorted_kills[j]
                    if next_kill.get("killer_team_id") == team_id:
                        next_time = _parse_iso_timestamp(next_kill.get("occurred_at"))
                        if next_time and (next_time - death_time).total_seconds() <= 5.0:
                            traded = True
                            break
                
                if not traded:
                    untraded_deaths += 1
        
        untraded_rate = round((untraded_deaths / total_deaths) * 100, 1) if total_deaths > 0 else 0.0
        
        # Eco round performance (rounds 2, 3, 14, 15 - simplified)
        eco_rounds = 0
        eco_wins = 0
        
        for round_key in round_winner.keys():
            round_number = round_key[1]
            if round_number in (2, 3, 14, 15):
                eco_rounds += 1
                if round_winner.get(round_key) == team_id:
                    eco_wins += 1
        
        eco_win_rate = round((eco_wins / eco_rounds) * 100, 1) if eco_rounds > 0 else 0.0
        
        # Bonus round conversion (round 3 and 16 after winning pistol+eco)
        # Simplified: just track rounds 3 and 16
        bonus_rounds = 0
        bonus_wins = 0
        
        for round_key in round_winner.keys():
            round_number = round_key[1]
            if round_number in (3, 16):
                # Check if we won the pistol (round 1 or 13)
                pistol_round = 1 if round_number == 3 else 13
                pistol_key = (round_key[0], pistol_round)
                
                if round_winner.get(pistol_key) == team_id:
                    bonus_rounds += 1
                    if round_winner.get(round_key) == team_id:
                        bonus_wins += 1
        
        bonus_conversion = round((bonus_wins / bonus_rounds) * 100, 1) if bonus_rounds > 0 else 0.0
        
        return {
            "untraded_deaths": untraded_deaths,
            "total_deaths": total_deaths,
            "untraded_rate": untraded_rate,
            "eco_rounds": eco_rounds,
            "eco_wins": eco_wins,
            "eco_win_rate": eco_win_rate,
            "eco_threat": eco_win_rate > 25,  # Above average eco threat
            "bonus_rounds": bonus_rounds,
            "bonus_wins": bonus_wins,
            "bonus_conversion": bonus_conversion,
            "insights": {
                "spacing": f"{'Good spacing' if untraded_rate < 40 else 'Poor spacing'}: {untraded_rate}% untraded deaths. {'Team supports well.' if untraded_rate < 40 else 'Isolate players for easy picks.'}",
                "eco": f"{'Dangerous' if eco_win_rate > 25 else 'Standard'} eco threat: {eco_win_rate}% eco win rate. {'Respect their low-buys.' if eco_win_rate > 25 else 'Push advantage on eco rounds.'}",
                "bonus": f"Bonus round conversion: {bonus_conversion}%. {'Closes out pistol advantages.' if bonus_conversion > 70 else 'Vulnerable in bonus rounds.'}"
            }
        }

    def _build_round_metadata(
        self,
    ) -> Tuple[Dict[Tuple[str, int], Optional[str]], Dict[Tuple[str, int, str], str]]:
        """Build round winner and team side mappings from matches and events with caching."""
        if self._round_metadata_cache is not None:
            return self._round_metadata_cache
        
        round_winner: Dict[Tuple[str, int], Optional[str]] = {}
        round_team_side: Dict[Tuple[str, int, str], str] = {}

        # First, get round winners from match segments
        for match in self.matches:
            match_id = str(match.get("id") or "")
            if not match_id:
                continue
            for segment in match.get("segments", []) or []:
                round_number = segment.get("segmentIndex")
                if not round_number:
                    continue
                winner = segment.get("winner") or {}
                winner_id = winner.get("id") if isinstance(winner, dict) else None
                if winner_id is not None:
                    winner_id = str(winner_id)
                round_winner[(match_id, round_number)] = winner_id
                # Try to get side from segment teams if available
                for team in segment.get("teams", []) or []:
                    team_id = team.get("id")
                    side = team.get("side")
                    if team_id and side:
                        round_team_side[(match_id, round_number, str(team_id))] = str(side)

        # Now extract side info from events (more reliable)
        for series_id, events in self.events_by_series.items():
            for envelope in events:
                for event in envelope.get("events") or []:
                    event_type = event.get("type")
                    
                    # Look for round-started events or team-won-round events that have team state
                    if event_type in ("game-started-round", "team-won-round", "game-ended-round"):
                        # Get game and round context
                        game_state = event.get("seriesState", {}).get("games", [])
                        for game in game_state:
                            game_id = str(game.get("id") or "")
                            if not game_id:
                                continue
                            round_state = game.get("currentRound") or game.get("segments", [{}])[-1] if game.get("segments") else {}
                            if isinstance(round_state, dict):
                                round_number = round_state.get("id") or round_state.get("segmentIndex")
                                if round_number:
                                    for team in round_state.get("teams", []) or []:
                                        team_id = team.get("id")
                                        side = team.get("side")
                                        if team_id and side and side in ("attacker", "defender"):
                                            round_team_side[(game_id, round_number, str(team_id))] = str(side)
                    
                    # Also check actor/target state for side info on kill events
                    if event_type == "player-killed-player":
                        actor = event.get("actor", {})
                        target = event.get("target", {})
                        actor_state = actor.get("state", {})
                        target_state = target.get("state", {})
                        
                        # Get game/round context from state
                        for state, prefix in [(actor_state, "actor"), (target_state, "target")]:
                            team_id = state.get("teamId")
                            side = state.get("side")
                            if team_id and side and side in ("attacker", "defender"):
                                # Try to get game_id and round from event context
                                game_id, round_number, _ = _event_round_context(event)
                                if game_id and round_number:
                                    round_team_side[(game_id, round_number, str(team_id))] = str(side)

        self._round_metadata_cache = (round_winner, round_team_side)
        return round_winner, round_team_side

    def _parse_first_kill_events(
        self, allowed_match_ids: set[str]
    ) -> Tuple[List[Dict], Dict[Tuple[str, int], str], set[Tuple[str, int]]]:
        first_kills: List[Dict] = []
        round_start_times: Dict[Tuple[str, int], str] = {}
        current_round: Dict[str, int] = {}
        event_rounds: set[Tuple[str, int]] = set()
        
        # Collect all kills first, then determine first kill per round
        all_kills_by_round: Dict[Tuple[str, int], List[Dict]] = {}

        for events in self.events_by_series.values():
            for envelope in events:
                occurred_at = envelope.get("occurredAt")
                for event in envelope.get("events") or []:
                    game_id, round_number, started_at = _event_round_context(event)
                    if game_id and allowed_match_ids and game_id not in allowed_match_ids:
                        continue

                    if game_id and round_number:
                        current_round[game_id] = round_number
                        event_rounds.add((game_id, round_number))
                        if started_at:
                            round_start_times[(game_id, round_number)] = started_at

                    if event.get("type") == "game-started-round":
                        if game_id and round_number:
                            round_start_times[(game_id, round_number)] = (
                                started_at or occurred_at
                            )

                    if event.get("type") != "player-killed-player":
                        continue

                    if not round_number and game_id and game_id in current_round:
                        round_number = current_round[game_id]
                    if not round_number:
                        continue

                    team_id = (
                        (event.get("actor") or {}).get("state", {}) or {}
                    ).get("teamId")
                    player_id = (event.get("actor") or {}).get("id")
                    weapon = _extract_weapon_from_event(event)

                    event_rounds.add((game_id, round_number))
                    
                    # Collect kill for later first-kill determination
                    round_key = (game_id, round_number)
                    if round_key not in all_kills_by_round:
                        all_kills_by_round[round_key] = []
                    all_kills_by_round[round_key].append({
                        "match_id": game_id,
                        "round_number": round_number,
                        "team_id": str(team_id) if team_id else None,
                        "player_id": str(player_id) if player_id else None,
                        "weapon": weapon,
                        "occurred_at": occurred_at,
                        "is_first_kill_flag": _is_first_kill(event),
                    })

        # Determine first kill per round (first kill with flag, or first chronologically)
        for round_key, kills in all_kills_by_round.items():
            # Sort by timestamp
            kills.sort(key=lambda k: k.get("occurred_at") or "")
            
            # Check if any has explicit firstKill flag
            flagged = [k for k in kills if k.get("is_first_kill_flag")]
            if flagged:
                first_kill = flagged[0]
            elif kills:
                first_kill = kills[0]
            else:
                continue
                
            time_to_first_kill = _compute_time_delta_seconds(
                first_kill.get("occurred_at"),
                round_start_times.get(round_key),
            )
            
            first_kills.append({
                "match_id": first_kill["match_id"],
                "round_number": first_kill["round_number"],
                "team_id": first_kill["team_id"],
                "player_id": first_kill["player_id"],
                "weapon": first_kill["weapon"],
                "occurred_at": first_kill["occurred_at"],
                "time_to_first_kill": time_to_first_kill,
            })

        return first_kills, round_start_times, event_rounds

    def _compute_pistol_round_metrics(
        self,
        team_id: str,
        round_winner: Dict[Tuple[str, int], Optional[str]],
        round_team_side: Dict[Tuple[str, int, str], str],
        team_first_kills: List[Dict],
        event_rounds: set[Tuple[str, int]],
    ) -> Dict:
        pistol_round_numbers = {1, 13}
        candidate_rounds = event_rounds or set(round_winner.keys())
        pistol_rounds = [key for key in candidate_rounds if key[1] in pistol_round_numbers]
        total = len(pistol_rounds)
        if total == 0:
            return {}

        wins = sum(1 for key in pistol_rounds if round_winner.get(key) == team_id)
        first_kill_rounds = sum(
            1
            for fk in team_first_kills
            if (fk.get("match_id"), fk.get("round_number")) in pistol_rounds
        )

        overall = {
            "rounds": total,
            "wins": wins,
            "win_rate": round((wins / total) * 100, 1),
            "first_kill_rate": round((first_kill_rounds / total) * 100, 1),
        }

        by_side = {}
        for side in ("attacker", "defender"):
            side_rounds = [
                key
                for key in pistol_rounds
                if round_team_side.get((key[0], key[1], team_id)) == side
            ]
            if not side_rounds:
                continue
            side_wins = sum(
                1 for key in side_rounds if round_winner.get(key) == team_id
            )
            side_first_kills = sum(
                1
                for fk in team_first_kills
                if (fk.get("match_id"), fk.get("round_number")) in side_rounds
            )
            by_side[side] = {
                "rounds": len(side_rounds),
                "wins": side_wins,
                "win_rate": round((side_wins / len(side_rounds)) * 100, 1),
                "first_kill_rate": round(
                    (side_first_kills / len(side_rounds)) * 100, 1
                ),
            }

        return {"overall": overall, "by_side": by_side}

    def _player_name_lookup(self) -> Dict[str, str]:
        """Get player ID to name mapping with caching."""
        if self._player_names_cache is not None:
            return self._player_names_cache
        names: Dict[str, str] = {}
        for match in self.matches:
            for player in match.get("players", []) or []:
                player_id = player.get("id")
                name = player.get("inGameName")
                if player_id and name:
                    names[str(player_id)] = str(name)
        self._player_names_cache = names
        return names

    def _site_preferences_from_events(self) -> Dict[str, float]:
        if not self.events_by_series or not SITE_CENTROIDS:
            return {}

        if not self.matches:
            return {}

        team_id = self._resolve_team_id()
        counts: Dict[str, int] = {}

        for envelopes in self.events_by_series.values():
            for envelope in envelopes:
                for event in envelope.get("events") or []:
                    event_type = str(event.get("type") or "").lower()
                    if "plantbomb" not in event_type:
                        continue

                    actor = event.get("actor") or {}
                    actor_state = actor.get("state") or {}
                    if team_id and actor_state.get("teamId") is not None:
                        if str(actor_state.get("teamId")) != str(team_id):
                            continue

                    game_state = actor_state.get("game") or {}
                    position = game_state.get("position") or actor_state.get("position")
                    if not position:
                        continue

                    game_id = game_state.get("id")
                    map_name = _map_name_from_event(event, game_id)
                    site = _infer_site_from_position(map_name, position)
                    if not site:
                        continue
                    counts[site] = counts.get(site, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return {}

        return {
            site: round((count / total) * 100, 1)
            for site, count in counts.items()
        }


def _parse_duration_seconds(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            pass
        if cleaned.startswith("PT"):
            try:
                return pd.to_timedelta(cleaned).total_seconds()
            except (ValueError, TypeError):
                return None
    return None


def _event_round_context(event: Dict) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    series_state_delta = event.get("seriesStateDelta") or {}
    games = series_state_delta.get("games") or []
    for game in games:
        game_id = str(game.get("id") or "")
        for segment in game.get("segments") or []:
            if str(segment.get("type")).lower() == "round":
                return (
                    game_id if game_id else None,
                    segment.get("sequenceNumber"),
                    segment.get("startedAt"),
                )
    return None, None, None


def _is_first_kill(event: Dict) -> bool:
    actor = event.get("actor") or {}
    state_delta = actor.get("stateDelta") or {}
    round_delta = state_delta.get("round") or {}
    if round_delta.get("firstKill") is True:
        return True
    state = actor.get("state") or {}
    round_state = state.get("round") or {}
    return round_state.get("firstKill") is True


def _extract_weapon_from_event(event: Dict) -> Optional[str]:
    actor = event.get("actor") or {}
    state_delta = actor.get("stateDelta") or {}
    round_delta = state_delta.get("round") or {}
    weapon_kills = round_delta.get("weaponKills")
    if isinstance(weapon_kills, dict) and weapon_kills:
        return next(iter(weapon_kills.keys()))
    state = actor.get("state") or {}
    round_state = state.get("round") or {}
    weapon_kills = round_state.get("weaponKills")
    if isinstance(weapon_kills, dict) and weapon_kills:
        return next(iter(weapon_kills.keys()))
    return None


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_time_delta_seconds(
    occurred_at: Optional[str], started_at: Optional[str]
) -> Optional[float]:
    if not occurred_at or not started_at:
        return None
    start_ts = _parse_iso_timestamp(started_at)
    end_ts = _parse_iso_timestamp(occurred_at)
    if not start_ts or not end_ts:
        return None
    delta = end_ts - start_ts
    return round(delta.total_seconds(), 1)


def _load_site_centroids() -> Dict[str, Dict[str, Dict[str, float]]]:
    path = Path(__file__).resolve().parents[1] / "data" / "valorant_map_sites.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    cleaned: Dict[str, Dict[str, Dict[str, float]]] = {}
    for map_name, sites in payload.items():
        if not isinstance(sites, dict):
            continue
        map_key = str(map_name).strip().lower()
        for site, coords in sites.items():
            if not isinstance(coords, dict):
                continue
            x = coords.get("x")
            y = coords.get("y")
            if x is None or y is None:
                continue
            cleaned.setdefault(map_key, {})[str(site)] = {"x": float(x), "y": float(y)}
    return cleaned


SITE_CENTROIDS = _load_site_centroids()


def _map_name_from_event(event: Dict, game_id: Optional[str]) -> Optional[str]:
    series_state = event.get("seriesState") or event.get("seriesStateDelta") or {}
    games = series_state.get("games") or []
    if game_id:
        for game in games:
            if str(game.get("id")) == str(game_id):
                return (game.get("map") or {}).get("name")
    if len(games) == 1:
        return (games[0].get("map") or {}).get("name")
    return None


def _infer_site_from_position(
    map_name: Optional[str], position: Dict[str, float]
) -> Optional[str]:
    if not map_name or not position:
        return None
    map_key = str(map_name).strip().lower()
    sites = SITE_CENTROIDS.get(map_key)
    if not sites:
        return None
    x = position.get("x")
    y = position.get("y")
    if x is None or y is None:
        return None
    best_site = None
    best_dist = None
    for site, coords in sites.items():
        dx = x - coords.get("x", 0.0)
        dy = y - coords.get("y", 0.0)
        dist = dx * dx + dy * dy
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_site = site
    return best_site


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    invalid = cleaned.str.lower().isin(["", "none", "nan"])
    return cleaned.mask(invalid)
