from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScoutRequest(BaseModel):
    team_name: str = Field(..., min_length=1)
    match_limit: int = Field(50, ge=1, le=100)
    map_filter: Optional[str] = None
    game_title: str = Field("VALORANT", min_length=1)


class AgentPick(BaseModel):
    agent: str
    pick_count: int
    pick_rate: float


class AggressionIndex(BaseModel):
    style: str
    avg_duration: float
    rush_rate: float


class PlayerTendency(BaseModel):
    player: str
    matches_played: int
    top_agent: Optional[str] = None
    top_agent_rate: float = 0.0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    avg_assists: float = 0.0
    kd_ratio: Optional[float] = None
    avg_acs: Optional[float] = None
    first_kill_rate: Optional[float] = None


class CompositionSummary(BaseModel):
    composition: List[str]
    matches: int
    pick_rate: float


class CompositionBreakdown(BaseModel):
    overall: List[CompositionSummary]
    by_map: Dict[str, List[CompositionSummary]]
    most_recent: List[str]


class MetricsSummary(BaseModel):
    win_rate: float
    win_rate_by_map: Dict[str, float]
    site_preferences: Dict[str, float]
    pistol_site_preferences: Dict[str, float]
    aggression: AggressionIndex
    agent_composition: List[AgentPick]
    player_tendencies: List[PlayerTendency]
    role_distribution: Dict[str, float]
    recent_compositions: CompositionBreakdown
    economy: Dict[str, float]
    first_duel: Optional[Dict] = None
    pistol_rounds: Optional[Dict] = None
    # Extended metrics from analyzer
    combat_metrics: Optional[Dict] = None
    side_metrics: Optional[Dict] = None
    map_metrics: Optional[Dict] = None
    economy_metrics: Optional[Dict] = None
    opponent_stats: Optional[List[Dict]] = None
    map_detailed: Optional[Dict] = None
    round_type_performance: Optional[Dict] = None


class ScoutReport(BaseModel):
    team_name: str
    matches_analyzed: int
    metrics: MetricsSummary
    insights: Dict[str, str] = Field(default_factory=dict)
