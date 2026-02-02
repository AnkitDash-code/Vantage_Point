// Shared types for dashboard metrics

export type ProgressState = {
  stage: string;
  progress: number;
  message?: string;
};

export type MapDetailedStats = {
  rounds_played: number;
  rounds_won: number;
  win_rate: number;
  attack_rounds: number;
  attack_wins: number;
  attack_win_rate: number;
  defense_rounds: number;
  defense_wins: number;
  defense_win_rate: number;
  top_agent?: string;
};

export type RoundTypePerformance = {
  rounds: number;
  wins: number;
  win_rate: number;
};

export type OpponentStat = {
  opponent: string;
  matches: number;
  rounds_played: number;
  rounds_won: number;
  win_rate: number;
};

export type MultiKiller = {
  player: string;
  "2k": number;
  "3k": number;
  "4k": number;
  "5k": number;
  total: number;
};

export type ClutchPerformer = {
  player: string;
  clutches_faced: number;
  clutches_won: number;
  clutch_rate: number;
};

export type OpeningDuel = {
  player: string;
  opening_wins: number;
  opening_losses: number;
  opening_duel_rate: number;
};

export type WeaponEffectiveness = {
  weapon: string;
  kills: number;
  deaths: number;
  kd_ratio: number;
};

export type CombatMetrics = {
  trade_kills?: number;
  trade_deaths?: number;
  trade_efficiency?: number;
  trade_opportunities?: number;
  multi_kills_2k?: number;
  multi_kills_3k?: number;
  multi_kills_4k?: number;
  multi_kills_5k?: number;
  multi_killers?: MultiKiller[];
  clutch_attempts?: number;
  clutch_wins?: number;
  clutch_win_rate?: number;
  clutch_performers?: ClutchPerformer[];
  opening_duels_won?: number;
  opening_duels_lost?: number;
  opening_duel_win_rate?: number;
  opening_duels?: OpeningDuel[];
  weapon_kills?: Record<string, number>;
  weapon_effectiveness?: WeaponEffectiveness[];
  total_kills_analyzed?: number;
};

export type SideMetrics = {
  attack_rounds?: number;
  attack_wins?: number;
  attack_win_rate?: number;
  attack_kills?: number;
  attack_deaths?: number;
  attack_kd?: number;
  defense_rounds?: number;
  defense_wins?: number;
  defense_win_rate?: number;
  defense_kills?: number;
  defense_deaths?: number;
  defense_kd?: number;
};

// New Advanced Metrics Types

export type PaceHistogramEntry = {
  count: number;
  percent: number;
};

export type PaceByMap = {
  avg_time: number;
  rush_percent: number;
  default_percent: number;
  late_percent: number;
  total_plants: number;
};

export type PaceMetrics = {
  style: string;
  avg_plant_time: number | null;
  histogram: {
    rush: PaceHistogramEntry;
    default: PaceHistogramEntry;
    late: PaceHistogramEntry;
  };
  by_map: Record<string, PaceByMap>;
  attack_first_damage_avg: number | null;
  defense_first_damage_avg: number | null;
  total_plants_analyzed: number;
};

export type SiteStats = {
  attempts: number;
  wins: number;
  win_rate: number;
  preference: number;
};

export type SiteBiasInsight = {
  map: string;
  insight: string;
  type: string;
};

export type SiteBiasMetrics = {
  by_map: Record<string, Record<string, SiteStats>>;
  insights: SiteBiasInsight[];
};

export type FirstDeathEntry = {
  player: string;
  agent: string;
  role: string;
  total: number;
  rate: number;
  attack_deaths: number;
  defense_deaths: number;
  top_weapon_died_to: string;
};

export type FirstDeathRedFlag = {
  player: string;
  issue: string;
  severity: "high" | "medium" | "low";
};

export type FirstDeathContext = {
  breakdown: FirstDeathEntry[];
  red_flags: FirstDeathRedFlag[];
  total_rounds: number;
};

export type UltimateAgentEntry = {
  agent: string;
  uses: number;
  wins: number;
  conversion_rate: number;
  players: string[];
};

export type UltimateInsight = {
  agent: string;
  insight: string;
  type: string;
};

export type UltimateImpact = {
  by_agent: UltimateAgentEntry[];
  total_ults: number;
  overall_conversion: number;
  insights: UltimateInsight[];
};

export type ManAdvantage = {
  situations: number;
  wins: number;
  conversion_rate: number;
  is_strong: boolean;
  insight: string;
};

export type DisciplineInsights = {
  spacing: string;
  eco: string;
  bonus: string;
};

export type DisciplineMetrics = {
  untraded_deaths: number;
  total_deaths: number;
  untraded_rate: number;
  eco_rounds: number;
  eco_wins: number;
  eco_win_rate: number;
  eco_threat: boolean;
  bonus_rounds: number;
  bonus_wins: number;
  bonus_conversion: number;
  insights: DisciplineInsights;
};

export type MapMetrics = {
  site_attack_success?: Record<string, Record<string, { attempts: number; wins: number; win_rate: number }>>;
  site_defense_success?: Record<string, Record<string, { attempts: number; wins: number; win_rate: number }>>;
  plant_timing?: Record<string, { avg_time: number; early: number; mid: number; late: number }>;
  map_site_preferences?: Record<string, Record<string, number>>;
};

export type EconomyMetrics = {
  economy_state_performance?: Record<string, { rounds: number; wins: number; win_rate: number }>;
  avg_team_economy?: number;
  economy_advantage_rate?: number;
};

export type MetricsSummary = {
  win_rate: number;
  win_rate_by_map: Record<string, number>;
  site_preferences: Record<string, number>;
  pistol_site_preferences: Record<string, number>;
  aggression: { style: string; avg_duration: number; rush_rate: number };
  agent_composition: Array<{ agent: string; pick_rate: number }>;
  role_distribution: Record<string, number>;
  economy: Record<string, number>;
  player_tendencies: Array<{
    player: string;
    top_agent?: string;
    top_agent_rate?: number;
    kd_ratio?: number;
    avg_acs?: number;
    first_kill_rate?: number;
  }>;
  recent_compositions: {
    overall: Array<{ composition: string[]; pick_rate: number }>;
    most_recent: string[];
  };
  first_duel?: {
    team_first_kill_rate?: number;
    first_kill_conversion_rate?: number;
  };
  // New expanded metrics
  opponent_stats?: OpponentStat[];
  map_detailed?: Record<string, MapDetailedStats>;
  round_type_performance?: {
    pistol: RoundTypePerformance;
    eco: RoundTypePerformance;
    force: RoundTypePerformance;
    full_buy: RoundTypePerformance;
  };
  combat_metrics?: CombatMetrics;
  side_metrics?: SideMetrics;
  map_metrics?: MapMetrics;
  economy_metrics?: EconomyMetrics;
  // Advanced scouting metrics
  pace_metrics?: PaceMetrics;
  site_bias?: SiteBiasMetrics;
  first_death_context?: FirstDeathContext;
  ultimate_impact?: UltimateImpact;
  man_advantage?: ManAdvantage;
  discipline?: DisciplineMetrics;
};

export type ScoutReport = {
  team_name: string;
  matches_analyzed: number;
  metrics: MetricsSummary;
  insights: Record<string, string>;
};

export type StreamProgress = {
  type: "progress";
  stage: string;
  progress: number;
  message?: string;
};

export type StreamWarning = {
  type: "warning";
  message: string;
};

export type StreamMetrics = {
  type: "metrics";
  team_name: string;
  matches_analyzed: number;
  metrics: MetricsSummary;
};

export type StreamDone = {
  type: "done";
  report: ScoutReport;
};

export type StreamInsightChunk = {
  type: "insight_chunk";
  section: string;
  content: string;
};

export type StreamError = {
  type: "error";
  message: string;
};
