"use client";

import { StatCard, SectionCard, DataTable, ProgressBar } from "./StatCards";
import type { MetricsSummary } from "../types";

interface PlayersTabProps {
  metrics: MetricsSummary;
}

// Agent role mapping (lowercase keys for case-insensitive lookup)
const AGENT_ROLES: Record<string, string> = {
  jett: "Duelist",
  raze: "Duelist",
  reyna: "Duelist",
  phoenix: "Duelist",
  neon: "Duelist",
  yoru: "Duelist",
  iso: "Duelist",
  waylay: "Duelist",
  sova: "Initiator",
  breach: "Initiator",
  skye: "Initiator",
  "kay/o": "Initiator",
  fade: "Initiator",
  gekko: "Initiator",
  tejo: "Initiator",
  omen: "Controller",
  brimstone: "Controller",
  viper: "Controller",
  astra: "Controller",
  harbor: "Controller",
  clove: "Controller",
  sage: "Sentinel",
  cypher: "Sentinel",
  killjoy: "Sentinel",
  chamber: "Sentinel",
  deadlock: "Sentinel",
  vyse: "Sentinel",
  veto: "Sentinel",
};

// Helper to get role from agent name (case-insensitive)
function getAgentRole(agent: string | null | undefined): string {
  if (!agent) return "Unknown";
  return AGENT_ROLES[agent.toLowerCase()] ?? "Unknown";
}

function getPlayerGrade(
  kd: number,
  firstKillRate: number,
): { grade: string; color: string } {
  const score = kd * 40 + (firstKillRate / 100) * 60;
  if (score >= 70) return { grade: "S", color: "text-yellow" };
  if (score >= 55) return { grade: "A", color: "text-green" };
  if (score >= 40) return { grade: "B", color: "text-cyan" };
  if (score >= 25) return { grade: "C", color: "text-muted" };
  return { grade: "D", color: "text-red" };
}

export function PlayersTab({ metrics }: PlayersTabProps) {
  const players = metrics.player_tendencies ?? [];

  // Sort players by KD ratio
  const sortedPlayers = [...players].sort(
    (a, b) => (b.kd_ratio ?? 0) - (a.kd_ratio ?? 0),
  );

  // Top 3 players for cards
  const topPlayers = sortedPlayers.slice(0, 3).map((p) => ({
    label: p.player,
    value: `KD ${p.kd_ratio?.toFixed(2) ?? 0}`,
    hint: `${p.top_agent ?? "Unknown"} (${p.top_agent_rate?.toFixed(0) ?? 0}%)`,
    percent: Math.min(100, (p.kd_ratio ?? 0) * 50),
    accent: (p.kd_ratio ?? 0) >= 1 ? ("green" as const) : ("red" as const),
  }));

  // Full player table
  const playerRows = sortedPlayers.map((p) => {
    const grade = getPlayerGrade(p.kd_ratio ?? 0, p.first_kill_rate ?? 0);
    return [
      p.player,
      p.top_agent ?? "—",
      getAgentRole(p.top_agent),
      p.kd_ratio?.toFixed(2) ?? "—",
      `${p.first_kill_rate?.toFixed(1) ?? 0}%`,
      grade.grade,
    ];
  });

  return (
    <div className="space-y-6">
      {/* Top Players Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {topPlayers.map((stat, i) => (
          <StatCard
            key={stat.label}
            {...stat}
            className={i === 0 ? "border-yellow/50" : undefined}
          />
        ))}
      </div>

      {/* Player Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sortedPlayers.map((player) => {
          const grade = getPlayerGrade(
            player.kd_ratio ?? 0,
            player.first_kill_rate ?? 0,
          );
          const role = getAgentRole(player.top_agent);

          return (
            <div
              key={player.player}
              className="cut-corner border border-ethereal bg-glass p-5"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg text-white">{player.player}</h3>
                  <p className="text-xs text-muted uppercase tracking-[0.2em]">
                    {role}
                  </p>
                </div>
                <div className={`text-3xl font-bold ${grade.color}`}>
                  {grade.grade}
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between text-xs text-muted mb-1">
                    <span>Main Agent</span>
                    <span className="text-white">
                      {player.top_agent ?? "—"}
                    </span>
                  </div>
                  <ProgressBar
                    value={player.top_agent_rate ?? 0}
                    accent="cyan"
                    showValue={false}
                  />
                  <div className="text-right text-xs text-muted mt-1">
                    {player.top_agent_rate?.toFixed(0) ?? 0}% pick rate
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-2 border-t border-ethereal">
                  <div className="text-center">
                    <div
                      className={`text-xl font-bold ${(player.kd_ratio ?? 0) >= 1 ? "text-green" : "text-red"}`}
                    >
                      {player.kd_ratio?.toFixed(2) ?? "—"}
                    </div>
                    <div className="text-xs text-muted">K/D Ratio</div>
                  </div>
                  <div className="text-center">
                    <div
                      className={`text-xl font-bold ${(player.first_kill_rate ?? 0) >= 15 ? "text-cyan" : "text-muted"}`}
                    >
                      {player.first_kill_rate?.toFixed(1) ?? 0}%
                    </div>
                    <div className="text-xs text-muted">First Kill %</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Full Player Table */}
      <SectionCard title="Player Statistics">
        <DataTable
          headers={["Player", "Main Agent", "Role", "K/D", "FK Rate", "Grade"]}
          rows={playerRows}
        />
      </SectionCard>

      {/* Player Insights */}
      <SectionCard title="Player Analysis">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Star Player
            </h4>
            <p className="text-sm text-muted">
              {sortedPlayers.length > 0
                ? `${sortedPlayers[0].player} leads with ${sortedPlayers[0].kd_ratio?.toFixed(2) ?? 0} K/D. Focus defensive utility and crossfires against them.`
                : "No player data available"}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Entry Fragger
            </h4>
            <p className="text-sm text-muted">
              {(() => {
                const entryPlayer = [...players].sort(
                  (a, b) => (b.first_kill_rate ?? 0) - (a.first_kill_rate ?? 0),
                )[0];
                return entryPlayer
                  ? `${entryPlayer.player} has ${entryPlayer.first_kill_rate?.toFixed(1) ?? 0}% first kill rate. Primary entry threat.`
                  : "No entry data available";
              })()}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Role Distribution
            </h4>
            <p className="text-sm text-muted">
              {(() => {
                const roles = players
                  .map((p) => AGENT_ROLES[p.top_agent ?? ""])
                  .filter(Boolean);
                const roleCounts = roles.reduce<Record<string, number>>(
                  (acc, role) => {
                    acc[role] = (acc[role] || 0) + 1;
                    return acc;
                  },
                  {},
                );
                const entries = Object.entries(roleCounts);
                if (entries.length === 0) return "No role data";
                const top = entries.sort((a, b) => b[1] - a[1])[0];
                return `${top[1]} ${top[0]}${top[1] > 1 ? "s" : ""} on roster. ${
                  top[0] === "Duelist"
                    ? "Aggressive team composition."
                    : top[0] === "Sentinel"
                      ? "Defense-oriented lineup."
                      : "Balanced team structure."
                }`;
              })()}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Weak Link
            </h4>
            <p className="text-sm text-muted">
              {sortedPlayers.length > 0
                ? `${sortedPlayers[sortedPlayers.length - 1].player} is statistically weakest (${
                    sortedPlayers[sortedPlayers.length - 1].kd_ratio?.toFixed(
                      2,
                    ) ?? 0
                  } K/D). Target their positions for advantages.`
                : "No player data available"}
            </p>
          </div>
        </div>
      </SectionCard>

      {/* Player Comparison */}
      <SectionCard title="Performance Comparison">
        <div className="space-y-4">
          <div className="grid grid-cols-6 gap-2 text-xs uppercase tracking-[0.2em] text-muted border-b border-ethereal pb-2">
            <div>Player</div>
            <div className="col-span-5">K/D Performance</div>
          </div>
          {sortedPlayers.map((player) => {
            const maxKD = Math.max(...players.map((p) => p.kd_ratio ?? 0), 2);
            const percent = ((player.kd_ratio ?? 0) / maxKD) * 100;
            return (
              <div
                key={player.player}
                className="grid grid-cols-6 gap-2 items-center"
              >
                <div className="text-sm text-white truncate">
                  {player.player}
                </div>
                <div className="col-span-5">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-4 bg-[rgba(255,255,255,0.08)] rounded overflow-hidden">
                      <div
                        className={`h-full rounded ${
                          (player.kd_ratio ?? 0) >= 1.2
                            ? "bg-[rgba(34,197,94,0.5)]"
                            : (player.kd_ratio ?? 0) >= 1
                              ? "bg-[rgba(6,182,212,0.5)]"
                              : "bg-[rgba(255,70,85,0.5)]"
                        }`}
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted w-12 text-right">
                      {player.kd_ratio?.toFixed(2) ?? "—"}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </SectionCard>
    </div>
  );
}
