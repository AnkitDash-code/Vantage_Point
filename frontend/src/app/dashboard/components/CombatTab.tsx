"use client";

import { StatCard, SectionCard, DataTable, ComparisonBar } from "./StatCards";
import type { MetricsSummary } from "../types";
import { getFallbackImage, getWeaponImage } from "./imageMaps";

interface CombatTabProps {
  metrics: MetricsSummary;
}

export function CombatTab({ metrics }: CombatTabProps) {
  const combat = metrics.combat_metrics ?? {};
  const side = metrics.side_metrics ?? {};
  const firstDuel = metrics.first_duel ?? {};
  const firstDeathContext = metrics.first_death_context;
  const manAdvantage = metrics.man_advantage;
  const discipline = metrics.discipline;
  const ultimateImpact = metrics.ultimate_impact;

  // Aggregate multi-kills from player data
  const multiKillers = combat.multi_killers ?? [];
  const totalMulti2k = multiKillers.reduce((sum, p) => sum + (p["2k"] ?? 0), 0);
  const totalMulti3k = multiKillers.reduce((sum, p) => sum + (p["3k"] ?? 0), 0);
  const totalMulti4k = multiKillers.reduce((sum, p) => sum + (p["4k"] ?? 0), 0);
  const totalMulti5k = multiKillers.reduce((sum, p) => sum + (p["5k"] ?? 0), 0);

  // Aggregate clutches
  const clutchPerformers = combat.clutch_performers ?? [];
  const totalClutchAttempts = clutchPerformers.reduce(
    (sum, p) => sum + (p.clutches_faced ?? 0),
    0,
  );
  const totalClutchWins = clutchPerformers.reduce(
    (sum, p) => sum + (p.clutches_won ?? 0),
    0,
  );
  const clutchWinRate =
    totalClutchAttempts > 0 ? (totalClutchWins / totalClutchAttempts) * 100 : 0;

  // Aggregate opening duels
  const openingDuels = combat.opening_duels ?? [];
  const totalOpeningWins = openingDuels.reduce(
    (sum, p) => sum + (p.opening_wins ?? 0),
    0,
  );
  const totalOpeningLosses = openingDuels.reduce(
    (sum, p) => sum + (p.opening_losses ?? 0),
    0,
  );
  const openingDuelWinRate =
    totalOpeningWins + totalOpeningLosses > 0
      ? (totalOpeningWins / (totalOpeningWins + totalOpeningLosses)) * 100
      : 0;

  // Combat stats cards
  const combatStats = [
    {
      label: "First Kill Rate",
      value: `${firstDuel.team_first_kill_rate?.toFixed(1) ?? 0}%`,
      hint: `Conversion: ${firstDuel.first_kill_conversion_rate?.toFixed(1) ?? 0}%`,
      percent: firstDuel.team_first_kill_rate ?? 0,
      accent: "cyan" as const,
    },
    {
      label: "Trade Efficiency",
      value: `${combat.trade_efficiency?.toFixed(1) ?? 0}%`,
      hint: `${combat.trade_kills ?? 0} trades / ${combat.trade_opportunities ?? 0} opportunities`,
      percent: combat.trade_efficiency ?? 0,
      accent: "green" as const,
    },
    {
      label: "Man Advantage Conv.",
      value: `${manAdvantage?.conversion_rate?.toFixed(1) ?? 0}%`,
      hint: `${manAdvantage?.wins ?? 0}W / ${manAdvantage?.situations ?? 0} 5v4s`,
      percent: manAdvantage?.conversion_rate ?? 0,
      accent: manAdvantage?.is_strong ? ("green" as const) : ("red" as const),
    },
    {
      label: "Clutch Win Rate",
      value: `${clutchWinRate.toFixed(1)}%`,
      hint: `${totalClutchWins} / ${totalClutchAttempts} attempts`,
      percent: clutchWinRate,
      accent: "violet" as const,
    },
  ];

  // Multi-kills breakdown
  const multiKills = [
    { label: "2K", value: totalMulti2k },
    { label: "3K", value: totalMulti3k },
    { label: "4K", value: totalMulti4k },
    { label: "Ace", value: totalMulti5k },
  ];
  const totalMultiKills = multiKills.reduce((sum, mk) => sum + mk.value, 0);

  // Weapon effectiveness (from array)
  const weaponEffectiveness = combat.weapon_effectiveness ?? [];
  const weaponRows = weaponEffectiveness.map((w) => [
    <div key={w.weapon} className="flex items-center gap-2">
      <img
        src={getWeaponImage(w.weapon)}
        alt={`${w.weapon} icon`}
        className="h-6 w-6 rounded object-cover border border-ethereal/60"
        loading="lazy"
        onError={(event) => {
          event.currentTarget.src = getFallbackImage(w.weapon, "ff4655");
        }}
      />
      <span>{w.weapon}</span>
    </div>,
    w.kills,
    w.kd_ratio?.toFixed(2) ?? "N/A",
  ]);

  return (
    <div className="space-y-6">
      {/* Combat Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {combatStats.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      {/* Discipline Metrics - NEW */}
      {discipline && (
        <SectionCard title="Team Discipline">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
                Spacing
              </h4>
              <div className="text-2xl font-bold text-white mb-1">
                {discipline.untraded_rate.toFixed(1)}%
              </div>
              <div className="text-xs text-muted">Untraded Deaths</div>
              <div className="mt-2 text-xs text-muted">
                {discipline.untraded_deaths} of {discipline.total_deaths} deaths
              </div>
              <div
                className={`mt-2 text-xs ${discipline.untraded_rate < 40 ? "text-green-400" : "text-red-400"}`}
              >
                {discipline.untraded_rate < 40
                  ? "✓ Good spacing"
                  : "⚠ Poor spacing - isolate players"}
              </div>
            </div>
            <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
                Eco Threat
              </h4>
              <div className="text-2xl font-bold text-white mb-1">
                {discipline.eco_win_rate.toFixed(1)}%
              </div>
              <div className="text-xs text-muted">Eco Round Win Rate</div>
              <div className="mt-2 text-xs text-muted">
                {discipline.eco_wins}W / {discipline.eco_rounds} eco rounds
              </div>
              <div
                className={`mt-2 text-xs ${discipline.eco_threat ? "text-amber-400" : "text-green-400"}`}
              >
                {discipline.eco_threat
                  ? "⚠ Dangerous low-buys - respect them"
                  : "✓ Standard eco threat"}
              </div>
            </div>
            <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
                Bonus Round Conv.
              </h4>
              <div className="text-2xl font-bold text-white mb-1">
                {discipline.bonus_conversion.toFixed(1)}%
              </div>
              <div className="text-xs text-muted">After Pistol Win</div>
              <div className="mt-2 text-xs text-muted">
                {discipline.bonus_wins}W / {discipline.bonus_rounds} bonus
                rounds
              </div>
              <div
                className={`mt-2 text-xs ${discipline.bonus_conversion >= 70 ? "text-green-400" : "text-red-400"}`}
              >
                {discipline.bonus_conversion >= 70
                  ? "✓ Closes out pistol wins"
                  : "⚠ Vulnerable in bonus rounds"}
              </div>
            </div>
          </div>
        </SectionCard>
      )}

      {/* First Death Context - NEW */}
      {firstDeathContext && firstDeathContext.breakdown.length > 0 && (
        <SectionCard title="First Death Analysis">
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <DataTable
                headers={[
                  "Player",
                  "Agent",
                  "Role",
                  "Deaths",
                  "Rate",
                  "ATK",
                  "DEF",
                ]}
                rows={firstDeathContext.breakdown.slice(0, 5).map((p) => [
                  p.player,
                  p.agent,
                  <span
                    key={p.player}
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      p.role === "Duelist"
                        ? "bg-red-500/20 text-red-400"
                        : p.role === "Sentinel"
                          ? "bg-green-500/20 text-green-400"
                          : p.role === "Controller"
                            ? "bg-purple-500/20 text-purple-400"
                            : "bg-cyan-500/20 text-cyan"
                    }`}
                  >
                    {p.role}
                  </span>,
                  p.total,
                  `${p.rate.toFixed(1)}%`,
                  p.attack_deaths,
                  p.defense_deaths,
                ])}
              />
            </div>

            {/* Red Flags */}
            {firstDeathContext.red_flags.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs uppercase tracking-[0.2em] text-red-400">
                  Red Flags
                </h4>
                {firstDeathContext.red_flags.map((flag, i) => (
                  <div
                    key={i}
                    className={`cut-corner p-3 text-sm ${
                      flag.severity === "high"
                        ? "bg-red-500/10 border border-red-500/30 text-red-300"
                        : "bg-amber-500/10 border border-amber-500/30 text-amber-300"
                    }`}
                  >
                    {flag.issue}
                  </div>
                ))}
              </div>
            )}
          </div>
        </SectionCard>
      )}

      {/* Ultimate Impact - NEW */}
      {ultimateImpact && ultimateImpact.by_agent.length > 0 && (
        <SectionCard title="Ultimate Impact">
          <div className="mb-4 flex items-center gap-4">
            <div className="text-sm text-muted">
              Overall Ult Conversion:
              <span
                className={`ml-2 text-lg font-bold ${ultimateImpact.overall_conversion >= 50 ? "text-green-400" : "text-red-400"}`}
              >
                {ultimateImpact.overall_conversion.toFixed(1)}%
              </span>
            </div>
            <div className="text-sm text-muted">
              Total Ults:{" "}
              <span className="text-white">{ultimateImpact.total_ults}</span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {ultimateImpact.by_agent.map((ult) => (
              <div
                key={ult.agent}
                className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-3"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-white">{ult.agent}</span>
                  <span
                    className={`text-lg font-bold ${ult.conversion_rate >= 50 ? "text-green-400" : "text-red-400"}`}
                  >
                    {ult.conversion_rate.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-[rgba(255,255,255,0.08)] rounded-full overflow-hidden">
                  <div
                    className={`h-full ${ult.conversion_rate >= 50 ? "bg-green-500" : "bg-red-500"}`}
                    style={{ width: `${ult.conversion_rate}%` }}
                  />
                </div>
                <div className="mt-2 text-xs text-muted">
                  {ult.wins}W / {ult.uses} uses by {ult.players.join(", ")}
                </div>
              </div>
            ))}
          </div>

          {/* Ult Insights */}
          {ultimateImpact.insights && ultimateImpact.insights.length > 0 && (
            <div className="mt-4 space-y-2">
              {ultimateImpact.insights.map((insight, i) => (
                <div
                  key={i}
                  className={`text-sm p-2 rounded ${
                    insight.type === "high_impact"
                      ? "bg-green-500/10 text-green-300"
                      : "bg-amber-500/10 text-amber-300"
                  }`}
                >
                  {insight.insight}
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      )}

      {/* Side Performance */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard title="Attack vs Defense">
          <div className="space-y-6">
            <ComparisonBar
              leftValue={side.attack_win_rate ?? 50}
              rightValue={side.defense_win_rate ?? 50}
              leftLabel="Attack"
              rightLabel="Defense"
              leftAccent="cyan"
              rightAccent="red"
            />

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <h4 className="text-xs uppercase tracking-[0.2em] text-cyan">
                  Attack Side
                </h4>
                <div className="flex justify-between text-muted">
                  <span>Rounds</span>
                  <span className="text-white">{side.attack_rounds ?? 0}</span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>Wins</span>
                  <span className="text-white">{side.attack_wins ?? 0}</span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>Win Rate</span>
                  <span className="text-white">
                    {side.attack_win_rate?.toFixed(1) ?? 0}%
                  </span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>K/D</span>
                  <span className="text-white">
                    {side.attack_kd?.toFixed(2) ?? 0}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <h4 className="text-xs uppercase tracking-[0.2em] text-red">
                  Defense Side
                </h4>
                <div className="flex justify-between text-muted">
                  <span>Rounds</span>
                  <span className="text-white">{side.defense_rounds ?? 0}</span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>Wins</span>
                  <span className="text-white">{side.defense_wins ?? 0}</span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>Win Rate</span>
                  <span className="text-white">
                    {side.defense_win_rate?.toFixed(1) ?? 0}%
                  </span>
                </div>
                <div className="flex justify-between text-muted">
                  <span>K/D</span>
                  <span className="text-white">
                    {side.defense_kd?.toFixed(2) ?? 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Multi-Kill Breakdown">
          <div className="space-y-4">
            <div className="text-sm text-muted">
              Total multi-kills:{" "}
              <span className="text-white">{totalMultiKills}</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {multiKills.map((mk) => (
                <div
                  key={mk.label}
                  className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-3 text-center"
                >
                  <div className="text-2xl font-bold text-white">
                    {mk.value}
                  </div>
                  <div className="text-xs uppercase tracking-[0.2em] text-muted">
                    {mk.label}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex h-4 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
              {totalMultiKills > 0 &&
                multiKills.map((mk, i) => {
                  const colors = [
                    "bg-[rgba(6,182,212,0.5)]",
                    "bg-[rgba(34,197,94,0.5)]",
                    "bg-[rgba(234,179,8,0.5)]",
                    "bg-[rgba(255,70,85,0.5)]",
                  ];
                  return (
                    <div
                      key={mk.label}
                      className={colors[i]}
                      style={{
                        width: `${(mk.value / totalMultiKills) * 100}%`,
                      }}
                    />
                  );
                })}
            </div>
          </div>
        </SectionCard>
      </div>

      {/* Top Performers */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Multi-Kill Leaders */}
        {multiKillers.length > 0 && (
          <SectionCard title="Multi-Kill Leaders">
            <DataTable
              headers={["Player", "2K", "3K", "4K", "Ace"]}
              rows={multiKillers
                .slice(0, 5)
                .map((p) => [p.player, p["2k"], p["3k"], p["4k"], p["5k"]])}
            />
          </SectionCard>
        )}

        {/* Clutch Leaders */}
        {clutchPerformers.length > 0 && (
          <SectionCard title="Clutch Leaders">
            <DataTable
              headers={["Player", "Won", "Faced", "Rate"]}
              rows={clutchPerformers
                .slice(0, 5)
                .map((p) => [
                  p.player,
                  p.clutches_won,
                  p.clutches_faced,
                  `${p.clutch_rate?.toFixed(1)}%`,
                ])}
            />
          </SectionCard>
        )}

        {/* Entry Fraggers */}
        {openingDuels.length > 0 && (
          <SectionCard title="Entry Fraggers">
            <DataTable
              headers={["Player", "W", "L", "Rate"]}
              rows={openingDuels
                .slice(0, 5)
                .map((p) => [
                  p.player,
                  p.opening_wins,
                  p.opening_losses,
                  `${p.opening_duel_rate?.toFixed(1)}%`,
                ])}
            />
          </SectionCard>
        )}
      </div>

      {/* Weapon Stats */}
      {weaponRows.length > 0 && (
        <SectionCard title="Weapon Effectiveness">
          <div className="grid gap-4 lg:grid-cols-2">
            <DataTable
              headers={["Weapon", "Kills", "K/D"]}
              rows={weaponRows.slice(0, 5)}
            />
            <DataTable
              headers={["Weapon", "Kills", "K/D"]}
              rows={weaponRows.slice(5, 10)}
            />
          </div>
        </SectionCard>
      )}

      {/* Combat Insights */}
      <SectionCard title="Combat Analysis">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Trading
            </h4>
            <p className="text-sm text-muted">
              {combat.trade_efficiency && combat.trade_efficiency > 50
                ? `Strong trading with ${combat.trade_efficiency.toFixed(1)}% efficiency. Team secures trades consistently.`
                : `Trading at ${combat.trade_efficiency?.toFixed(1) ?? 0}% efficiency (${combat.trade_kills ?? 0}/${combat.trade_opportunities ?? 0}). Focus on positioning for refrag opportunities.`}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Opening Duels
            </h4>
            <p className="text-sm text-muted">
              {firstDuel.team_first_kill_rate &&
              firstDuel.team_first_kill_rate > 50
                ? `Aggressive openers with ${firstDuel.team_first_kill_rate.toFixed(1)}% first blood rate.`
                : `Defensive opener style at ${firstDuel.team_first_kill_rate?.toFixed(1) ?? 0}%. Consider varying entry timing.`}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Clutches
            </h4>
            <p className="text-sm text-muted">
              {clutchWinRate > 30
                ? `Reliable clutch players with ${clutchWinRate.toFixed(1)}% win rate (${totalClutchWins}/${totalClutchAttempts}).`
                : `Clutch situations at ${clutchWinRate.toFixed(1)}% (${totalClutchWins}/${totalClutchAttempts}). ${totalClutchAttempts > 0 ? "Focus on post-plant positioning." : ""}`}
            </p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
