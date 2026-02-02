"use client";

import {
  StatCard,
  SectionCard,
  DataTable,
  ProgressBar,
  ComparisonBar,
} from "./StatCards";
import type { MetricsSummary } from "../types";

interface EconomyTabProps {
  metrics: MetricsSummary;
}

export function EconomyTab({ metrics }: EconomyTabProps) {
  const economy = metrics.economy ?? {};
  const roundPerf = metrics.round_type_performance;
  const ecoMetrics = metrics.economy_metrics;

  // Economy distribution
  const economyStats = [
    { label: "Full Buy", value: economy.full ?? 0, accent: "green" as const },
    { label: "Half Buy", value: economy.half ?? 0, accent: "yellow" as const },
    { label: "Eco", value: economy.eco ?? 0, accent: "red" as const },
    {
      label: "Force Buy",
      value: economy.force ?? 0,
      accent: "violet" as const,
    },
  ];

  // Economy state performance (if available)
  const ecoStatePerf = ecoMetrics?.economy_state_performance;
  const ecoStateRows = ecoStatePerf
    ? Object.entries(ecoStatePerf).map(([state, perf]) => [
        state.charAt(0).toUpperCase() + state.slice(1),
        perf.rounds,
        perf.wins,
        `${perf.win_rate.toFixed(1)}%`,
      ])
    : [];

  return (
    <div className="space-y-6">
      {/* Economy Distribution */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {economyStats.map((stat) => (
          <StatCard
            key={stat.label}
            label={stat.label}
            value={`${stat.value}%`}
            hint="of rounds played"
            percent={stat.value}
            accent={stat.accent}
          />
        ))}
      </div>

      {/* Economy Efficiency */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard title="Round Type Win Rates">
          {roundPerf ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted">Pistol Rounds</span>
                  <span className="text-white">
                    {roundPerf.pistol.wins}/{roundPerf.pistol.rounds} (
                    {roundPerf.pistol.win_rate}%)
                  </span>
                </div>
                <ProgressBar
                  value={roundPerf.pistol.win_rate}
                  accent={roundPerf.pistol.win_rate >= 50 ? "green" : "red"}
                  showValue={false}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted">Eco Rounds</span>
                  <span className="text-white">
                    {roundPerf.eco.wins}/{roundPerf.eco.rounds} (
                    {roundPerf.eco.win_rate}%)
                  </span>
                </div>
                <ProgressBar
                  value={roundPerf.eco.win_rate}
                  accent={roundPerf.eco.win_rate >= 25 ? "green" : "red"}
                  showValue={false}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted">Force Buy Rounds</span>
                  <span className="text-white">
                    {roundPerf.force.wins}/{roundPerf.force.rounds} (
                    {roundPerf.force.win_rate}%)
                  </span>
                </div>
                <ProgressBar
                  value={roundPerf.force.win_rate}
                  accent={roundPerf.force.win_rate >= 40 ? "green" : "yellow"}
                  showValue={false}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted">Full Buy Rounds</span>
                  <span className="text-white">
                    {roundPerf.full_buy.wins}/{roundPerf.full_buy.rounds} (
                    {roundPerf.full_buy.win_rate}%)
                  </span>
                </div>
                <ProgressBar
                  value={roundPerf.full_buy.win_rate}
                  accent={roundPerf.full_buy.win_rate >= 50 ? "green" : "red"}
                  showValue={false}
                />
              </div>
            </div>
          ) : (
            <p className="text-muted text-sm">No round type data available</p>
          )}
        </SectionCard>

        <SectionCard title="Economy Insights">
          <div className="space-y-4">
            {ecoMetrics?.avg_team_economy !== undefined && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted">Avg Team Economy</span>
                <span className="text-white">
                  ${ecoMetrics.avg_team_economy.toFixed(0)}
                </span>
              </div>
            )}
            {ecoMetrics?.economy_advantage_rate !== undefined && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted">Economy Advantage Rate</span>
                <span className="text-white">
                  {ecoMetrics.economy_advantage_rate.toFixed(1)}%
                </span>
              </div>
            )}

            <div className="mt-4 pt-4 border-t border-ethereal">
              <h4 className="text-xs uppercase tracking-[0.2em] text-muted mb-3">
                Key Takeaways
              </h4>
              <ul className="space-y-2 text-sm text-muted">
                <li className="flex items-start gap-2">
                  <span className="text-cyan">•</span>
                  <span>
                    {roundPerf?.eco.win_rate && roundPerf.eco.win_rate > 30
                      ? `Strong eco round conversion (${roundPerf.eco.win_rate}%)`
                      : `Eco rounds are a weakness (${roundPerf?.eco.win_rate ?? 0}%)`}
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan">•</span>
                  <span>
                    {roundPerf?.pistol.win_rate &&
                    roundPerf.pistol.win_rate >= 50
                      ? `Solid pistol round performance (${roundPerf.pistol.win_rate}%)`
                      : `Pistol rounds need improvement (${roundPerf?.pistol.win_rate ?? 0}%)`}
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan">•</span>
                  <span>Full buy reliance: {economy.full ?? 0}% of rounds</span>
                </li>
              </ul>
            </div>
          </div>
        </SectionCard>
      </div>

      {/* Economy State Performance Table */}
      {ecoStateRows.length > 0 && (
        <SectionCard title="Economy State Performance">
          <DataTable
            headers={["Economy State", "Rounds", "Wins", "Win Rate"]}
            rows={ecoStateRows}
          />
        </SectionCard>
      )}

      {/* Economy Distribution Visual */}
      <SectionCard title="Buy Distribution Breakdown">
        <div className="space-y-4">
          <div className="flex h-8 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
            <div
              className="bg-[rgba(34,197,94,0.5)] flex items-center justify-center text-xs text-white"
              style={{ width: `${economy.full ?? 0}%` }}
            >
              {(economy.full ?? 0) > 10 && `${economy.full}%`}
            </div>
            <div
              className="bg-[rgba(234,179,8,0.5)] flex items-center justify-center text-xs text-white"
              style={{ width: `${economy.half ?? 0}%` }}
            >
              {(economy.half ?? 0) > 10 && `${economy.half}%`}
            </div>
            <div
              className="bg-[rgba(109,40,217,0.5)] flex items-center justify-center text-xs text-white"
              style={{ width: `${economy.force ?? 0}%` }}
            >
              {(economy.force ?? 0) > 10 && `${economy.force}%`}
            </div>
            <div
              className="bg-[rgba(255,70,85,0.5)] flex items-center justify-center text-xs text-white"
              style={{ width: `${economy.eco ?? 0}%` }}
            >
              {(economy.eco ?? 0) > 10 && `${economy.eco}%`}
            </div>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-muted">
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded bg-[rgba(34,197,94,0.5)]"></span>
              Full Buy
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded bg-[rgba(234,179,8,0.5)]"></span>
              Half Buy
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded bg-[rgba(109,40,217,0.5)]"></span>
              Force Buy
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded bg-[rgba(255,70,85,0.5)]"></span>
              Eco
            </span>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
