"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { StatCard, SectionCard, DataTable, ProgressBar } from "./StatCards";
import type { MetricsSummary } from "../types";

interface OverviewTabProps {
  metrics: MetricsSummary;
  matchesAnalyzed: number;
  teamName: string;
  insights: Record<string, string>;
}

function getTopEntry(values: Record<string, number> | undefined) {
  const entries = Object.entries(values ?? {});
  if (!entries.length) return ["—", 0] as const;
  return entries.reduce(
    (best, entry) => (entry[1] > best[1] ? entry : best),
    entries[0],
  );
}

export function OverviewTab({
  metrics,
  matchesAnalyzed,
  teamName,
  insights,
}: OverviewTabProps) {
  const [bestMap, bestMapRate] = getTopEntry(metrics.win_rate_by_map);
  const [siteName, siteRate] = getTopEntry(metrics.site_preferences);
  const [pistolSite, pistolRate] = getTopEntry(metrics.pistol_site_preferences);

  const quickStats = [
    {
      label: "Overall Win Rate",
      value: `${metrics.win_rate?.toFixed(1) ?? 0}%`,
      hint: `Across ${matchesAnalyzed} matches`,
      percent: metrics.win_rate ?? 0,
      accent: "cyan" as const,
    },
    {
      label: "Best Map",
      value: `${bestMap} (${bestMapRate}%)`,
      hint: "Highest win rate map",
      percent: bestMapRate,
      accent: "green" as const,
    },
    {
      label: "Aggression Style",
      value: metrics.aggression?.style ?? "Unknown",
      hint: `Avg round ${metrics.aggression?.avg_duration ?? 0}s | Rush ${metrics.aggression?.rush_rate ?? 0}%`,
      percent: metrics.aggression?.rush_rate ?? 0,
      accent: "violet" as const,
    },
    {
      label: "First Kill Rate",
      value: `${metrics.first_duel?.team_first_kill_rate?.toFixed(1) ?? 0}%`,
      hint: `Conversion: ${metrics.first_duel?.first_kill_conversion_rate?.toFixed(1) ?? 0}%`,
      percent: metrics.first_duel?.team_first_kill_rate ?? 0,
      accent: "red" as const,
    },
  ];

  // Round type performance
  const roundPerf = metrics.round_type_performance;
  const roundTypes = roundPerf
    ? [
        { label: "Pistol", ...roundPerf.pistol },
        { label: "Eco", ...roundPerf.eco },
        { label: "Force Buy", ...roundPerf.force },
        { label: "Full Buy", ...roundPerf.full_buy },
      ]
    : [];

  // Opponent stats
  const opponentRows = (metrics.opponent_stats ?? [])
    .slice(0, 5)
    .map((opp) => [
      opp.opponent,
      opp.matches,
      opp.rounds_played,
      `${opp.win_rate}%`,
    ]);

  // Insights sections (brief overview - full insights on Insights tab)
  const reportSections = [
    {
      title: "Common Strategies",
      body: insights?.strategies ?? "Awaiting analysis...",
    },
    {
      title: "Player Tendencies",
      body: insights?.tendencies ?? "Awaiting analysis...",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Quick Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {quickStats.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      {/* Team Identity Row */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard title="Team Snapshot">
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted">Preferred Site</span>
              <span className="text-white">
                {siteName} ({siteRate}%)
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted">Pistol Site</span>
              <span className="text-white">
                {pistolSite} ({pistolRate}%)
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted">Full Buy %</span>
              <span className="text-white">{metrics.economy?.full ?? 0}%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted">Eco %</span>
              <span className="text-white">{metrics.economy?.eco ?? 0}%</span>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Round Type Performance">
          {roundTypes.length > 0 ? (
            <div className="space-y-3">
              {roundTypes.map((rt) => (
                <div key={rt.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>{rt.label}</span>
                    <span>
                      {rt.wins}/{rt.rounds} rounds ({rt.win_rate}%)
                    </span>
                  </div>
                  <ProgressBar
                    value={rt.win_rate}
                    accent={rt.win_rate >= 50 ? "green" : "red"}
                    showValue={false}
                  />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm">No round type data available</p>
          )}
        </SectionCard>
      </div>

      {/* Opponent History */}
      {opponentRows.length > 0 && (
        <SectionCard title="Recent Opponents">
          <DataTable
            headers={["Opponent", "Matches", "Rounds", "Win Rate"]}
            rows={opponentRows}
          />
        </SectionCard>
      )}

      {/* AI Insights */}
      <SectionCard title="Tactical Briefing">
        <div className="grid gap-4 md:grid-cols-2">
          {reportSections.map((section) => (
            <div
              key={section.title}
              className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
            >
              <div className="text-sm uppercase tracking-[0.2em] text-cyan">
                {section.title}
              </div>
              <div className="mt-2 text-sm text-muted prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0 [&_table]:w-full [&_table]:text-xs [&_th]:text-cyan [&_th]:border [&_th]:border-ethereal [&_th]:px-2 [&_th]:py-1 [&_th]:bg-glass [&_td]:border [&_td]:border-ethereal [&_td]:px-2 [&_td]:py-1">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {section.body}
                </ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
