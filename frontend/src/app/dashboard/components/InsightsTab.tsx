"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SectionCard, ProgressBar } from "./StatCards";
import type { MetricsSummary } from "../types";

interface InsightsTabProps {
  insights: Record<string, string>;
  teamName: string;
  metrics?: MetricsSummary;
}

// Parse markdown-style content into structured data (excluding tables which render separately)
function parseInsightContent(content: string): string[] {
  if (!content) return [];
  return content
    .split(/(?:^|\n)(?:[-•*]|\d+\.|#{1,3})\s*/g)
    .map((line) => line.replace(/\*\*/g, "").trim())
    .filter((line) => {
      if (line.length === 0 || line.length > 200) return false;
      // Skip table rows - they'll be rendered by ReactMarkdown
      if (line.startsWith("|") || line.includes("|---")) return false;
      if (line.match(/^\|.*\|$/)) return false;
      return true;
    });
}

// Extract key-value pairs from content (excluding table syntax)
function extractKeyValues(
  content: string,
): { label: string; value: string; highlight?: boolean }[] {
  if (!content) return [];
  const pairs: { label: string; value: string; highlight?: boolean }[] = [];
  const lines = content.split("\n");

  for (const line of lines) {
    // Skip markdown table lines - they render separately
    if (
      line.startsWith("|") ||
      line.includes("|---") ||
      line.match(/^\|.*\|$/)
    ) {
      continue;
    }
    const match = line.match(/\*?\*?([^:*|]+)\*?\*?:\s*([^|]+)/);
    if (match) {
      const label = match[1].trim().replace(/\*\*/g, "");
      const value = match[2].trim().replace(/\*\*/g, "");
      if (label.length < 50 && value.length < 100 && !label.includes("---")) {
        const highlight =
          value.includes("%") ||
          value.toLowerCase().includes("high") ||
          value.toLowerCase().includes("critical");
        pairs.push({ label, value, highlight });
      }
    }
  }
  return pairs.slice(0, 12);
}

// Stat card component
function StatItem({
  label,
  value,
  accent = "cyan",
}: {
  label: string;
  value: string;
  accent?: "cyan" | "valorant" | "green" | "yellow";
}) {
  const accentColors = {
    cyan: "text-cyan border-cyan/30",
    valorant: "text-valorant border-valorant/30",
    green: "text-green border-green/30",
    yellow: "text-yellow border-yellow/30",
  };

  return (
    <div
      className={`p-3 bg-glass border ${accentColors[accent]} rounded flex flex-col`}
    >
      <span className="text-[10px] uppercase tracking-wider text-muted mb-1">
        {label}
      </span>
      <span
        className={`text-sm font-medium ${accentColors[accent].split(" ")[0]}`}
      >
        {value}
      </span>
    </div>
  );
}

// Player card component
function PlayerCard({
  name,
  role,
  threat,
}: {
  name: string;
  role: string;
  threat: "HUNT" | "WATCH" | "AVOID" | "NEUTRAL";
}) {
  const threatColors = {
    HUNT: "bg-valorant/20 text-valorant border-valorant/40",
    WATCH: "bg-yellow/20 text-yellow border-yellow/40",
    AVOID: "bg-red-500/20 text-red-400 border-red-500/40",
    NEUTRAL: "bg-cyan/20 text-cyan border-cyan/40",
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-glass border border-ethereal rounded">
      <div
        className={`w-10 h-10 rounded flex items-center justify-center text-lg font-bold ${threatColors[threat]}`}
      >
        {name.charAt(0).toUpperCase()}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-white font-medium truncate">{name}</div>
        <div className="text-xs text-muted truncate">{role}</div>
      </div>
      <div
        className={`px-2 py-0.5 text-[10px] uppercase tracking-wider rounded border ${threatColors[threat]}`}
      >
        {threat}
      </div>
    </div>
  );
}

// Protocol card for IF/THEN statements
function ProtocolCard({
  condition,
  action,
}: {
  condition: string;
  action: string;
}) {
  return (
    <div className="p-3 bg-glass border border-ethereal rounded">
      <div className="flex items-start gap-2">
        <span className="text-[10px] uppercase tracking-wider text-valorant font-semibold whitespace-nowrap">
          IF
        </span>
        <span className="text-xs text-white">{condition}</span>
      </div>
      <div className="flex items-start gap-2 mt-2 pl-4 border-l-2 border-cyan/40">
        <span className="text-[10px] uppercase tracking-wider text-cyan font-semibold whitespace-nowrap">
          THEN
        </span>
        <span className="text-xs text-muted">{action}</span>
      </div>
    </div>
  );
}

export function InsightsTab({ insights, teamName, metrics }: InsightsTabProps) {
  const strategyData = extractKeyValues(insights?.strategies || "");
  const compositionData = extractKeyValues(insights?.compositions || "");
  const counterPoints = parseInsightContent(insights?.counters || "");

  // Extract IF/THEN protocols
  const protocols: { condition: string; action: string }[] = [];
  const howToWin = insights?.how_to_win || "";
  const ifThenMatches = howToWin.matchAll(
    /IF[:\s]+([^→\n]+?)(?:→|THEN|then)[:\s]*([^\n]+)/gi,
  );
  for (const match of ifThenMatches) {
    protocols.push({
      condition: match[1].replace(/\*\*/g, "").trim(),
      action: match[2].replace(/\*\*/g, "").trim(),
    });
  }

  const sideMetrics = metrics?.side_metrics;
  const firstDuel = metrics?.first_duel;
  const roundPerf = metrics?.round_type_performance as Record<
    string,
    { win_rate: number }
  >;
  const combat = metrics?.combat_metrics;

  // Determine player threat levels based on metrics
  const sortedPlayers = [...(metrics?.player_tendencies || [])].sort(
    (a, b) => (b.kd_ratio ?? 0) - (a.kd_ratio ?? 0),
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">
            Tactical Briefing vs{" "}
            <span className="text-valorant">{teamName}</span>
          </h2>
          <p className="text-xs text-muted mt-1">
            {sortedPlayers.length} players •{" "}
            {Object.keys(metrics?.win_rate_by_map || {}).length} maps
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs text-muted">Live</span>
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatItem
          label="Win Rate"
          value={`${metrics?.win_rate?.toFixed(1) || 0}%`}
          accent={(metrics?.win_rate || 0) >= 50 ? "green" : "valorant"}
        />
        <StatItem
          label="Attack WR"
          value={`${sideMetrics?.attack_win_rate?.toFixed(1) || 0}%`}
          accent={
            (sideMetrics?.attack_win_rate || 0) >= 50 ? "green" : "valorant"
          }
        />
        <StatItem
          label="Defense WR"
          value={`${sideMetrics?.defense_win_rate?.toFixed(1) || 0}%`}
          accent={
            (sideMetrics?.defense_win_rate || 0) >= 50 ? "green" : "valorant"
          }
        />
        <StatItem
          label="First Blood"
          value={`${firstDuel?.team_first_kill_rate?.toFixed(1) || 0}%`}
          accent="cyan"
        />
        <StatItem
          label="Pistol WR"
          value={`${roundPerf?.pistol?.win_rate?.toFixed(1) || 0}%`}
          accent="yellow"
        />
        <StatItem
          label="Trade Eff"
          value={`${combat?.trade_efficiency?.toFixed(1) || 0}%`}
          accent="cyan"
        />
      </div>

      {/* Main Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Side Strength */}
        <SectionCard title="Side Performance">
          <div className="space-y-4">
            <div className="p-3 bg-glass rounded border border-ethereal">
              <div className="flex justify-between text-xs text-muted mb-2">
                <span>
                  Attack {sideMetrics?.attack_win_rate?.toFixed(1) || 0}%
                </span>
                <span>
                  Defense {sideMetrics?.defense_win_rate?.toFixed(1) || 0}%
                </span>
              </div>
              <div className="h-3 bg-ethereal rounded overflow-hidden flex">
                <div
                  className="h-full bg-gradient-to-r from-valorant to-valorant/70"
                  style={{
                    width: `${((sideMetrics?.attack_win_rate || 50) / ((sideMetrics?.attack_win_rate || 50) + (sideMetrics?.defense_win_rate || 50))) * 100}%`,
                  }}
                />
                <div className="h-full bg-gradient-to-r from-cyan/70 to-cyan flex-1" />
              </div>
              <div className="text-center text-xs text-muted mt-2">
                {(sideMetrics?.attack_win_rate || 0) >
                (sideMetrics?.defense_win_rate || 0)
                  ? "⚔️ Attack-favored"
                  : "🛡️ Defense-favored"}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {strategyData.slice(0, 6).map((item, i) => (
                <div
                  key={i}
                  className="p-2 bg-[rgba(15,18,25,0.5)] rounded border border-ethereal/50"
                >
                  <div className="text-[10px] text-muted uppercase tracking-wider">
                    {item.label}
                  </div>
                  <div
                    className={`text-sm ${item.highlight ? "text-valorant" : "text-white"}`}
                  >
                    {item.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </SectionCard>

        {/* Player Threats */}
        <SectionCard title="Player Threat Assessment">
          <div className="space-y-2">
            {sortedPlayers.slice(0, 5).map((p, i) => (
              <PlayerCard
                key={i}
                name={p.player}
                role={`KD: ${p.kd_ratio?.toFixed(2) || "—"} • ${p.top_agent || "—"}`}
                threat={
                  i === 0
                    ? "HUNT"
                    : (p.kd_ratio || 0) < 0.9
                      ? "HUNT"
                      : (p.first_kill_rate || 0) > 20
                        ? "AVOID"
                        : "WATCH"
                }
              />
            ))}
          </div>
        </SectionCard>

        {/* Compositions */}
        <SectionCard title="Agent Core">
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {metrics?.agent_composition?.slice(0, 5).map((agent, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 px-3 py-1.5 bg-glass border border-ethereal rounded"
                >
                  <span className="text-white text-sm">{agent.agent}</span>
                  <span className="text-xs text-muted">
                    {agent.pick_rate?.toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>

            <div className="space-y-2 pt-2 border-t border-ethereal">
              <div className="text-xs text-muted uppercase tracking-wider">
                Role Balance
              </div>
              {Object.entries(metrics?.role_distribution || {}).map(
                ([role, pct]) => (
                  <div key={role} className="flex items-center gap-3">
                    <span className="text-xs text-white w-20">{role}</span>
                    <div className="flex-1">
                      <ProgressBar
                        value={pct}
                        accent="cyan"
                        showValue={false}
                      />
                    </div>
                    <span className="text-xs text-muted w-12 text-right">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                ),
              )}
            </div>
          </div>
        </SectionCard>

        {/* Counter Protocols */}
        <SectionCard title="Counter Protocols">
          <div className="max-h-[320px] overflow-y-auto prose prose-invert prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-li:my-1 prose-headings:text-cyan prose-headings:text-sm prose-h2:text-base prose-h3:text-cyan/80 prose-strong:text-white [&_table]:w-full [&_table]:text-xs [&_th]:text-cyan [&_th]:border [&_th]:border-ethereal [&_th]:px-2 [&_th]:py-1 [&_th]:bg-glass [&_td]:border [&_td]:border-ethereal [&_td]:px-2 [&_td]:py-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {insights?.counters || "Analyzing counter strategies..."}
            </ReactMarkdown>
          </div>
        </SectionCard>
      </div>

      {/* IF/THEN Decision Protocols */}
      {protocols.length > 0 && (
        <SectionCard title="Decision Protocols">
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {protocols.slice(0, 6).map((protocol, i) => (
              <ProtocolCard
                key={i}
                condition={protocol.condition}
                action={protocol.action}
              />
            ))}
          </div>
        </SectionCard>
      )}

      {/* Site Intel */}
      <SectionCard title="Site Intelligence">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs text-muted uppercase tracking-wider mb-2">
              Attack Distribution
            </div>
            <div className="space-y-2">
              {Object.entries(metrics?.site_preferences || {})
                .sort(([, a], [, b]) => b - a)
                .map(([site, pct]) => (
                  <div key={site} className="flex items-center gap-3">
                    <span className="text-sm text-white w-16">{site}-Site</span>
                    <div className="flex-1">
                      <ProgressBar
                        value={pct}
                        accent={pct > 40 ? "red" : "cyan"}
                        showValue={false}
                      />
                    </div>
                    <span className="text-sm text-muted w-12 text-right">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted uppercase tracking-wider mb-2">
              Pistol Sites
            </div>
            <div className="space-y-2">
              {Object.entries(metrics?.pistol_site_preferences || {})
                .sort(([, a], [, b]) => b - a)
                .map(([site, pct]) => (
                  <div key={site} className="flex items-center gap-3">
                    <span className="text-sm text-white w-16">{site}-Site</span>
                    <div className="flex-1">
                      <ProgressBar
                        value={pct}
                        accent="yellow"
                        showValue={false}
                      />
                    </div>
                    <span className="text-sm text-muted w-12 text-right">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Footer */}
      <div className="flex items-center justify-center gap-6 pt-4 border-t border-ethereal text-xs text-muted">
        <span className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" /> Stats
        </span>
        <span className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan" /> Patterns
        </span>
        <span className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-valorant" /> AI
        </span>
      </div>
    </div>
  );
}
