"use client";

import {
  StatCard,
  SectionCard,
  DataTable,
  ProgressBar,
  ComparisonBar,
} from "./StatCards";
import type { MetricsSummary } from "../types";
import { getFallbackImage, getMapImage } from "./imageMaps";

interface MapsTabProps {
  metrics: MetricsSummary;
}

export function MapsTab({ metrics }: MapsTabProps) {
  const mapWinRates = metrics.win_rate_by_map ?? {};
  const mapDetailed = metrics.map_detailed ?? {};
  const sitePrefs = metrics.site_preferences ?? {};
  const mapMetrics = metrics.map_metrics ?? {};
  const paceMetrics = metrics.pace_metrics;
  const siteBias = metrics.site_bias;

  // Sort maps by win rate
  const mapsSorted = Object.entries(mapWinRates).sort((a, b) => b[1] - a[1]);

  // Generate map cards for ALL maps
  const mapCards = mapsSorted.map(([map, winRate]) => {
    const detailed = mapDetailed[map];
    return {
      label: map,
      value: `${winRate}% WR`,
      hint: detailed
        ? `${detailed.rounds_played} rounds | ATK ${detailed.attack_win_rate}% DEF ${detailed.defense_win_rate}%`
        : "No detailed stats",
      percent: winRate,
      accent: winRate >= 50 ? ("green" as const) : ("red" as const),
      imageUrl: getMapImage(map),
      imageAlt: `${map} map`,
    };
  });

  // Site preferences table
  const siteRows = Object.entries(sitePrefs)
    .sort((a, b) => b[1] - a[1])
    .map(([site, rate]) => [site, `${rate}%`]);

  // Detailed map table
  const mapTableRows = Object.entries(mapDetailed).map(([map, stats]) => [
    <div key={map} className="flex items-center gap-2">
      <img
        src={getMapImage(map)}
        alt={`${map} map`}
        className="h-6 w-6 rounded object-cover border border-ethereal/60"
        loading="lazy"
        onError={(event) => {
          event.currentTarget.src = getFallbackImage(map, "6d28d9");
        }}
      />
      <span>{map}</span>
    </div>,
    stats.rounds_played,
    `${stats.win_rate}%`,
    `${stats.attack_win_rate}%`,
    `${stats.defense_win_rate}%`,
    stats.top_agent ?? "—",
  ]);

  // Map-specific site preferences
  const mapSitePrefs = mapMetrics.map_site_preferences ?? {};

  // Plant timing data (from pace metrics)
  const paceByMap = paceMetrics?.by_map ?? {};

  return (
    <div className="space-y-6">
      {/* All Maps Grid */}
      <SectionCard title="Map Win Rates">
        <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {mapCards.map((stat) => (
            <StatCard key={stat.label} {...stat} />
          ))}
        </div>
      </SectionCard>

      {/* Map Performance Table */}
      {mapTableRows.length > 0 && (
        <SectionCard title="Map Performance Breakdown">
          <DataTable
            headers={[
              "Map",
              "Rounds",
              "Win Rate",
              "Attack WR",
              "Defense WR",
              "Top Agent",
            ]}
            rows={mapTableRows}
          />
        </SectionCard>
      )}

      {/* Pace of Play - NEW */}
      {paceMetrics && (
        <SectionCard title="Pace of Play">
          <div className="grid gap-4 lg:grid-cols-2">
            {/* Overall Pace Histogram */}
            <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-4">
                Attack Tempo: {paceMetrics.style}
              </h4>
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted">
                    <span>Rush (&lt;30s)</span>
                    <span className="text-white">
                      {paceMetrics.histogram.rush.percent}%
                    </span>
                  </div>
                  <div className="h-3 bg-[rgba(255,255,255,0.08)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan to-cyan/50"
                      style={{
                        width: `${paceMetrics.histogram.rush.percent}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted">
                    <span>Default (30-60s)</span>
                    <span className="text-white">
                      {paceMetrics.histogram.default.percent}%
                    </span>
                  </div>
                  <div className="h-3 bg-[rgba(255,255,255,0.08)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 to-green-500/50"
                      style={{
                        width: `${paceMetrics.histogram.default.percent}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted">
                    <span>Late (&gt;60s)</span>
                    <span className="text-white">
                      {paceMetrics.histogram.late.percent}%
                    </span>
                  </div>
                  <div className="h-3 bg-[rgba(255,255,255,0.08)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-amber-500 to-amber-500/50"
                      style={{
                        width: `${paceMetrics.histogram.late.percent}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-ethereal/30 text-sm text-muted">
                Avg Plant Time:{" "}
                <span className="text-white">
                  {paceMetrics.avg_plant_time ?? 0}s
                </span>
                {" | "}
                {paceMetrics.total_plants_analyzed} plants analyzed
              </div>
            </div>

            {/* First Damage Timing */}
            <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-4">
                First Contact Timing
              </h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-muted text-sm">
                    Attack First Contact
                  </span>
                  <span className="text-lg font-bold text-cyan">
                    {paceMetrics.attack_first_damage_avg ?? "—"}s
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted text-sm">
                    Defense First Contact
                  </span>
                  <span className="text-lg font-bold text-red">
                    {paceMetrics.defense_first_damage_avg ?? "—"}s
                  </span>
                </div>
                <div className="text-xs text-muted pt-2 border-t border-ethereal/30">
                  {paceMetrics.attack_first_damage_avg &&
                  paceMetrics.attack_first_damage_avg < 25
                    ? "Fast openers - expect early aggression and map control fights."
                    : "Methodical approach - they take time to set up."}
                </div>
              </div>
            </div>
          </div>
        </SectionCard>
      )}

      {/* Site Bias with Win Rates - NEW */}
      {siteBias && Object.keys(siteBias.by_map).length > 0 && (
        <SectionCard title="Site Bias & Win Rates">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(siteBias.by_map).map(([map, sites]) => (
              <div
                key={map}
                className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
              >
                <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-3 flex items-center gap-2">
                  <img
                    src={getMapImage(map)}
                    alt={map}
                    className="h-5 w-5 rounded object-cover"
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                    }}
                  />
                  {map}
                </h4>
                <div className="space-y-2">
                  {Object.entries(sites)
                    .sort((a, b) => b[1].preference - a[1].preference)
                    .map(([site, data]) => (
                      <div key={site} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted">{site}</span>
                          <span
                            className={
                              data.win_rate >= 50
                                ? "text-green-400"
                                : "text-red-400"
                            }
                          >
                            {data.preference}% pick | {data.win_rate}% WR
                          </span>
                        </div>
                        <div className="flex gap-1 h-2">
                          <div
                            className="bg-[rgba(6,182,212,0.5)] rounded-l"
                            style={{ width: `${data.preference}%` }}
                            title={`Pick rate: ${data.preference}%`}
                          />
                          <div
                            className={`rounded-r ${data.win_rate >= 50 ? "bg-green-500/50" : "bg-red-500/50"}`}
                            style={{ width: `${data.win_rate}%` }}
                            title={`Win rate: ${data.win_rate}%`}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>

          {/* Site Bias Insights */}
          {siteBias.insights && siteBias.insights.length > 0 && (
            <div className="mt-4 space-y-2">
              <h4 className="text-xs uppercase tracking-[0.2em] text-amber-400">
                Actionable Insights
              </h4>
              {siteBias.insights.map((insight, i) => (
                <div
                  key={i}
                  className="cut-corner bg-amber-500/10 border border-amber-500/30 p-3 text-sm text-muted"
                >
                  {insight.insight}
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      )}

      {/* Site Preferences & Attack/Defense Split */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard title="Overall Site Preferences">
          {siteRows.length > 0 ? (
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {siteRows.map(([site, rate]) => (
                <div key={site} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>{site}</span>
                    <span>{rate}</span>
                  </div>
                  <ProgressBar
                    value={parseFloat(rate)}
                    accent="cyan"
                    showValue={false}
                  />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm">No site preference data</p>
          )}
        </SectionCard>

        <SectionCard title="Attack vs Defense by Map">
          {Object.entries(mapDetailed).length > 0 ? (
            <div className="space-y-4 max-h-80 overflow-y-auto">
              {Object.entries(mapDetailed).map(([map, stats]) => (
                <div key={map} className="space-y-2">
                  <div className="text-sm text-white">{map}</div>
                  <ComparisonBar
                    leftValue={stats.attack_win_rate}
                    rightValue={stats.defense_win_rate}
                    leftLabel="ATK"
                    rightLabel="DEF"
                    leftAccent="cyan"
                    rightAccent="red"
                  />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm">No map data available</p>
          )}
        </SectionCard>
      </div>

      {/* Map-Specific Site Preferences */}
      {Object.entries(mapSitePrefs).length > 0 && (
        <SectionCard title="Site Preferences by Map">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(mapSitePrefs).map(([map, sites]) => (
              <div
                key={map}
                className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
              >
                <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-3">
                  {map}
                </h4>
                <div className="space-y-2">
                  {Object.entries(sites as Record<string, number>)
                    .sort((a, b) => b[1] - a[1])
                    .map(([site, rate]) => (
                      <div key={site} className="flex justify-between text-sm">
                        <span className="text-muted">{site}</span>
                        <span className="text-white">{rate}%</span>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Pace by Map */}
      {Object.entries(paceByMap).length > 0 && (
        <SectionCard title="Pace by Map">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(paceByMap).map(([map, pace]) => (
              <div
                key={map}
                className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
              >
                <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-3">
                  {map}
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted">Avg Plant Time</span>
                    <span className="text-white">{pace.avg_time}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Rush ({`<`}30s)</span>
                    <span className="text-cyan">{pace.rush_percent}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Default (30-60s)</span>
                    <span className="text-green-400">
                      {pace.default_percent}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Late ({`>`}60s)</span>
                    <span className="text-amber-400">{pace.late_percent}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Map Insights */}
      <SectionCard title="Map Analysis">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Best Map
            </h4>
            <p className="text-sm text-muted">
              {mapsSorted.length > 0
                ? `${mapsSorted[0][0]} with ${mapsSorted[0][1]}% win rate. Consider picking this map when possible.`
                : "Insufficient map data"}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Worst Map
            </h4>
            <p className="text-sm text-muted">
              {mapsSorted.length > 0
                ? `${mapsSorted[mapsSorted.length - 1][0]} at ${mapsSorted[mapsSorted.length - 1][1]}% win rate. Target for opponent's map pick.`
                : "Insufficient map data"}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Side Preference
            </h4>
            <p className="text-sm text-muted">
              {Object.entries(mapDetailed).length > 0
                ? (() => {
                    const avgAtk =
                      Object.values(mapDetailed).reduce(
                        (s, m) => s + m.attack_win_rate,
                        0,
                      ) / Object.values(mapDetailed).length;
                    const avgDef =
                      Object.values(mapDetailed).reduce(
                        (s, m) => s + m.defense_win_rate,
                        0,
                      ) / Object.values(mapDetailed).length;
                    return avgAtk > avgDef
                      ? `Attack-sided team (${avgAtk.toFixed(1)}% vs ${avgDef.toFixed(1)}% def)`
                      : `Defense-sided team (${avgDef.toFixed(1)}% vs ${avgAtk.toFixed(1)}% atk)`;
                  })()
                : "Insufficient side data"}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Site Bias
            </h4>
            <p className="text-sm text-muted">
              {siteRows.length > 0
                ? `Primary site: ${siteRows[0][0]} (${siteRows[0][1]}). Secondary: ${siteRows[1]?.[0] ?? "N/A"}.`
                : "Insufficient site data"}
            </p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
