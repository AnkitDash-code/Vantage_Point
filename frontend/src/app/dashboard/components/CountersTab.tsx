"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { SectionCard, DataTable } from "./StatCards";
import type { MetricsSummary } from "../types";

interface CountersTabProps {
  metrics: MetricsSummary;
  teamName: string;
  insights: Record<string, string>;
}

// Agent counter database (derived from VALORANT meta knowledge)
const AGENT_COUNTERS: Record<string, { counters: string[]; strategy: string }> =
  {
    Jett: {
      counters: ["Cypher", "KAY/O", "Fade"],
      strategy:
        "Use detection utility to deny dash escapes. Stack off-angles to punish aggressive peeks.",
    },
    Raze: {
      counters: ["Killjoy", "Cypher", "Sage"],
      strategy:
        "Zone control denies satchel entries. Wall off choke points before explosive utility.",
    },
    Reyna: {
      counters: ["Breach", "Fade", "Skye"],
      strategy:
        "Flash before she can dismiss. Team-fire to prevent heal chains. Deny orb pickups.",
    },
    Phoenix: {
      counters: ["Viper", "Brimstone", "KAY/O"],
      strategy:
        "Mollies cancel his heal. Suppress ultimate usage. Wait out wall duration.",
    },
    Neon: {
      counters: ["Sage", "Vyse", "Deadlock"],
      strategy:
        "Slow utility denies slide momentum. Trap zones force predictable paths.",
    },
    Yoru: {
      counters: ["Cypher", "Killjoy", "Fade"],
      strategy:
        "Info util reveals fake teleports. Sound cues telegraph flanks. Pre-aim common TP spots.",
    },
    Iso: {
      counters: ["Fade", "Skye", "Breach"],
      strategy:
        "Flash through shield timing. Team trades deny isolation. Prowler reveals position.",
    },
    Sova: {
      counters: ["Viper", "Omen", "Harbor"],
      strategy:
        "One-way smokes deny dart value. Aggressive pressure before recon completes.",
    },
    Breach: {
      counters: ["Jett", "Chamber", "Reyna"],
      strategy:
        "Dash/dismiss through flashes. Off-angle positions dodge fault lines. Mobility counters static utility.",
    },
    Skye: {
      counters: ["Yoru", "Neon", "Jett"],
      strategy:
        "Fast entries overwhelm heal cooldowns. Dash through dog. Fake flash triggers early.",
    },
    "KAY/O": {
      counters: ["Reyna", "Raze", "Jett"],
      strategy:
        "Gun skill duelists thrive when util is suppressed. Dash/dismiss escapes knife radius.",
    },
    Fade: {
      counters: ["Jett", "Chamber", "Omen"],
      strategy:
        "TP out of tether range. Dash before seize lands. Mobility beats tracking util.",
    },
    Gekko: {
      counters: ["Cypher", "Killjoy", "KAY/O"],
      strategy:
        "Suppress before Dizzy deploy. Info util punishes predictable Wingman plants. Destroy flash early.",
    },
    Omen: {
      counters: ["Fade", "Sova", "Skye"],
      strategy:
        "Info util reveals paranoia timings. Recon dart exposes smoke positions. Flash TP targets.",
    },
    Brimstone: {
      counters: ["Harbor", "Astra", "Jett"],
      strategy:
        "Water blocks molly. Smoke wars favor quick redeploy. Dash through stim beacon.",
    },
    Viper: {
      counters: ["Harbor", "Astra", "Breach"],
      strategy:
        "Water blocks decay. Gravity well pulls through wall. Aftershock destroys orb.",
    },
    Astra: {
      counters: ["KAY/O", "Yoru", "Omen"],
      strategy:
        "Suppress before star placement. TP behind smoke setups. Fast site hits before setup.",
    },
    Harbor: {
      counters: ["Sova", "Fade", "Breach"],
      strategy:
        "Dart through cascade. Prowler reveals wall positions. Aftershock forces early wall pop.",
    },
    Clove: {
      counters: ["Fade", "Sova", "KAY/O"],
      strategy:
        "Info util denies smoke creativity. Suppress before smoke. Track movement with recon.",
    },
    Sage: {
      counters: ["Raze", "Breach", "Viper"],
      strategy:
        "Satchel over wall. Aftershock destroys wall. Molly denies heal.",
    },
    Cypher: {
      counters: ["Breach", "Raze", "Sova"],
      strategy:
        "Explosive util destroys setups. Dart reveals camera. Aftershock clears trips.",
    },
    Killjoy: {
      counters: ["Jett", "Yoru", "Sova"],
      strategy:
        "Dash past nanoswarm. TP flank bot. Shock dart turret before entry.",
    },
    Chamber: {
      counters: ["Fade", "Skye", "Omen"],
      strategy:
        "Tether prevents TP escapes. Flash TP angles. Smoke off-angles denies Op value.",
    },
    Deadlock: {
      counters: ["Jett", "Neon", "Raze"],
      strategy:
        "Dash/slide through sensors. Satchel over net. Mobility beats wall util.",
    },
    Vyse: {
      counters: ["Omen", "Yoru", "Chamber"],
      strategy:
        "TP out of slows. Reposition before flash walls. Off-angles dodge static setups.",
    },
  };

// Map-specific counter strategies
const MAP_COUNTERS: Record<
  string,
  { weakness: string; exploit: string; setup: string }
> = {
  Ascent: {
    weakness: "Mid control dominates both sites",
    exploit: "Win Catwalk → pinch A Main/B Main simultaneously",
    setup: "Default: 2 Mid, 1 A, 1 B, 1 flex. Force mid fight early rounds.",
  },
  Bind: {
    weakness: "No mid → predictable rotations",
    exploit: "Fake A teleporter audio → fast B execute through showers",
    setup: "Stack hookah on B. One-way smokes on A short. Deny TP control.",
  },
  Haven: {
    weakness: "Three sites spread defenders thin",
    exploit: "Rush C long with numbers (4v2 advantage). Fake A after plant.",
    setup: "Fast C hits pre-1:30. Lurk A garage. Time B split with C rotate.",
  },
  Split: {
    weakness: "Vertical angles favor defenders",
    exploit: "Ropes fast B with smokes. Molly heaven/rafter pre-plant.",
    setup:
      "Sage wall + smokes mandatory. Lurk mid for late flanks. Avoid A main.",
  },
  Icebox: {
    weakness: "Long rotate times between sites",
    exploit: "Mid control → force B tube commit → rotate A",
    setup: "B default with Sage wall tube. Op A main. Rush mid kitchen rounds.",
  },
  Breeze: {
    weakness: "Long sightlines favor Operators",
    exploit: "Fast A cave rush denies Op setup time. Viper wall mid→A.",
    setup:
      "2 mid 2 A 1 B default. Molly elbow. Flash then dash A main. Ban Jett.",
  },
  Fracture: {
    weakness: "Attackers spawn near both sites",
    exploit:
      "Instant A main pressure OR B arcade rush. Punish indecisive defenders.",
    setup: "Split 3A-2B or 3B-2A. Never mid. Ultra-fast site hits (<20s).",
  },
  Pearl: {
    weakness: "Mid control → both site access",
    exploit: "Win mid → fake B link → A site pressure through mid",
    setup:
      "Viper mid wall. Fight for connector. Sage wall B long. Default mid-heavy.",
  },
  Lotus: {
    weakness: "Three sites but strong rotate paths",
    exploit: "C rush through main → rotate defenders open A/B flanks",
    setup: "C fast hits. A main control. B upper deny. Rotating pillar plays.",
  },
  Sunset: {
    weakness: "Mid control gives A or B access",
    exploit: "Market control → B elbow pressure while faking A main",
    setup: "Default 2-1-2. Fight mid early. Lurk A courtyard. Fast B executes.",
  },
  Abyss: {
    weakness: "Fall-off zones limit defensive rotations",
    exploit: "Pressure one site → punish slow rotates through dangerous paths",
    setup: "Control mid towers. Use smokes to block fall-off sightlines.",
  },
};

function getTopAgents(metrics: MetricsSummary): string[] {
  return (metrics.agent_composition || [])
    .slice(0, 5)
    .map((a) => a.agent)
    .filter((agent) => agent in AGENT_COUNTERS);
}

function getTopMap(metrics: MetricsSummary): string | null {
  const maps = Object.entries(metrics.win_rate_by_map || {});
  if (maps.length === 0) return null;
  return maps.sort((a, b) => b[1] - a[1])[0][0];
}

function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

export function CountersTab({ metrics, teamName, insights }: CountersTabProps) {
  const topAgents = getTopAgents(metrics);
  const topMap = getTopMap(metrics);
  const normalizedTopMap = topMap ? capitalizeFirst(topMap) : null;
  const aggressionStyle = metrics.aggression?.style || "Unknown";
  const rushRate = metrics.aggression?.rush_rate || 0;
  const avgDuration = metrics.aggression?.avg_duration || 0;
  const firstKillRate = metrics.first_duel?.team_first_kill_rate || 0;
  const sitePrefs = metrics.site_preferences || {};
  const sortedSites = Object.entries(sitePrefs).sort((a, b) => b[1] - a[1]);
  const primarySite = sortedSites[0]?.[0] || "Unknown";
  const topSiteRate = sortedSites[0]?.[1] || 0;

  // Generate exploit patterns
  const exploits: Array<{
    category: string;
    pattern: string;
    counter: string;
  }> = [];

  // Economy exploit
  const ecoDistribution = metrics.economy || {};
  if (ecoDistribution.force && ecoDistribution.force > 35) {
    exploits.push({
      category: "Economy",
      pattern: `High force buy rate (${ecoDistribution.force}%)`,
      counter:
        "Anti-eco setups after losses. Stack close angles. Deny sheriff plays.",
    });
  }

  // Tempo exploit
  if (rushRate > 40) {
    exploits.push({
      category: "Tempo",
      pattern: `Rush-heavy style (${rushRate}% under 40s)`,
      counter: `Stack ${primarySite} site with util. Delay with mollies. Off-angles punish predictable rushes.`,
    });
  } else if (avgDuration > 60) {
    exploits.push({
      category: "Tempo",
      pattern: `Slow default style (avg ${avgDuration}s)`,
      counter:
        "Aggressive lurks disrupt setups. Force fights before 1:30. Deny post-plant time.",
    });
  }

  // Site bias exploit
  if (topSiteRate > 60) {
    exploits.push({
      category: "Site Bias",
      pattern: `Over-reliance on ${primarySite} (${topSiteRate}%)`,
      counter: `Stack ${primarySite} rounds 3-5. Fake pressure then rotate. Punish predictability.`,
    });
  }

  // First blood exploit
  if (firstKillRate < 45) {
    exploits.push({
      category: "First Bloods",
      pattern: `Weak entry (${firstKillRate.toFixed(1)}% first kill rate)`,
      counter:
        "Aggressive peeks. Contest map control early. Force unfavorable duels.",
    });
  } else if (firstKillRate > 55) {
    exploits.push({
      category: "First Bloods",
      pattern: `Strong entry fragger (${firstKillRate.toFixed(1)}% first kill rate)`,
      counter:
        "Crossfire setups. Don't peek alone. Trade immediately if teammate dies.",
    });
  }

  // Combat exploit (if available)
  const combatMetrics = metrics.combat_metrics || {};
  if (
    combatMetrics.clutch_win_rate !== undefined &&
    combatMetrics.clutch_win_rate < 30
  ) {
    exploits.push({
      category: "Clutch",
      pattern: `Poor clutch success (${combatMetrics.clutch_win_rate.toFixed(1)}%)`,
      counter:
        "Prioritize man-advantage trades. Isolate players. Force 1vX situations post-plant.",
    });
  }

  if (
    combatMetrics.trade_efficiency !== undefined &&
    combatMetrics.trade_efficiency < 40
  ) {
    exploits.push({
      category: "Trading",
      pattern: `Weak trade efficiency (${combatMetrics.trade_efficiency.toFixed(1)}%)`,
      counter:
        "Target isolated players. Aggressive picks. They won't trade effectively.",
    });
  }

  // Side exploit
  const sideMetrics = metrics.side_metrics || {};
  if (
    sideMetrics.attack_win_rate !== undefined &&
    sideMetrics.defense_win_rate !== undefined
  ) {
    if (sideMetrics.attack_win_rate < 45) {
      exploits.push({
        category: "Attack Side",
        pattern: `Weak attack (${sideMetrics.attack_win_rate.toFixed(1)}% WR)`,
        counter:
          "Force attacker side. Deny space. Make them initiate unfavorable fights.",
      });
    }
    if (sideMetrics.defense_win_rate < 45) {
      exploits.push({
        category: "Defense Side",
        pattern: `Weak defense (${sideMetrics.defense_win_rate.toFixed(1)}% WR)`,
        counter:
          "Aggressive site hits. Fast executes. Overwhelm defensive setups early.",
      });
    }
  }

  // Round type exploits
  const roundPerf = metrics.round_type_performance;
  if (roundPerf) {
    if (roundPerf.pistol.win_rate < 40) {
      exploits.push({
        category: "Pistol Rounds",
        pattern: `Weak pistol rounds (${roundPerf.pistol.win_rate}% WR)`,
        counter:
          "Coordinate pistol executes. Free eco advantage if you win pistols.",
      });
    }
    if (roundPerf.eco.win_rate > 35) {
      exploits.push({
        category: "Eco Rounds",
        pattern: `Strong eco rounds (${roundPerf.eco.win_rate}% WR)`,
        counter:
          "Don't underestimate their ecos. Play safe angles. Avoid sheriff duels.",
      });
    }
  }

  const mapCounterData = normalizedTopMap
    ? MAP_COUNTERS[normalizedTopMap]
    : null;

  return (
    <div className="space-y-6">
      {/* Executive Summary */}
      <SectionCard title="Counter-Strategy Executive Summary">
        <div className="space-y-4">
          <div className="cut-corner border border-cyan/30 bg-cyan/10 p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Team Identity
            </h4>
            <p className="text-sm text-white">
              {teamName} plays a{" "}
              <span className="text-cyan">{aggressionStyle}</span> style with{" "}
              <span className="text-cyan">{rushRate}%</span> rush rate. Primary
              site bias: <span className="text-cyan">{primarySite}</span> (
              {topSiteRate}%). Entry power:{" "}
              <span className={firstKillRate >= 50 ? "text-green" : "text-red"}>
                {firstKillRate.toFixed(1)}% first kill rate
              </span>
              .
            </p>
          </div>

          {insights.how_to_win && (
            <div className="cut-corner border border-violet/30 bg-violet/10 p-4">
              <h4 className="text-sm uppercase tracking-[0.2em] text-violet mb-2">
                AI-Generated Insights
              </h4>
              <div className="text-sm text-muted prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0 prose-table:border prose-table:border-white/20 prose-th:border prose-th:border-white/20 prose-th:px-2 prose-th:py-1 prose-th:bg-white/5 prose-td:border prose-td:border-white/20 prose-td:px-2 prose-td:py-1">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                >
                  {insights.how_to_win}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Exploit Patterns */}
      <SectionCard title="Identified Exploitable Patterns">
        {exploits.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {exploits.map((exploit, i) => (
              <div
                key={i}
                className="cut-corner border border-red/30 bg-red/5 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs uppercase tracking-[0.2em] text-red">
                    {exploit.category}
                  </span>
                  <span className="text-xs text-red">⚠️ EXPLOIT</span>
                </div>
                <div className="text-sm text-white mb-2">{exploit.pattern}</div>
                <div className="text-sm text-muted">
                  <span className="text-cyan">→</span> {exploit.counter}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted text-sm">
            No major exploitable patterns detected. Team shows balanced play
            across metrics.
          </p>
        )}
      </SectionCard>

      {/* Agent Counters */}
      {topAgents.length > 0 && (
        <SectionCard title="Agent Counter Matrix">
          <div className="space-y-4">
            <p className="text-sm text-muted mb-4">
              Based on their top {topAgents.length} most-picked agents.
              Counter-pick and strategy recommendations:
            </p>
            {topAgents.map((agent) => {
              const counter = AGENT_COUNTERS[agent];
              if (!counter) return null;
              return (
                <div
                  key={agent}
                  className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <h4 className="text-sm uppercase tracking-[0.2em] text-white">
                      {agent}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {counter.counters.map((c) => (
                        <span
                          key={c}
                          className="px-2 py-1 rounded text-xs bg-green/20 text-green border border-green/30"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                  <p className="text-sm text-muted">{counter.strategy}</p>
                </div>
              );
            })}
          </div>
        </SectionCard>
      )}

      {/* Map-Specific Counters */}
      {mapCounterData && (
        <SectionCard
          title={`Map-Specific: ${normalizedTopMap} Counter-Strategy`}
        >
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
                <h4 className="text-xs uppercase tracking-[0.2em] text-red mb-2">
                  Weakness
                </h4>
                <p className="text-sm text-white">{mapCounterData.weakness}</p>
              </div>
              <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
                <h4 className="text-xs uppercase tracking-[0.2em] text-yellow mb-2">
                  Exploit
                </h4>
                <p className="text-sm text-white">{mapCounterData.exploit}</p>
              </div>
              <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
                <h4 className="text-xs uppercase tracking-[0.2em] text-green mb-2">
                  Setup
                </h4>
                <p className="text-sm text-white">{mapCounterData.setup}</p>
              </div>
            </div>
          </div>
        </SectionCard>
      )}

      {/* All Maps Quick Reference */}
      <SectionCard title="Map Counter Quick Reference">
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {Object.entries(MAP_COUNTERS).map(([map, counter]) => (
            <div
              key={map}
              className={`cut-corner border p-3 ${
                map === normalizedTopMap
                  ? "border-cyan/50 bg-cyan/10"
                  : "border-ethereal bg-[rgba(15,18,25,0.7)]"
              }`}
            >
              <h5 className="text-sm font-medium text-white mb-1">{map}</h5>
              <p className="text-xs text-muted">{counter.exploit}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Coaching Recommendations */}
      <SectionCard title="Coaching Action Items">
        <div className="space-y-3">
          <div className="cut-corner border border-cyan/30 bg-cyan/10 p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Pre-Match Prep
            </h4>
            <ul className="space-y-2 text-sm text-muted">
              <li className="flex items-start gap-2">
                <span className="text-cyan">1.</span>
                <span>
                  Review their top {topAgents.length || "5"} agents. Practice
                  counter-comps in scrims.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-cyan">2.</span>
                <span>
                  {normalizedTopMap
                    ? `Study ${normalizedTopMap} - their best map. Practice specific site counters.`
                    : "Analyze their map pool and prepare counter-strats for each."}
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-cyan">3.</span>
                <span>
                  {rushRate > 40
                    ? "Prepare anti-rush setups with util stacks. Practice quick trades."
                    : "Practice early aggression to disrupt slow defaults."}
                </span>
              </li>
            </ul>
          </div>

          <div className="cut-corner border border-violet/30 bg-violet/10 p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-violet mb-2">
              In-Game Adjustments
            </h4>
            <ul className="space-y-2 text-sm text-muted">
              <li className="flex items-start gap-2">
                <span className="text-violet">•</span>
                <span>
                  If they force buy after losses, stack anti-eco angles and deny
                  sheriff plays.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-violet">•</span>
                <span>
                  On defense, read their site bias ({primarySite} {topSiteRate}
                  %) and stack rounds 4-6.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-violet">•</span>
                <span>
                  Target their weak side (
                  {sideMetrics.attack_win_rate !== undefined &&
                  sideMetrics.defense_win_rate !== undefined &&
                  sideMetrics.attack_win_rate < sideMetrics.defense_win_rate
                    ? "attack"
                    : "defense"}
                  ) for momentum swings.
                </span>
              </li>
            </ul>
          </div>

          <div className="cut-corner border border-green/30 bg-green/10 p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-green mb-2">
              VOD Review Focus
            </h4>
            <ul className="space-y-2 text-sm text-muted">
              <li className="flex items-start gap-2">
                <span className="text-green">→</span>
                <span>
                  Watch how they handle post-plant (clutch WR:{" "}
                  {combatMetrics.clutch_win_rate?.toFixed(1) || "N/A"}%)
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green">→</span>
                <span>
                  Study their utility usage patterns on{" "}
                  {normalizedTopMap || "their best map"}
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green">→</span>
                <span>
                  Identify their IGL&apos;s tendencies (likely tied to{" "}
                  {aggressionStyle} playstyle)
                </span>
              </li>
            </ul>
          </div>
        </div>
      </SectionCard>

      {/* Player-Specific Counters */}
      {metrics.player_tendencies && metrics.player_tendencies.length > 0 && (
        <SectionCard title="Player-Specific Counters">
          <div className="grid gap-4 md:grid-cols-2">
            {metrics.player_tendencies.slice(0, 4).map((player) => {
              const agent = player.top_agent || "Unknown";
              const counter =
                agent !== "Unknown" ? AGENT_COUNTERS[agent] : null;
              const isHighKD = (player.kd_ratio || 0) > 1.1;
              const isHighFK = (player.first_kill_rate || 0) > 15;

              return (
                <div
                  key={player.player}
                  className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm text-white">{player.player}</h4>
                    <span className="text-xs text-cyan">{agent}</span>
                  </div>
                  <div className="space-y-1 text-xs text-muted">
                    {isHighKD && (
                      <p>
                        <span className="text-red">⚠️</span> High K/D (
                        {player.kd_ratio?.toFixed(2)}) - Don&apos;t peek alone
                      </p>
                    )}
                    {isHighFK && (
                      <p>
                        <span className="text-red">⚠️</span> Entry threat (
                        {player.first_kill_rate?.toFixed(1)}% FK) - Crossfire
                        setup
                      </p>
                    )}
                    {counter && (
                      <p className="text-green">
                        Counter with: {counter.counters.join(", ")}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </SectionCard>
      )}
    </div>
  );
}
