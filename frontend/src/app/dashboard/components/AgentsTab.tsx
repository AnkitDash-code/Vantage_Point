"use client";

import { useEffect, useMemo, useState } from "react";
import { StatCard, SectionCard, DataTable, ProgressBar } from "./StatCards";
import type { MetricsSummary } from "../types";
import { getAgentImage, getFallbackImage } from "./imageMaps";

interface AgentsTabProps {
  metrics: MetricsSummary;
}

// Agent role mapping
const AGENT_ROLES: Record<string, string> = {
  Jett: "Duelist",
  Raze: "Duelist",
  Reyna: "Duelist",
  Phoenix: "Duelist",
  Neon: "Duelist",
  Yoru: "Duelist",
  Iso: "Duelist",
  Sova: "Initiator",
  Breach: "Initiator",
  Skye: "Initiator",
  "KAY/O": "Initiator",
  Fade: "Initiator",
  Gekko: "Initiator",
  Omen: "Controller",
  Brimstone: "Controller",
  Viper: "Controller",
  Astra: "Controller",
  Harbor: "Controller",
  Clove: "Controller",
  Sage: "Sentinel",
  Cypher: "Sentinel",
  Killjoy: "Sentinel",
  Chamber: "Sentinel",
  Deadlock: "Sentinel",
  Vyse: "Sentinel",
};

const ROLE_COLORS: Record<string, string> = {
  Duelist: "text-red",
  Initiator: "text-cyan",
  Controller: "text-violet",
  Sentinel: "text-green",
};

// Case-insensitive lookup helper
const findRole = (name: string): string => {
  const normalized = name.trim().toLowerCase();
  const found = Object.keys(AGENT_ROLES).find(
    (k) => k.toLowerCase() === normalized,
  );
  return found ? AGENT_ROLES[found] : "Unknown";
};

export function AgentsTab({ metrics }: AgentsTabProps) {
  const agentComp = metrics.agent_composition ?? [];
  const roleDist = metrics.role_distribution ?? {};
  const recentComps = metrics.recent_compositions ?? {
    overall: [],
    most_recent: [],
  };
  const playerTendencies = metrics.player_tendencies ?? [];
  const [agentIcons, setAgentIcons] = useState<Record<string, string>>({});

  useEffect(() => {
    let active = true;
    fetch("https://valorant-api.com/v1/agents?isPlayableCharacter=true")
      .then((response) => response.json())
      .then((payload) => {
        if (!active || !payload?.data) return;
        const mapped: Record<string, string> = {};
        payload.data.forEach(
          (agent: { displayName: string; displayIcon: string }) => {
            if (agent?.displayName && agent?.displayIcon) {
              mapped[agent.displayName] = agent.displayIcon.replace(
                /^http:\/\//,
                "https://",
              );
            }
          },
        );
        setAgentIcons(mapped);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const resolveAgentImage = useMemo(
    () => (name: string) => {
      // Case-insensitive lookup in fetched icons
      const normalizedName = name.trim().toLowerCase();
      const foundKey = Object.keys(agentIcons).find(
        (k) => k.toLowerCase() === normalizedName,
      );
      return foundKey ? agentIcons[foundKey] : getAgentImage(name);
    },
    [agentIcons],
  );

  // All agents cards
  const topAgents = agentComp.map((a) => ({
    label: a.agent,
    value: `${a.pick_rate}%`,
    hint: `Role: ${findRole(a.agent)}`,
    percent: a.pick_rate,
    accent: "cyan" as const,
    imageUrl: resolveAgentImage(a.agent),
    imageAlt: `${a.agent} portrait`,
  }));

  // Agent composition table
  const agentRows = agentComp.map((a) => [
    <div key={a.agent} className="flex items-center gap-2">
      <img
        src={resolveAgentImage(a.agent)}
        alt={`${a.agent} portrait`}
        className="h-6 w-6 rounded object-cover border border-ethereal/60"
        loading="lazy"
        onError={(event) => {
          event.currentTarget.src = getFallbackImage(a.agent, "06b6d4");
        }}
      />
      <span>{a.agent}</span>
    </div>,
    findRole(a.agent),
    `${a.pick_rate}%`,
  ]);

  // Role distribution
  const roleData = Object.entries(roleDist)
    .sort((a, b) => b[1] - a[1])
    .map(([role, rate]) => ({ role, rate }));

  // Recent compositions
  const compData = recentComps.overall.slice(0, 5);

  // Player agent preferences
  const playerAgentData = playerTendencies
    .filter((p) => p.top_agent)
    .map((p) => [
      p.player,
      p.top_agent ?? "—",
      `${p.top_agent_rate?.toFixed(1) ?? 0}%`,
    ]);

  return (
    <div className="space-y-6">
      {/* All Agents Grid */}
      <SectionCard title="Agent Pick Rates">
        <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {topAgents.map((stat) => (
            <StatCard key={stat.label} {...stat} />
          ))}
        </div>
      </SectionCard>

      {/* Role Distribution & Agent List */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard title="Role Distribution">
          <div className="space-y-4">
            {roleData.map(({ role, rate }) => (
              <div key={role} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className={ROLE_COLORS[role] ?? "text-muted"}>
                    {role}
                  </span>
                  <span className="text-white">{rate}%</span>
                </div>
                <ProgressBar
                  value={rate}
                  accent={
                    role === "Duelist"
                      ? "red"
                      : role === "Initiator"
                        ? "cyan"
                        : role === "Controller"
                          ? "violet"
                          : "green"
                  }
                  showValue={false}
                />
              </div>
            ))}

            <div className="mt-4 pt-4 border-t border-ethereal">
              <h4 className="text-xs uppercase tracking-[0.2em] text-muted mb-3">
                Role Balance
              </h4>
              <div className="flex h-6 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
                {roleData.map(({ role, rate }) => {
                  const colors: Record<string, string> = {
                    Duelist: "bg-[rgba(255,70,85,0.5)]",
                    Initiator: "bg-[rgba(6,182,212,0.5)]",
                    Controller: "bg-[rgba(109,40,217,0.5)]",
                    Sentinel: "bg-[rgba(34,197,94,0.5)]",
                  };
                  return (
                    <div
                      key={role}
                      className={colors[role] ?? "bg-gray-500"}
                      style={{ width: `${rate}%` }}
                      title={`${role}: ${rate}%`}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Agent Pick Rates">
          <div className="max-h-80 overflow-y-auto">
            <DataTable
              headers={["Agent", "Role", "Pick Rate"]}
              rows={agentRows}
            />
          </div>
        </SectionCard>
      </div>

      {/* Recent Compositions */}
      <SectionCard title="Team Compositions">
        <div className="space-y-4">
          {recentComps.most_recent && recentComps.most_recent.length > 0 && (
            <div className="cut-corner border border-cyan/30 bg-cyan/10 p-4 mb-4">
              <h4 className="text-xs uppercase tracking-[0.2em] text-cyan mb-2">
                Most Recent Comp
              </h4>
              <div className="flex flex-wrap gap-2">
                {recentComps.most_recent.map((agent) => (
                  <span
                    key={agent}
                    className={`px-3 py-1 rounded text-sm ${
                      ROLE_COLORS[findRole(agent)] ?? "text-muted"
                    } bg-[rgba(255,255,255,0.05)] border border-ethereal`}
                  >
                    {agent}
                  </span>
                ))}
              </div>
            </div>
          )}

          {compData.length > 0 ? (
            <div className="space-y-3">
              {compData.map((comp, i) => (
                <div
                  key={comp.composition.join("-")}
                  className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs uppercase tracking-[0.2em] text-muted">
                      Composition #{i + 1}
                    </span>
                    <span className="text-sm text-cyan">{comp.pick_rate}%</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {comp.composition.map((agent) => (
                      <span
                        key={agent}
                        className={`px-2 py-1 rounded text-xs ${
                          ROLE_COLORS[findRole(agent)] ?? "text-muted"
                        } bg-[rgba(255,255,255,0.05)]`}
                      >
                        {agent}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm">No composition data available</p>
          )}
        </div>
      </SectionCard>

      {/* Player Agent Preferences */}
      {playerAgentData.length > 0 && (
        <SectionCard title="Player Agent Preferences">
          <DataTable
            headers={["Player", "Main Agent", "Pick Rate"]}
            rows={playerAgentData}
          />
        </SectionCard>
      )}

      {/* Agent Insights */}
      <SectionCard title="Composition Analysis">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Flex Potential
            </h4>
            <p className="text-sm text-muted">
              {agentComp.length > 10
                ? `Wide agent pool (${agentComp.length} agents used). Team can adapt compositions.`
                : `Limited agent pool (${agentComp.length} agents). Predictable compositions.`}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Role Focus
            </h4>
            <p className="text-sm text-muted">
              {roleData.length > 0 && roleData[0].rate > 30
                ? `${roleData[0].role}-heavy team (${roleData[0].rate}%). Consider exploiting role gaps.`
                : "Balanced role distribution across compositions."}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Signature Agent
            </h4>
            <p className="text-sm text-muted">
              {agentComp.length > 0
                ? `${agentComp[0].agent} is the most picked (${agentComp[0].pick_rate}%). Key to their strategy.`
                : "No clear signature agent."}
            </p>
          </div>
          <div className="cut-corner border border-ethereal bg-[rgba(15,18,25,0.7)] p-4">
            <h4 className="text-sm uppercase tracking-[0.2em] text-cyan mb-2">
              Counter Strategy
            </h4>
            <p className="text-sm text-muted">
              {agentComp.length > 0 &&
              findRole(agentComp[0].agent) === "Duelist"
                ? "Duelist-focused. Consider heavy utility to slow entries."
                : agentComp.length > 0 &&
                    findRole(agentComp[0].agent) === "Controller"
                  ? "Controller-reliant. Early pressure can disrupt setups."
                  : "Balanced approach. Adapt based on map and half."}
            </p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
