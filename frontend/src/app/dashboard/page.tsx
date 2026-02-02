"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  TabNav,
  type TabId,
  OverviewTab,
  InsightsTab,
  EconomyTab,
  CombatTab,
  MapsTab,
  AgentsTab,
  PlayersTab,
  CountersTab,
  AimTrainer,
} from "./components";
import { getMapImage } from "./components/imageMaps";
import type {
  ProgressState,
  ScoutReport,
  StreamProgress,
  StreamWarning,
  StreamMetrics,
  StreamDone,
  StreamInsightChunk,
  StreamError,
} from "./types";

const USE_PRECOMPUTED = process.env.NEXT_PUBLIC_USE_PRECOMPUTED === "true";
const PRECOMPUTED_BASE_URL =
  process.env.NEXT_PUBLIC_PRECOMPUTED_BASE_URL || "/precomputed";

const stageLabels: Record<string, string> = {
  fetch_matches: "Fetching matches",
  filter_matches: "Filtering data",
  fetch_events: "Loading events",
  analyze: "Analyzing stats",
  rag_init: "Preparing context",
  rag_generate: "Generating insights",
};

function DashboardContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [progress, setProgress] = useState<ProgressState>({
    stage: "initializing",
    progress: 5,
    message: "Preparing scouting pipeline",
  });
  const [report, setReport] = useState<ScoutReport | null>(null);
  const [streamInsights, setStreamInsights] = useState<Record<string, string>>(
    {},
  );
  const [metricsOnly, setMetricsOnly] = useState<StreamMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const [showWarmup, setShowWarmup] = useState(true);
  const hasPayloadRef = useRef(false);

  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";
  const teamName = (searchParams.get("team") ?? "").trim();
  const matchLimit = searchParams.get("match_limit") ?? "20";
  const mapFilter = searchParams.get("map_filter");
  const gameTitle = searchParams.get("game_title") ?? "VALORANT";
  const usePrecomputed =
    searchParams.get("precomputed") === "true" || USE_PRECOMPUTED;

  // Load precomputed data if in precomputed mode
  useEffect(() => {
    if (!usePrecomputed || !teamName) return;

    const loadPrecomputed = async () => {
      try {
        setProgress({
          stage: "loading",
          progress: 30,
          message: "Loading precomputed data...",
        });

        // Convert team name to slug for file lookup
        const teamSlug = teamName.toLowerCase().replace(/\s+/g, "_");
        const response = await fetch(
          `${PRECOMPUTED_BASE_URL}/teams/${teamSlug}.json`,
        );

        if (!response.ok) {
          throw new Error(`Precomputed data not found for ${teamName}`);
        }

        const data: ScoutReport = await response.json();

        setReport(data);
        setStreamInsights(data.insights ?? {});
        setProgress({
          stage: "complete",
          progress: 100,
          message: "Report ready (precomputed)",
        });
        setStreamEnabled(false);
        setShowWarmup(false);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load precomputed data",
        );
        setShowWarmup(false);
      }
    };

    loadPrecomputed();
  }, [usePrecomputed, teamName]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedReport = sessionStorage.getItem("scoutReport");
    if (storedReport) {
      try {
        const parsed = JSON.parse(storedReport) as ScoutReport;
        setReport(parsed);
        setStreamInsights(parsed.insights ?? {});
        setProgress({
          stage: "complete",
          progress: 100,
          message: "Report ready",
        });
        setStreamEnabled(false);
        setShowWarmup(false);
      } catch {
        setError("Failed to read saved report.");
        setShowWarmup(false);
      } finally {
        sessionStorage.removeItem("scoutReport");
      }
    }
    const storedError = sessionStorage.getItem("scoutError");
    if (storedError) {
      setError(storedError);
      setShowWarmup(false);
      sessionStorage.removeItem("scoutError");
    }
  }, []);

  useEffect(() => {
    if (!teamName || !streamEnabled || usePrecomputed) {
      setShowWarmup(false);
      return;
    }

    let active = true;
    hasPayloadRef.current = false;
    setWarning(null);
    const url = new URL(`${apiBase}/api/scout/stream`);
    url.searchParams.set("team_name", teamName);
    url.searchParams.set("match_limit", matchLimit);
    url.searchParams.set("game_title", gameTitle);
    if (mapFilter) {
      url.searchParams.set("map_filter", mapFilter);
    }

    const source = new EventSource(url.toString());
    source.onmessage = (event) => {
      if (!active) {
        return;
      }
      try {
        const payload = JSON.parse(event.data) as
          | StreamProgress
          | StreamWarning
          | StreamMetrics
          | StreamInsightChunk
          | StreamDone
          | StreamError;
        if (payload.type === "progress") {
          setProgress({
            stage: payload.stage,
            progress: payload.progress,
            message: payload.message,
          });
          return;
        }
        if (payload.type === "warning") {
          setWarning(payload.message);
          return;
        }
        if (payload.type === "metrics") {
          hasPayloadRef.current = true;
          setMetricsOnly(payload);
          setStreamInsights({});
          setShowWarmup(false);
          return;
        }
        if (payload.type === "insight_chunk") {
          setStreamInsights((prev) => ({
            ...prev,
            [payload.section]: (prev[payload.section] ?? "") + payload.content,
          }));
          return;
        }
        if (payload.type === "done") {
          hasPayloadRef.current = true;
          setReport(payload.report);
          setStreamInsights(payload.report.insights ?? {});
          setProgress({
            stage: "complete",
            progress: 100,
            message: "Report ready",
          });
          setWarning(null);
          setShowWarmup(false);
          source.close();
          return;
        }
        if (payload.type === "error") {
          setError(payload.message);
          setShowWarmup(false);
          source.close();
        }
      } catch {
        setError("Failed to parse server update.");
        setShowWarmup(false);
        source.close();
      }
    };

    source.onerror = () => {
      if (!active) {
        return;
      }
      if (hasPayloadRef.current) {
        source.close();
        return;
      }
      setError("Lost connection to the scout stream.");
      setShowWarmup(false);
      source.close();
    };

    return () => {
      active = false;
      source.close();
    };
  }, [apiBase, gameTitle, mapFilter, matchLimit, streamEnabled, teamName]);

  const metricsData = report?.metrics ?? metricsOnly?.metrics;
  const matchesAnalyzed =
    report?.matches_analyzed ?? metricsOnly?.matches_analyzed ?? 0;
  const displayTeam =
    (report?.team_name ?? metricsOnly?.team_name ?? teamName) || "Unknown";

  const displayLabel =
    stageLabels[progress.stage] ?? progress.message ?? "Processing";
  const percent = Math.min(100, Math.max(0, progress.progress));

  const insightSource = report?.insights ?? streamInsights;

  const renderActiveTab = () => {
    if (!metricsData) {
      return (
        <div className="flex items-center justify-center h-64 text-muted">
          <div className="text-center">
            <div className="text-lg mb-2">Loading metrics...</div>
            <div className="text-sm">{displayLabel}</div>
          </div>
        </div>
      );
    }

    switch (activeTab) {
      case "overview":
        return (
          <OverviewTab
            metrics={metricsData}
            matchesAnalyzed={matchesAnalyzed}
            teamName={displayTeam}
            insights={insightSource}
          />
        );
      case "insights":
        return (
          <InsightsTab
            insights={insightSource}
            teamName={displayTeam}
            metrics={metricsData}
          />
        );
      case "economy":
        return <EconomyTab metrics={metricsData} />;
      case "combat":
        return <CombatTab metrics={metricsData} />;
      case "maps":
        return <MapsTab metrics={metricsData} />;
      case "agents":
        return <AgentsTab metrics={metricsData} />;
      case "players":
        return <PlayersTab metrics={metricsData} />;
      case "counters":
        return (
          <CountersTab
            metrics={metricsData}
            teamName={displayTeam}
            insights={insightSource}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`relative min-h-screen bg-void ${showWarmup && !error ? "scan-line" : ""}`}
    >
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="pointer-events-none absolute inset-0 noise-overlay opacity-70" />
      <div className="pointer-events-none absolute inset-0 gradient-halo opacity-80" />

      <div className="relative z-10">
        <AimTrainer
          isLoading={showWarmup && !error}
          progress={percent}
          statusLabel={displayLabel}
        />
        {/* Header */}
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
          <div className="flex items-center gap-3 text-xs uppercase tracking-[0.35em] text-muted">
            <span className="h-2 w-2 rounded-full bg-[rgba(6,182,212,0.8)]" />
            Vantage Point Dashboard
          </div>
          <button
            type="button"
            onClick={() => router.push("/")}
            className="cut-corner border border-ethereal px-4 py-2 text-xs uppercase tracking-[0.3em] text-muted hover:text-white transition-colors"
          >
            Back to Landing
          </button>
        </nav>

        {/* Main Content */}
        <section className="mx-auto max-w-7xl px-6 pb-12">
          {/* Team Header Card */}
          <div className="cut-corner border border-white/10 bg-glass glass-sheen p-6 mb-6 relative overflow-hidden hover:border-cyan-400/50 transition-colors duration-300">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan/10 via-transparent to-violet/10" />
            <div className="relative flex flex-wrap items-center justify-between gap-6">
              <div className="flex items-center gap-5">
                <div className="h-16 w-24 rounded-xl border border-ethereal/60 bg-[rgba(255,255,255,0.04)] p-1">
                  <img
                    src={
                      mapFilter ? getMapImage(mapFilter) : getMapImage("Ascent")
                    }
                    alt={mapFilter ? `${mapFilter} map` : "Map preview"}
                    className="h-full w-full object-cover rounded-lg"
                    loading="lazy"
                  />
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.3em] text-muted mb-2">
                    Scouting Report
                  </div>
                  <div className="text-3xl text-white font-light">
                    {displayTeam}
                  </div>
                  <div className="mt-2 text-sm text-muted">
                    {matchesAnalyzed} matches analyzed • {gameTitle}
                    {mapFilter && (
                      <span className="ml-2 text-cyan">• Map: {mapFilter}</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs uppercase tracking-[0.3em] text-muted mb-1">
                  Progress
                </div>
                <div className="text-2xl text-cyan font-mono">{percent}%</div>
                <div className="text-xs text-muted mt-1">{displayLabel}</div>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan to-violet transition-all duration-700"
                style={{ width: `${percent}%` }}
              />
            </div>
            {warning && (
              <div className="mt-4 rounded-lg border border-[rgba(255,192,0,0.35)] bg-[rgba(255,192,0,0.08)] px-4 py-2 text-xs text-[rgba(255,220,140,0.95)]">
                {warning}
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="cut-corner border border-red/30 bg-red/10 p-4 text-sm text-red mb-6">
              {error}
            </div>
          )}

          {/* Tab Navigation */}
          <div className="mb-6">
            <TabNav activeTab={activeTab} onTabChange={setActiveTab} />
          </div>

          {/* Tab Content */}
          <div className="min-h-[500px]">{renderActiveTab()}</div>
        </section>
      </div>
    </div>
  );
}

function LoadingFallback() {
  return (
    <div className="relative min-h-screen bg-void flex items-center justify-center">
      <div className="text-center">
        <div className="text-lg text-muted mb-2">Loading Dashboard...</div>
        <div className="h-2 w-48 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)] mx-auto">
          <div className="h-full w-1/3 rounded-full bg-gradient-to-r from-cyan to-violet animate-pulse" />
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <DashboardContent />
    </Suspense>
  );
}
