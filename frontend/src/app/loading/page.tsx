"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function LoadingContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const teamName = (searchParams.get("team") ?? "").trim();
  const matchLimit = searchParams.get("match_limit") ?? "20";
  const mapFilter = searchParams.get("map_filter");
  const gameTitle = searchParams.get("game_title") ?? "VALORANT";

  useEffect(() => {
    if (!teamName) {
      return;
    }
    const next = new URL("/dashboard", window.location.origin);
    next.searchParams.set("team", teamName);
    next.searchParams.set("match_limit", matchLimit);
    next.searchParams.set("game_title", gameTitle);
    if (mapFilter) {
      next.searchParams.set("map_filter", mapFilter);
    }
    router.replace(next.toString());
  }, [gameTitle, mapFilter, matchLimit, router, teamName]);

  const currentError = teamName ? null : "Missing team name for scouting.";
  const display = currentError
    ? { label: "❌ Unable to generate report", message: currentError }
    : {
        label: "⏭️ Redirecting to live dashboard",
        message: "Opening the live metrics stream now.",
      };
  const percent = currentError ? 0 : 12;

  return (
    <div className="relative min-h-screen bg-void">
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="pointer-events-none absolute inset-0 noise-overlay opacity-70" />

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="h-96 w-96 animate-pulse rounded-full bg-[rgba(6,182,212,0.1)] blur-3xl" />
        </div>
        <div className="absolute left-1/4 top-1/4 animate-pulse delay-1000">
          <div className="h-64 w-64 rounded-full bg-[rgba(109,40,217,0.1)] blur-2xl" />
        </div>
        <div className="absolute right-1/4 bottom-1/4 animate-pulse delay-2000">
          <div className="h-48 w-48 rounded-full bg-[rgba(255,70,85,0.1)] blur-xl" />
        </div>
      </div>

      <div className="relative z-10 flex min-h-screen items-center justify-center px-6">
        <div className="w-full max-w-2xl space-y-8 text-center data-stream">
          {/* Loading spinner */}
          <div className="relative mx-auto h-24 w-24">
            <div className="absolute inset-0 animate-spin rounded-full border-4 border-[rgba(6,182,212,0.2)]" />
            <div
              className="absolute inset-0 animate-spin rounded-full border-4 border-transparent border-t-cyan"
              style={{ animationDuration: "1.5s" }}
            />
            <div className="absolute inset-2 flex items-center justify-center">
              <div className="h-4 w-4 animate-pulse rounded-full bg-cyan" />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.3em] text-muted">
              Vantage Point Live Scout
            </p>
            <h1 className="text-4xl text-white md:text-5xl font-display">
              Generating tactical report
            </h1>
            <p className="text-xl text-muted">
              for{" "}
              <span className="text-cyan">{teamName || "your opponent"}</span>
            </p>
          </div>

          <div className="cut-corner border border-ethereal-strong bg-glass p-8 shadow-2xl scan-line">
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-muted">
              <span className="flex items-center gap-2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-cyan" />
                {display.label}
              </span>
              <span className="font-mono">{percent}%</span>
            </div>

            <div className="relative mt-6 h-4 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan/20 to-violet/20" />
              <div
                className="relative h-full rounded-full bg-gradient-to-r from-cyan to-violet transition-all duration-700 ease-out"
                style={{ width: `${percent}%` }}
              />
            </div>

            <p className="mt-6 text-sm text-muted">{display.message}</p>
          </div>

          {currentError && (
            <div className="space-y-4">
              <div className="cut-corner border border-red/30 bg-red/10 p-4">
                <p className="text-sm text-red">{currentError}</p>
              </div>
              <button
                type="button"
                onClick={() => router.replace("/")}
                className="inline-flex items-center justify-center gap-2 cut-corner border border-ethereal px-6 py-3 text-sm uppercase tracking-[0.2em] text-muted hover:text-white transition-colors"
              >
                Return to Dashboard
              </button>
            </div>
          )}

          {/* Subtle hint text */}
          {!currentError && (
            <p className="text-xs text-muted opacity-60">
              This usually takes 15-30 seconds. The report will open
              automatically.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function LoadingFallback() {
  return (
    <div className="relative min-h-screen bg-void flex items-center justify-center">
      <div className="text-muted">Loading...</div>
    </div>
  );
}

export default function LoadingPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <LoadingContent />
    </Suspense>
  );
}
