"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

const staggerDelays = [
  "stagger-delay-1",
  "stagger-delay-2",
  "stagger-delay-3",
  "stagger-delay-4",
];

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.7 } },
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};

interface Team {
  name: string;
  slug: string;
  match_count: number;
  file_size?: number;
  has_insights?: boolean;
}

interface Manifest {
  version: string;
  generated_at: string;
  match_limit: number;
  teams: Team[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const USE_PRECOMPUTED = process.env.NEXT_PUBLIC_USE_PRECOMPUTED === "true";
const PRECOMPUTED_BASE_URL =
  process.env.NEXT_PUBLIC_PRECOMPUTED_BASE_URL || "/precomputed";

const previewSnapshot = {
  team: "Sentinels",
  matches: 20,
  siteBias: "A 58%",
  pistolSite: "A 64%",
  bestMap: "Ascent",
  tempo: "Default (57s)",
};

const quickStats = [
  { label: "Rounds parsed", value: "12,840+" },
  { label: "Event signals", value: "240K" },
  { label: "Maps profiled", value: "9" },
  { label: "Patterns flagged", value: "128" },
];

const featureCards = [
  {
    title: "Site Bias Scanner",
    body: "Tracks spike plant locations across maps, highlighting A/B/C site preferences and pistol-round tendencies.",
    stat: "A Site 58% overall",
  },
  {
    title: "Agent Pool Heat",
    body: "Top five agents with pick rates and role balance to reveal comfort comps and flex gaps.",
    stat: "Jett 68% | Omen 61%",
  },
  {
    title: "Role Balance",
    body: "Duelist-heavy lineups signal aggression; stacked controllers suggest late-round executes.",
    stat: "Duelist 38% | Controller 24%",
  },
  {
    title: "Pistol Setups",
    body: "Maps early-round habits to pre-plan anti-rush setups and counter-utility.",
    stat: "Pistol A-hit 64%",
  },
];

const pipeline = [
  {
    title: "GRID Data",
    body: "Pulls recent matches, rounds, and player stats with official VALORANT telemetry.",
  },
  {
    title: "Analyzer",
    body: "Normalizes matches into scouting metrics: pace, economy, roles, sites, and agent pools.",
  },
  {
    title: "RAG Engine",
    body: "Retrieves Valorant strategy context to explain patterns and surface counters.",
  },
  {
    title: "Scouting Report",
    body: "Delivers a concise, coach-ready brief with actionable steps.",
  },
];

function GlitchHeader({ text }: { text: string }) {
  return (
    <span className="glitch font-display" data-text={text}>
      {text}
    </span>
  );
}

function TacticalButton({
  label,
  href,
  variant = "primary",
  type = "button",
  disabled = false,
  onClick,
}: {
  label: string;
  href?: string;
  variant?: "primary" | "ghost";
  type?: "button" | "submit";
  disabled?: boolean;
  onClick?: () => void;
}) {
  const base =
    "group relative inline-flex items-center justify-center gap-2 px-6 py-3 text-sm uppercase tracking-[0.22em] transition-all";
  const styles =
    variant === "primary"
      ? "cut-corner border border-ethereal-strong text-cyan glow-cyan hover:text-white"
      : "cut-corner border border-ethereal text-muted hover:text-white";

  if (href) {
    return (
      <a href={href} className={`${base} ${styles} overflow-hidden`}>
        <span className="absolute inset-0 -translate-x-full bg-[rgba(6,182,212,0.2)] transition-transform duration-300 group-hover:translate-x-0" />
        <span className="relative">{label}</span>
      </a>
    );
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${styles} overflow-hidden disabled:cursor-not-allowed disabled:opacity-50`}
    >
      <span className="absolute inset-0 -translate-x-full bg-[rgba(6,182,212,0.2)] transition-transform duration-300 group-hover:translate-x-0" />
      <span className="relative">{label}</span>
    </button>
  );
}

export default function Home() {
  const [teamName, setTeamName] = useState("");
  const [matchLimit, setMatchLimit] = useState("20");
  const [availableTeams, setAvailableTeams] = useState<Team[]>([]);
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [typewriterText, setTypewriterText] = useState(
    "INITIALIZING SCOUTING PROTOCOL...",
  );
  const [typewriterIndex, setTypewriterIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const router = useRouter();

  const matchDelayHint = (() => {
    const limit = Number(matchLimit) || 20;
    if (limit <= 5) {
      return "1-5 matches: Fastest response.";
    }
    if (limit <= 10) {
      return "5-10 matches: Quick response.";
    }
    if (limit <= 20) {
      return "10-20 matches: Moderate delay.";
    }
    return "20+ matches: Longer delay expected.";
  })();

  useEffect(() => {
    const phrases = [
      "INITIALIZING SCOUTING PROTOCOL...",
      "SEARCHING GRID DATABASE...",
      "ANALYZE. ADAPT. WIN.",
    ];
    const currentPhrase = phrases[typewriterIndex % phrases.length];
    const timeout = setTimeout(
      () => {
        if (!isDeleting) {
          const nextText = currentPhrase.slice(0, typewriterText.length + 1);
          setTypewriterText(nextText);
          if (nextText === currentPhrase) {
            setTimeout(() => setIsDeleting(true), 900);
          }
        } else {
          const nextText = currentPhrase.slice(0, typewriterText.length - 1);
          setTypewriterText(nextText);
          if (nextText.length === 0) {
            setIsDeleting(false);
            setTypewriterIndex((prev) => prev + 1);
          }
        }
      },
      isDeleting ? 40 : 70,
    );

    return () => clearTimeout(timeout);
  }, [isDeleting, typewriterIndex, typewriterText]);

  // Fetch available teams on mount (precomputed or live)
  useEffect(() => {
    const fetchTeams = async () => {
      try {
        if (USE_PRECOMPUTED) {
          // Load from precomputed manifest
          let response = await fetch(`${PRECOMPUTED_BASE_URL}/manifest.json`);
          if (response.ok) {
            const manifest: Manifest = await response.json();
            setAvailableTeams(manifest.teams || []);
            return;
          }

          // Fallback to backend precomputed endpoint
          response = await fetch(`${API_BASE_URL}/api/precomputed/teams`);
          if (response.ok) {
            const data = await response.json();
            setAvailableTeams(data.teams || []);
            return;
          }

          throw new Error("Precomputed teams not found");
        } else {
          // Load from live backend
          const response = await fetch(`${API_BASE_URL}/api/teams`);
          const data = await response.json();
          setAvailableTeams(data.teams || []);
        }
      } catch (error) {
        console.error("Failed to fetch teams:", error);
        setError("Unable to load teams. Check precomputed data or backend.");
      } finally {
        setLoadingTeams(false);
      }
    };
    fetchTeams();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const elements = Array.from(
      document.querySelectorAll("[data-scroll]"),
    ) as HTMLElement[];
    if (!elements.length) {
      return;
    }
    if (!("IntersectionObserver" in window)) {
      elements.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.2 },
    );
    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!teamName.trim()) {
      setError("Enter a team name to scout.");
      return;
    }

    // In precomputed mode, verify team exists in manifest
    if (USE_PRECOMPUTED) {
      const teamSlug = teamName.trim().toLowerCase().replace(/\s+/g, "_");
      const teamExists = availableTeams.some(
        (t) =>
          t.slug === teamSlug ||
          t.name.toLowerCase() === teamName.trim().toLowerCase(),
      );
      if (!teamExists) {
        setError(
          "Team not found in precomputed data. Select from the dropdown.",
        );
        return;
      }
    }

    setLoading(true);
    setError(null);
    const next = new URL("/dashboard", window.location.origin);
    next.searchParams.set("team", teamName.trim());
    next.searchParams.set("match_limit", matchLimit);
    next.searchParams.set("game_title", "VALORANT");
    if (USE_PRECOMPUTED) {
      next.searchParams.set("precomputed", "true");
    }
    router.push(next.toString());
  };

  return (
    <div className="relative min-h-screen bg-void">
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="pointer-events-none absolute inset-0 noise-overlay opacity-70" />
      <div className="pointer-events-none absolute inset-0 gradient-halo opacity-90" />

      <div className="relative z-10">
        <motion.nav
          className="sticky top-4 z-20 mx-auto flex max-w-6xl items-center justify-between px-6 py-4 backdrop-blur-md bg-[rgba(10,14,20,0.65)] border border-white/10 shadow-[0_12px_35px_rgba(0,0,0,0.35)] cut-corner"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center gap-3 text-sm uppercase tracking-[0.4em] text-muted">
            <span className="h-2 w-2 rounded-full bg-[rgba(6,182,212,0.8)]" />
            Vantage Point
          </div>
          <div className="hidden items-center gap-6 text-xs uppercase tracking-[0.2em] text-muted md:flex">
            <a href="#pipeline" className="hover:text-white">
              Pipeline
            </a>
            <a href="#cta" className="hover:text-white">
              Launch
            </a>
          </div>
          <TacticalButton label="Generate Report" href="#cta" />
        </motion.nav>

        <motion.section
          className="mx-auto flex max-w-6xl flex-col gap-12 px-6 pb-16 pt-10 md:flex-row md:items-center"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.3 }}
        >
          <motion.div className="max-w-xl" variants={fadeUp}>
            <p className="text-xs uppercase tracking-[0.3em] text-muted">
              Cloud9 Scouting Suite
            </p>
            <h1 className="mt-4 text-5xl leading-tight text-white md:text-6xl">
              <GlitchHeader text="Ethereal Tactical" /> scouting reports for
              every match.
            </h1>
            <p className="mt-4 text-sm uppercase tracking-[0.35em] text-cyan font-mono">
              {typewriterText}
              <span className="typewriter-caret" />
            </p>
            <p className="mt-6 text-lg text-muted">
              Vantage Point turns GRID match data into actionable insights,
              highlighting strategies, player habits, and counter-play in one
              concise briefing.
            </p>
            <form
              onSubmit={handleSubmit}
              className="mt-8 flex flex-wrap items-center gap-4"
            >
              <div className="relative group min-w-[260px] flex-1">
                <div className="absolute -inset-1 rounded-lg bg-gradient-to-r from-cyan-400/40 to-blue-600/40 blur opacity-30 transition duration-700 group-hover:opacity-70" />
                <div className="absolute -right-6 -top-6 h-14 w-14 rounded-full border border-cyan-400/30 bg-[radial-gradient(circle_at_center,rgba(19,243,255,0.35),transparent_60%)] spin-slow" />
                <input
                  value={teamName}
                  onChange={(event) => setTeamName(event.target.value)}
                  placeholder={
                    loadingTeams
                      ? "Loading teams..."
                      : "ENTER TARGET TEAM ID..."
                  }
                  list="team-options"
                  disabled={loadingTeams}
                  className="relative h-16 w-full cut-corner border border-cyan-500/30 bg-black/50 px-5 text-xl font-mono text-white placeholder:text-slate-500 focus:border-cyan-400 focus:ring-4 focus:ring-cyan-500/20 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed pulse-border"
                />
              </div>
              <select
                value={matchLimit}
                onChange={(event) => setMatchLimit(event.target.value)}
                className="min-w-[170px] h-16 cut-corner border border-cyan-500/30 bg-black/50 px-4 text-sm font-mono text-white focus:border-cyan-400 focus:ring-4 focus:ring-cyan-500/20 focus:outline-none"
              >
                <option value="5">1-5 matches</option>
                <option value="10">5-10 matches</option>
                <option value="20">10-20 matches</option>
                <option value="60">20+ matches</option>
              </select>
              <datalist id="team-options">
                {availableTeams.map((team) => (
                  <option key={team.slug} value={team.name}>
                    {team.name} ({team.match_count} matches)
                  </option>
                ))}
              </datalist>
              <TacticalButton
                label={loading ? "Generating..." : "Scout Now"}
                type="submit"
                disabled={loading}
              />
            </form>
            <p className="mt-2 text-xs text-muted">{matchDelayHint}</p>
            {error && <p className="mt-3 text-sm text-red">{error}</p>}
            <div className="mt-6 flex flex-wrap gap-4">
              <TacticalButton label="See Pipeline" href="#pipeline" />
              <TacticalButton
                label="Launch Scout"
                href="#cta"
                variant="ghost"
              />
            </div>
          </motion.div>

          <motion.div className="relative w-full max-w-md" variants={fadeUp}>
            <div className="absolute -inset-8 rounded-[36px] bg-[radial-gradient(circle,rgba(0,229,255,0.25),transparent_60%)]" />
            <div className="absolute -inset-6 rounded-[32px] bg-[radial-gradient(circle,rgba(255,70,85,0.2),transparent_60%)]" />
            <div className="cut-corner border border-ethereal-strong bg-glass p-6 shadow-[0_0_30px_rgba(6,182,212,0.15)]">
              <p className="text-xs uppercase tracking-[0.3em] text-muted">
                Live Snapshot
              </p>
              <h2 className="mt-3 text-2xl text-white">
                Opponent: {previewSnapshot.team}
              </h2>
              <div className="mt-6 space-y-4 text-sm text-muted">
                <div className="flex items-center justify-between">
                  <span>Site Bias</span>
                  <span className="text-white">{previewSnapshot.siteBias}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Pistol Play</span>
                  <span className="text-white">
                    {previewSnapshot.pistolSite}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Best Map</span>
                  <span className="text-white">{previewSnapshot.bestMap}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Tempo</span>
                  <span className="text-white">{previewSnapshot.tempo}</span>
                </div>
              </div>
              <div className="mt-6 flex gap-3">
                <span className="text-xs uppercase tracking-[0.3em] text-cyan">
                  RAG Powered
                </span>
                <span className="text-xs uppercase tracking-[0.3em] text-muted">
                  {previewSnapshot.matches} matches
                </span>
              </div>
            </div>
          </motion.div>
        </motion.section>

        <motion.section
          className="mx-auto max-w-6xl px-6 py-10"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="grid gap-4 md:grid-cols-3">
            {[
              {
                title: "Command Center Pulse",
                body: "Live telemetry sync · Threat models hot",
                stat: "Systems: ONLINE",
              },
              {
                title: "Threat Radar",
                body: "Pressure vectors detected across recent series.",
                stat: "Alert Level: AMBER",
              },
              {
                title: "Operator Console",
                body: "Agent pools calibrated · Strategy profiles locked.",
                stat: "Briefing: READY",
              },
            ].map((panel) => (
              <motion.div
                key={panel.title}
                variants={fadeUp}
                className="cut-corner border border-white/10 bg-glass p-5 relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-cyan/10 via-transparent to-violet/10 opacity-70" />
                <div className="relative">
                  <p className="text-xs uppercase tracking-[0.35em] text-muted font-display">
                    {panel.title}
                  </p>
                  <p className="mt-3 text-sm text-muted">{panel.body}</p>
                  <p className="mt-4 text-sm uppercase tracking-[0.3em] text-cyan">
                    {panel.stat}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        <motion.section
          className="mx-auto max-w-6xl px-6 py-10"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="grid gap-4 md:grid-cols-4">
            {quickStats.map((stat) => (
              <motion.div
                key={stat.label}
                variants={fadeUp}
                className="cut-corner border border-ethereal-strong bg-glass p-5 text-center"
              >
                <div className="text-xs uppercase tracking-[0.3em] text-muted">
                  {stat.label}
                </div>
                <div className="mt-3 text-3xl font-semibold text-white">
                  {stat.value}
                </div>
                <div className="mt-2 text-xs uppercase tracking-[0.3em] text-cyan">
                  Live telemetry
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        <motion.section
          className="mx-auto max-w-6xl px-6 py-12"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="grid gap-4 md:grid-cols-2">
            {featureCards.map((card, index) => (
              <motion.div
                key={card.title}
                variants={fadeUp}
                className={`cut-corner border border-ethereal bg-glass p-6 ${staggerDelays[index]} md:p-7`}
              >
                <p className="text-xs uppercase tracking-[0.3em] text-muted">
                  {card.stat}
                </p>
                <h3 className="mt-3 text-2xl text-white">{card.title}</h3>
                <p className="mt-3 text-sm text-muted">{card.body}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        <motion.section
          className="mx-auto max-w-6xl px-6 py-12"
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="cut-corner border border-ethereal-strong bg-glass p-8 md:p-10">
            <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-muted">
                  Threat Matrix
                </p>
                <h2 className="mt-3 text-3xl text-white">
                  Identify pressure points before pistol round.
                </h2>
                <p className="mt-3 text-sm text-muted">
                  Live models flag defaulting cadence, site pressure, and
                  utility cadence for each opponent.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="rounded-xl border border-ethereal bg-[rgba(12,14,20,0.8)] p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">
                    Pressure Index
                  </p>
                  <p className="mt-2 text-2xl text-white">82%</p>
                </div>
                <div className="rounded-xl border border-ethereal bg-[rgba(12,14,20,0.8)] p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">
                    Exec Speed
                  </p>
                  <p className="mt-2 text-2xl text-white">53s</p>
                </div>
                <div className="rounded-xl border border-ethereal bg-[rgba(12,14,20,0.8)] p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">
                    Trade Rate
                  </p>
                  <p className="mt-2 text-2xl text-white">71%</p>
                </div>
                <div className="rounded-xl border border-ethereal bg-[rgba(12,14,20,0.8)] p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">
                    Econ Shift
                  </p>
                  <p className="mt-2 text-2xl text-white">+11%</p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        <motion.section
          id="pipeline"
          className="mx-auto max-w-6xl px-6 py-12"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-muted">
                Pipeline
              </p>
              <h2 className="mt-3 text-3xl text-white">
                From GRID telemetry to coach-ready insight.
              </h2>
            </div>
            <p className="max-w-md text-sm text-muted">
              Each step is optimized for speed: cached data, vectorized
              analytics, and RAG-grounded summaries.
            </p>
          </div>
          <div className="mt-8 grid gap-4 md:grid-cols-4">
            {pipeline.map((step, index) => (
              <motion.div
                key={step.title}
                variants={fadeUp}
                className="cut-corner border border-ethereal bg-glass p-5"
              >
                <div className="text-xs uppercase tracking-[0.3em] text-cyan">
                  0{index + 1}
                </div>
                <h3 className="mt-3 text-xl text-white">{step.title}</h3>
                <p className="mt-2 text-sm text-muted">{step.body}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        <motion.section
          id="cta"
          className="mx-auto max-w-6xl px-6 pb-20 pt-8"
          variants={fadeUp}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="cut-corner border border-ethereal-strong bg-glass p-10 text-center">
            <h2 className="text-3xl text-white">Ready to brief the team?</h2>
            <p className="mt-4 text-sm text-muted">
              Generate a scouting report in seconds. Plug in a team name, select
              a map filter, and get a tactical edge.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-4">
              <TacticalButton label="Launch Scout Mode" href="#" />
              <TacticalButton label="View API Docs" href="#" variant="ghost" />
            </div>
          </div>
        </motion.section>

        <motion.footer
          className="border-t border-ethereal py-6"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true, amount: 0.2 }}
        >
          <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 text-xs uppercase tracking-[0.3em] text-muted md:flex-row">
            <span>Cloud9 x JetBrains Hackathon</span>
            <span>Made by Rock Walker - @AnkitDash-code</span>
            <span>Powered by GRID + RAG</span>
          </div>
        </motion.footer>
      </div>
    </div>
  );
}
