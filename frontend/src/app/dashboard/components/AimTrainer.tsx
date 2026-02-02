"use client";

import { useEffect, useState, useRef, useCallback } from "react";

interface Target {
  id: number;
  x: number;
  y: number;
  createdAt: number;
  ttl: number;
  params: {
    startSize: number;
    shrinkSpeed: number;
    minSize: number;
    multiplier: number;
    theme: "standard" | "bonus";
  };
}

interface AimTrainerProps {
  onAutoClose?: () => void;
  isLoading?: boolean;
  progress?: number;
  statusLabel?: string;
}

export function AimTrainer({
  isLoading = false,
  progress = 0,
  statusLabel = "Processing Match Data...",
}: AimTrainerProps) {
  const [targets, setTargets] = useState<Target[]>([]);
  const [score, setScore] = useState(0);
  const [misses, setMisses] = useState(0);
  const [streak, setStreak] = useState(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  const [gameActive, setGameActive] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef<number | null>(null);
  const cleanupRef = useRef<number | null>(null);
  const nextId = useRef(0);
  const lastSpawnTime = useRef(0);

  // Sound effects (simulated with visuals for now, or use Audio if assets existed)

  const spawnTarget = useCallback(() => {
    if (!containerRef.current) return;
    const { clientWidth, clientHeight } = containerRef.current;

    // Margin to keep circles inside
    const margin = 50;
    const maxX = clientWidth - margin * 2;
    const maxY = clientHeight - margin * 2;

    const x = Math.random() * maxX + margin;
    const y = Math.random() * maxY + margin;

    const isBonus = Math.random() < 0.2;
    const newTarget: Target = {
      id: nextId.current++,
      x,
      y,
      createdAt: Date.now(),
      ttl: 2000 + Math.random() * 1200,
      params: {
        startSize: 70 + Math.random() * 50,
        shrinkSpeed: 0.45 + Math.random() * 0.6,
        minSize: 22,
        multiplier: isBonus ? 2 : 1,
        theme: isBonus ? "bonus" : "standard",
      },
    };

    setTargets((prev) => [...prev, newTarget]);
  }, []);

  const handleClick = (id: number) => {
    setTargets((prev) => {
      const target = prev.find((t) => t.id === id);
      const multiplier = target?.params.multiplier ?? 1;
      setScore((s) => s + multiplier);
      setStreak((s) => {
        const next = s + 1;
        setBestStreak((b) => Math.max(b, next));
        return next;
      });
      return prev.filter((t) => t.id !== id);
    });
  };

  const handleMiss = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      setMisses((m) => m + 1);
      setStreak(0);
    }
  };

  useEffect(() => {
    if (!gameActive || !isLoading) {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
      return;
    }

    const updateLoop = (time: number) => {
      if (time - lastSpawnTime.current > 600) {
        // Spawn every 600ms
        if (targets.length < 5) {
          // Max 5 targets
          spawnTarget();
        }
        lastSpawnTime.current = time;
      }
      requestRef.current = requestAnimationFrame(updateLoop);
    };

    requestRef.current = requestAnimationFrame(updateLoop);
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
    };
  }, [gameActive, isLoading, spawnTarget, targets.length]);

  useEffect(() => {
    if (!gameActive || !isLoading) {
      if (cleanupRef.current) {
        window.clearInterval(cleanupRef.current);
        cleanupRef.current = null;
      }
      return;
    }
    cleanupRef.current = window.setInterval(() => {
      const timestamp = Date.now();
      setNow(timestamp);
      setTargets((prev) => {
        let expired = 0;
        const next = prev.filter((t) => {
          if (timestamp - t.createdAt >= t.ttl) {
            expired += 1;
            return false;
          }
          return true;
        });
        if (expired > 0) {
          setMisses((m) => m + expired);
          setStreak(0);
        }
        return next;
      });
    }, 120);
    return () => {
      if (cleanupRef.current) {
        window.clearInterval(cleanupRef.current);
        cleanupRef.current = null;
      }
    };
  }, [gameActive, isLoading]);

  useEffect(() => {
    if (!isLoading) {
      setGameActive(false);
      setTargets([]);
    } else {
      setGameActive(true);
    }
  }, [isLoading]);

  if (!isLoading) return null;

  const accuracy =
    score + misses > 0 ? Math.round((score / (score + misses)) * 100) : 100;
  const cappedProgress = Math.min(100, Math.max(0, progress));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="w-[800px] h-[600px] bg-zinc-900 border border-zinc-700 rounded-xl overflow-hidden shadow-2xl relative flex flex-col">
        {/* Header */}
        <div className="h-14 border-b border-zinc-800 bg-zinc-900 flex items-center justify-between px-6 z-10">
          <div className="flex items-center gap-4">
            <h2 className="text-red-500 font-bold uppercase tracking-wider font-mono">
              Warmup Module
            </h2>
            <div className="h-4 w-[1px] bg-zinc-700" />
            <div className="text-zinc-400 text-sm animate-pulse">
              {statusLabel}
            </div>
          </div>
          <div className="flex gap-6 font-mono">
            <div className="flex flex-col items-center leading-none">
              <span className="text-xs text-zinc-500 uppercase">Score</span>
              <span className="text-xl font-bold text-white">{score}</span>
            </div>
            <div className="flex flex-col items-center leading-none">
              <span className="text-xs text-zinc-500 uppercase">Streak</span>
              <span className="text-xl font-bold text-cyan-300">x{streak}</span>
            </div>
            <div className="flex flex-col items-center leading-none">
              <span className="text-xs text-zinc-500 uppercase">Misses</span>
              <span className="text-xl font-bold text-red-500">{misses}</span>
            </div>
            <div className="flex flex-col items-center leading-none">
              <span className="text-xs text-zinc-500 uppercase">Accuracy</span>
              <span className="text-xl font-bold text-white">{accuracy}%</span>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="h-3 bg-zinc-800">
          <div
            className="h-full bg-gradient-to-r from-cyan-400 via-red-500 to-violet-500 transition-all duration-500"
            style={{ width: `${cappedProgress}%` }}
          />
        </div>

        {/* Game Area */}
        <div
          ref={containerRef}
          className="flex-1 relative cursor-crosshair bg-[url('/grid.svg')] bg-opacity-10"
          onClick={handleMiss}
        >
          {/* Targets */}
          {targets.map((target) => {
            const elapsed = now - target.createdAt;
            const size = Math.max(
              target.params.minSize,
              target.params.startSize - elapsed * target.params.shrinkSpeed,
            );
            const isBonus = target.params.theme === "bonus";
            return (
              <button
                key={target.id}
                onClick={(e) => {
                  e.stopPropagation();
                  handleClick(target.id);
                }}
                className={`absolute rounded-full transition-transform active:scale-95 border-2 border-white/30 ${
                  isBonus
                    ? "bg-cyan-400 hover:bg-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.8)]"
                    : "bg-red-500 hover:bg-red-400 shadow-[0_0_15px_rgba(239,68,68,0.7)]"
                }`}
                style={{
                  left: target.x,
                  top: target.y,
                  width: size,
                  height: size,
                  transform: "translate(-50%, -50%)",
                  animation: `popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)`,
                }}
              >
                <div className="absolute inset-2 rounded-full border border-black/20" />
                <div
                  className={`absolute inset-[40%] rounded-full ${
                    isBonus ? "bg-cyan-900/50" : "bg-red-900/50"
                  }`}
                />
              </button>
            );
          })}
        </div>

        {/* Footer */}
        <div className="h-10 border-t border-zinc-800 bg-zinc-900/50 flex items-center justify-center text-xs text-zinc-600 uppercase tracking-widest">
          Bonus targets are cyan • Best streak: {bestStreak}
        </div>
      </div>
    </div>
  );
}
