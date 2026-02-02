"use client";

import { cn } from "@/lib/utils";
import { getFallbackImage } from "./imageMaps";

type Accent = "cyan" | "violet" | "red" | "green" | "yellow";

interface StatCardProps {
  label: string;
  value: string | number;
  hint?: string;
  percent?: number;
  accent?: Accent;
  className?: string;
  imageUrl?: string;
  imageAlt?: string;
}

const accentStyles: Record<Accent, string> = {
  cyan: "bg-[rgba(6,182,212,0.3)]",
  violet: "bg-[rgba(109,40,217,0.3)]",
  red: "bg-[rgba(255,70,85,0.3)]",
  green: "bg-[rgba(34,197,94,0.3)]",
  yellow: "bg-[rgba(234,179,8,0.3)]",
};

export function StatCard({
  label,
  value,
  hint,
  percent,
  accent = "cyan",
  className,
  imageUrl,
  imageAlt,
}: StatCardProps) {
  return (
    <div
      className={cn(
        "cut-corner border border-white/10 bg-glass p-5 relative overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:border-cyan-400/50 hover:shadow-[0_0_28px_rgba(6,182,212,0.25)]",
        className,
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-cyan/10 via-transparent to-violet/10 opacity-0 transition-opacity duration-300 hover:opacity-100" />
      <div className="relative flex items-start gap-4">
        {imageUrl && (
          <div className="h-14 w-14 shrink-0 rounded-xl border border-ethereal/60 bg-[rgba(255,255,255,0.04)] p-1">
            <img
              src={imageUrl}
              alt={imageAlt ?? label}
              className="h-full w-full object-cover rounded-lg"
              loading="lazy"
              onError={(event) => {
                event.currentTarget.src = getFallbackImage(imageAlt ?? label);
              }}
            />
          </div>
        )}
        <div className="flex-1">
          <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-muted font-display">
            <span>{label}</span>
            {percent !== undefined && <span>{percent.toFixed(1)}%</span>}
          </div>
          <p className="mt-2 text-xl text-white font-semibold">{value}</p>
          {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
        </div>
      </div>
      {percent !== undefined && (
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
          <div
            className={cn("h-full rounded-full", accentStyles[accent])}
            style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
          />
        </div>
      )}
    </div>
  );
}

interface DataTableProps {
  headers: string[];
  rows: React.ReactNode[][];
  className?: string;
}

export function DataTable({ headers, rows, className }: DataTableProps) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-ethereal text-xs uppercase tracking-[0.2em] text-muted bg-[rgba(255,255,255,0.02)]">
            {headers.map((header) => (
              <th key={header} className="px-4 py-3 font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-ethereal/30 hover:bg-white/5 transition-colors"
            >
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-3 text-muted">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface SectionCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function SectionCard({ title, children, className }: SectionCardProps) {
  return (
    <div
      className={cn(
        "cut-corner border border-white/10 bg-glass p-6 relative overflow-hidden hover:border-cyan-400/50 transition-colors duration-300",
        className,
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-cyan/5 via-transparent to-violet/5 opacity-60" />
      <div className="relative text-xs uppercase tracking-[0.35em] text-muted mb-4 font-display">
        {title}
      </div>
      {children}
    </div>
  );
}

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  accent?: Accent;
  showValue?: boolean;
}

export function ProgressBar({
  value,
  max = 100,
  label,
  accent = "cyan",
  showValue = true,
}: ProgressBarProps) {
  const percent = (value / max) * 100;
  return (
    <div className="space-y-1">
      {(label || showValue) && (
        <div className="flex items-center justify-between text-xs text-muted">
          {label && <span>{label}</span>}
          {showValue && <span>{value.toFixed(1)}%</span>}
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            accentStyles[accent],
          )}
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
    </div>
  );
}

interface ComparisonBarProps {
  leftValue: number;
  rightValue: number;
  leftLabel: string;
  rightLabel: string;
  leftAccent?: Accent;
  rightAccent?: Accent;
}

export function ComparisonBar({
  leftValue,
  rightValue,
  leftLabel,
  rightLabel,
  leftAccent = "cyan",
  rightAccent = "red",
}: ComparisonBarProps) {
  const total = leftValue + rightValue;
  const leftPercent = total > 0 ? (leftValue / total) * 100 : 50;
  const rightPercent = total > 0 ? (rightValue / total) * 100 : 50;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted">
        <span>
          {leftLabel}:{" "}
          <span className="text-white">{leftValue.toFixed(1)}%</span>
        </span>
        <span>
          {rightLabel}:{" "}
          <span className="text-white">{rightValue.toFixed(1)}%</span>
        </span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
        <div
          className={cn("h-full", accentStyles[leftAccent])}
          style={{ width: `${leftPercent}%` }}
        />
        <div
          className={cn("h-full", accentStyles[rightAccent])}
          style={{ width: `${rightPercent}%` }}
        />
      </div>
    </div>
  );
}
