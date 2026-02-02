"use client";

import { cn } from "@/lib/utils";

export type TabId =
  | "overview"
  | "insights"
  | "economy"
  | "combat"
  | "maps"
  | "agents"
  | "players"
  | "counters";

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

export const TABS: Tab[] = [
  { id: "overview", label: "Overview", icon: "📊" },
  { id: "insights", label: "Insights", icon: "🏆" },
  { id: "economy", label: "Economy", icon: "💰" },
  { id: "combat", label: "Combat", icon: "⚔️" },
  { id: "maps", label: "Maps", icon: "🗺️" },
  { id: "agents", label: "Agents", icon: "👤" },
  { id: "players", label: "Players", icon: "🎮" },
  { id: "counters", label: "Counters", icon: "🎯" },
];

interface TabNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="flex flex-wrap gap-2 border-b border-ethereal pb-4">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onTabChange(tab.id)}
          className={cn(
            "cut-corner flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-[0.2em] transition-all relative overflow-hidden",
            activeTab === tab.id
              ? "border border-cyan bg-gradient-to-r from-cyan/30 to-violet/20 text-white shadow-[0_0_18px_rgba(6,182,212,0.25)]"
              : "border border-ethereal bg-glass text-muted hover:border-cyan/50 hover:text-white hover:bg-[rgba(255,255,255,0.03)]",
          )}
        >
          <span>{tab.icon}</span>
          <span>{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
