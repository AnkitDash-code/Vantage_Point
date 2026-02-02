from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import requests

logger = logging.getLogger(__name__)

try:
    from langchain.document_loaders import WebBaseLoader
except Exception:
    WebBaseLoader = None


class RAGEngine:
    """Retrieval-Augmented Generation for scouting insights."""

    def __init__(
        self,
        knowledge_base_dir: Optional[Path] = None,
        use_web: bool = False,
        urls: Optional[List[str]] = None,
    ) -> None:
        self.knowledge_base_dir = knowledge_base_dir or (
            Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
        )
        self.use_web = use_web
        self.urls = urls or []
        self.jina_api_key = os.getenv("JINA_API_KEY", "").strip()
        self.jina_url = os.getenv("JINA_EMBEDDING_URL", "https://api.jina.ai/v1/embeddings").strip()
        self.jina_model = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3").strip()
        self.jina_late_chunking = (
            os.getenv("JINA_LATE_CHUNKING", "true").strip().lower() == "true"
        )
        self.jina_batch_size = self._int_env("JINA_BATCH_SIZE", 64)
        self.jina_max_input_chars = self._int_env("JINA_MAX_INPUT_CHARS", 1200)
        self.semantic_max_chars = self._int_env("JINA_SEMANTIC_MAX_CHARS", 800)
        self.semantic_min_chars = self._int_env("JINA_SEMANTIC_MIN_CHARS", 150)
        self.semantic_threshold = self._float_env("JINA_SEMANTIC_THRESHOLD", 0.75)
        if self.semantic_max_chars > self.jina_max_input_chars:
            self.semantic_max_chars = self.jina_max_input_chars
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY is required for embedding requests.")
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[str] = []
        # LLM insights cache
        self._insights_cache_dir = self.knowledge_base_dir / ".insights_cache"
        self._insights_cache_dir.mkdir(parents=True, exist_ok=True)
        self._insights_memory_cache: Dict[str, Dict[str, str]] = {}
        self._build_knowledge_base()

    def _build_knowledge_base(self) -> None:
        cache_dir = self.knowledge_base_dir / ".rag_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"
        embeddings_path = cache_dir / "embeddings.npy"
        index_path = cache_dir / "faiss.index"
        docs_manifest_path = self.knowledge_base_dir / "knowledge_base_docs.md"
        excluded_names = {docs_manifest_path.name}

        file_entries = []
        for pattern in ("*.txt", "*.md"):
            for path in self.knowledge_base_dir.glob(pattern):
                if path.name in excluded_names:
                    continue
                content = path.read_text(encoding="utf-8")
                if not self._is_valorant_doc(path, content):
                    continue
                rel = path.relative_to(self.knowledge_base_dir).as_posix()
                file_entries.append((rel, path, content))
        file_entries.sort(key=lambda entry: entry[0])

        cached = self._load_cache(manifest_path, embeddings_path)
        cached_files = {}
        cached_chunks: List[str] = []
        cached_sources: List[str] = []
        cached_embeddings = None
        if cached:
            cached_manifest, cached_embeddings = cached
            cached_files = cached_manifest.get("files", {})
            cached_chunks = cached_manifest.get("chunks", [])
            cached_sources = cached_manifest.get("chunk_sources", [])

        changes_detected = False
        current_files = {}
        for rel, path, content in file_entries:
            file_hash = self._hash_text(content)
            size = path.stat().st_size
            mtime = path.stat().st_mtime
            current_files[rel] = {"hash": file_hash, "size": size, "mtime": mtime}
            cached_entry = cached_files.get(rel)
            if not cached_entry or cached_entry.get("hash") != file_hash:
                changes_detected = True

        if cached_files and set(cached_files.keys()) != set(current_files.keys()):
            changes_detected = True

        if cached and not changes_detected and index_path.exists() and not self._should_embed_web():
            self.documents = cached_chunks
            self.index = faiss.read_index(str(index_path))
            self._write_docs_manifest(docs_manifest_path, current_files, cached_files)
            return

        merged_chunks: List[str] = []
        merged_sources: List[str] = []
        merged_embeddings: List[np.ndarray] = []
        new_files: Dict[str, Dict[str, object]] = {}

        for rel, path, content in file_entries:
            file_hash = self._hash_text(content)
            size = path.stat().st_size
            mtime = path.stat().st_mtime
            cached_entry = cached_files.get(rel)
            reuse_cached = (
                cached_entry
                and cached_entry.get("hash") == file_hash
                and cached_embeddings is not None
                and cached_chunks
                and cached_sources
            )
            if reuse_cached:
                start, end = cached_entry.get("chunk_range", [0, 0])
                start = max(0, int(start))
                end = max(start, int(end))
                merged_chunks.extend(cached_chunks[start:end])
                merged_sources.extend(cached_sources[start:end])
                merged_embeddings.append(cached_embeddings[start:end])
                new_files[rel] = {
                    "hash": file_hash,
                    "size": size,
                    "mtime": mtime,
                    "chunk_range": [len(merged_chunks) - (end - start), len(merged_chunks)],
                }
                continue

            file_chunks = [chunk for chunk in self._semantic_split(content) if chunk.strip()]
            if not file_chunks:
                continue
            file_embeddings = self._embed_texts(file_chunks, task="retrieval.passage")
            if file_embeddings.size == 0:
                continue
            start_index = len(merged_chunks)
            merged_chunks.extend(file_chunks)
            merged_sources.extend([rel] * len(file_chunks))
            merged_embeddings.append(file_embeddings)
            new_files[rel] = {
                "hash": file_hash,
                "size": size,
                "mtime": mtime,
                "chunk_range": [start_index, len(merged_chunks)],
            }

        if self._should_embed_web():
            loader = WebBaseLoader(self.urls)
            web_docs = loader.load()
            for idx, doc in enumerate(web_docs):
                web_chunks = [chunk for chunk in self._semantic_split(doc.page_content) if chunk.strip()]
                if not web_chunks:
                    continue
                web_embeddings = self._embed_texts(web_chunks, task="retrieval.passage")
                if web_embeddings.size == 0:
                    continue
                merged_chunks.extend(web_chunks)
                merged_sources.extend([f"web:{idx}"] * len(web_chunks))
                merged_embeddings.append(web_embeddings)

        if not merged_chunks or not merged_embeddings:
            return

        embeddings = np.vstack(merged_embeddings).astype("float32")
        self.documents = merged_chunks
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        np.save(str(embeddings_path), embeddings)
        manifest_payload = {
            "version": 1,
            "files": new_files,
            "chunks": merged_chunks,
            "chunk_sources": merged_sources,
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
        faiss.write_index(self.index, str(index_path))
        self._write_docs_manifest(docs_manifest_path, current_files, new_files)

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        if not self.index or not self.documents:
            return []

        query_embedding = self._embed_texts([query], task="retrieval.query")
        if query_embedding.size == 0:
            return []
        _, indices = self.index.search(query_embedding, k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]

    def _semantic_split(self, text: str) -> List[str]:
        candidates = self._sentence_candidates(text)
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        embeddings = self._embed_texts(candidates, task="retrieval.passage")
        if embeddings.size == 0:
            return candidates
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        prev_vector = None
        for idx, sentence in enumerate(candidates):
            vector = embeddings[idx]
            if not current:
                current = [sentence]
                current_len = len(sentence)
                prev_vector = vector
                continue
            similarity = self._cosine_similarity(prev_vector, vector)
            would_exceed = current_len + len(sentence) > self.semantic_max_chars
            should_split = similarity < self.semantic_threshold and current_len >= self.semantic_min_chars
            if would_exceed or should_split:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1
            prev_vector = vector
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _sentence_candidates(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sentences: List[str] = []
        for line in lines:
            if line.startswith("-") or line.endswith(":"):
                sentences.append(line)
                continue
            parts = re.split(r"(?<=[.!?])\s+", line)
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    sentences.append(cleaned)
        return sentences

    def _should_embed_web(self) -> bool:
        return bool(self.use_web and self.urls and WebBaseLoader)

    def _is_valorant_doc(self, path: Path, content: str) -> bool:
        name = path.name.lower()
        if "valorant" in name:
            return True
        return "valorant" in content.lower()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self, manifest_path: Path, embeddings_path: Path) -> Optional[tuple[Dict, np.ndarray]]:
        if not manifest_path.exists() or not embeddings_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        try:
            embeddings = np.load(str(embeddings_path))
        except Exception:
            return None
        if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
            return None
        return manifest, embeddings

    def _write_docs_manifest(
        self,
        docs_manifest_path: Path,
        current_files: Dict[str, Dict[str, object]],
        cached_files: Dict[str, Dict[str, object]],
    ) -> None:
        lines = ["Knowledge Base Documents", ""]
        for rel in sorted(current_files.keys()):
            entry = cached_files.get(rel) or current_files.get(rel) or {}
            file_hash = entry.get("hash", "")
            chunk_range = entry.get("chunk_range", [0, 0])
            chunk_count = 0
            if isinstance(chunk_range, list) and len(chunk_range) == 2:
                chunk_count = max(0, int(chunk_range[1]) - int(chunk_range[0]))
            lines.append(f"- {rel} | sha256: {file_hash} | chunks: {chunk_count}")
        docs_manifest_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _truncate_text(self, text: str) -> str:
        if not text:
            return ""
        limit = max(1, self.jina_max_input_chars)
        return text.strip()[:limit]

    def _embed_texts(self, texts: List[str], task: str) -> np.ndarray:
        if not texts:
            return np.array([], dtype="float32")
        trimmed = [self._truncate_text(text) for text in texts]
        trimmed = [text for text in trimmed if text]
        if not trimmed:
            return np.array([], dtype="float32")
        batches = []
        for start in range(0, len(trimmed), self.jina_batch_size):
            batch = trimmed[start : start + self.jina_batch_size]
            batches.append(self._jina_request(batch, task))
        return np.vstack(batches).astype("float32") if batches else np.array([], dtype="float32")

    def _jina_request(self, inputs: List[str], task: str) -> np.ndarray:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}",
        }
        payload = {
            "model": self.jina_model,
            "task": task,
            "late_chunking": self.jina_late_chunking,
            "input": inputs,
        }
        response = requests.post(self.jina_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        embeddings = [item.get("embedding") for item in data.get("data", [])]
        if len(embeddings) != len(inputs):
            raise ValueError("Embedding response size mismatch.")
        return np.array(embeddings, dtype="float32")

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return value

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value

    def _build_moneyball_context(self, metrics: Dict, team_name: str) -> Dict[str, str]:
        """Build structured statistical context from metrics for Moneyball analysis."""
        context = {}
        
        # --- MACRO STATS ---
        site_pref = metrics.get("site_preferences", {})
        site_a = site_pref.get("A", site_pref.get("a", 0))
        site_b = site_pref.get("B", site_pref.get("b", 0))
        site_c = site_pref.get("C", site_pref.get("c", 0))
        context["site_bias"] = f"A-Site: {site_a:.1f}%, B-Site: {site_b:.1f}%" + (f", C-Site: {site_c:.1f}%" if site_c > 0 else "")
        
        # Side metrics
        side = metrics.get("side_metrics", {})
        context["attack_win_rate"] = f"{side.get('attack_win_rate', 0):.1f}%"
        context["defense_win_rate"] = f"{side.get('defense_win_rate', 0):.1f}%"
        context["attack_rounds"] = side.get("attack_rounds", 0)
        context["defense_rounds"] = side.get("defense_rounds", 0)
        
        # First duel stats
        first_duel = metrics.get("first_duel", {})
        context["first_kill_rate"] = f"{first_duel.get('team_first_kill_rate', 0):.1f}%"
        context["first_kill_conversion"] = f"{first_duel.get('first_kill_conversion_rate', 0):.1f}%"
        
        # Combat metrics
        combat = metrics.get("combat_metrics", {})
        context["trade_efficiency"] = f"{combat.get('trade_efficiency', 0):.1f}%"
        context["trade_kills"] = combat.get("trade_kills", 0)
        context["trade_opportunities"] = combat.get("trade_opportunities", 0)
        
        # Round type performance
        round_perf = metrics.get("round_type_performance", {})
        pistol = round_perf.get("pistol", {})
        eco = round_perf.get("eco", {})
        full_buy = round_perf.get("full_buy", {})
        context["pistol_win_rate"] = f"{pistol.get('win_rate', 0):.1f}%"
        context["eco_win_rate"] = f"{eco.get('win_rate', 0):.1f}%"
        context["full_buy_win_rate"] = f"{full_buy.get('win_rate', 0):.1f}%"
        
        # --- PLAYER STATS ---
        opening_duels = combat.get("opening_duels", [])
        if opening_duels:
            opener = max(opening_duels, key=lambda x: x.get("opening_wins", 0) + x.get("opening_losses", 0))
            total_duels = opener.get("opening_wins", 0) + opener.get("opening_losses", 0)
            win_rate = (opener.get("opening_wins", 0) / total_duels * 100) if total_duels > 0 else 0
            context["primary_opener"] = f"{opener.get('player', 'Unknown')} ({win_rate:.1f}% win rate, {total_duels} duels)"
        else:
            context["primary_opener"] = "No data"
        
        clutch_performers = combat.get("clutch_performers", [])
        if clutch_performers:
            best_clutcher = max(clutch_performers, key=lambda x: x.get("clutches_won", 0))
            context["best_clutcher"] = f"{best_clutcher.get('player', 'Unknown')} ({best_clutcher.get('clutches_won', 0)}/{best_clutcher.get('clutches_faced', 0)} clutches, {best_clutcher.get('clutch_rate', 0):.1f}%)"
        else:
            context["best_clutcher"] = "No data"
        
        multi_killers = combat.get("multi_killers", [])
        if multi_killers:
            top_multi = max(multi_killers, key=lambda x: x.get("total", 0))
            context["top_multi_killer"] = f"{top_multi.get('player', 'Unknown')} (2K:{top_multi.get('2k', 0)}, 3K:{top_multi.get('3k', 0)}, 4K:{top_multi.get('4k', 0)}, ACE:{top_multi.get('5k', 0)})"
        else:
            context["top_multi_killer"] = "No data"
        
        # Player tendencies for weak link analysis
        players = metrics.get("player_tendencies", [])
        if players:
            # Find player with lowest KD as potential weak link
            players_with_kd = [p for p in players if p.get("kd_ratio") is not None]
            if players_with_kd:
                weak_link = min(players_with_kd, key=lambda x: x.get("kd_ratio", 999))
                context["weak_link"] = f"{weak_link.get('player', 'Unknown')} (KD: {weak_link.get('kd_ratio', 0):.2f})"
            else:
                context["weak_link"] = "No KD data"
        else:
            context["weak_link"] = "No player data"
        
        # Weapon effectiveness
        weapons = combat.get("weapon_effectiveness", [])
        if weapons:
            top_weapons = sorted(weapons, key=lambda x: x.get("kills", 0), reverse=True)[:3]
            weapon_lines = [f"{w.get('weapon', '?').title()}: {w.get('kills', 0)} kills, {w.get('kd_ratio', 0):.2f} KD" for w in top_weapons]
            context["weapon_profile"] = "; ".join(weapon_lines)
        else:
            context["weapon_profile"] = "No weapon data"
        
        # --- COMPOSITION STATS ---
        agents = metrics.get("agent_composition", [])
        context["agent_core"] = ", ".join([f"{a.get('agent', '?')} ({a.get('pick_rate', 0):.1f}%)" for a in agents[:5]])
        
        role_dist = metrics.get("role_distribution", {})
        context["role_balance"] = ", ".join([f"{role}: {pct:.1f}%" for role, pct in role_dist.items()])
        
        # Map stats
        map_detailed = metrics.get("map_detailed", {})
        map_lines = []
        for map_name, stats in (map_detailed or {}).items():
            if isinstance(stats, dict):
                wr = stats.get("win_rate", 0)
                atk = stats.get("attack_win_rate", 0)
                def_ = stats.get("defense_win_rate", 0)
                map_lines.append(f"{map_name}: {wr:.1f}% WR (ATK:{atk:.1f}%, DEF:{def_:.1f}%)")
        context["map_breakdown"] = "; ".join(map_lines[:5]) if map_lines else "No map data"
        
        # Overall win rate
        context["overall_win_rate"] = f"{metrics.get('win_rate', 0):.1f}%"
        
        return context

    def _build_insight_prompts(self, metrics: Dict, team_name: str) -> Dict[str, str]:
        """Build Moneyball-style prompts for statistical scouting analysis."""
        ctx = self._build_moneyball_context(metrics, team_name)
        
        # Global format rules for all prompts
        format_rules = """
FORMAT RULES:
- Use markdown tables for player data and comparisons (| Column | Column |)
- Use bullet points for strategies and protocols
- Keep responses concise and actionable
- Use bold (**text**) for emphasis
"""
        
        # Retrieve counter-strategy context
        agents = metrics.get("agent_composition", [])
        agent_names = [a.get("agent") for a in agents[:2] if a.get("agent")]
        weakness_query = f"Counter strategies for {', '.join(agent_names)}" if agent_names else "VALORANT counter strategies"
        counter_context = self.retrieve_context(weakness_query, k=3)
        counter_text = " ".join(counter_context[:2]) if counter_context else ""

        # STRATEGIES PROMPT - Deep macro analysis with specific patterns
        strategy_prompt = f"""You are a VCT Data Analyst generating an exhaustive scouting report.
{format_rules}
TEAM: {team_name}
CONSTRAINT: Every claim MUST use "X% of rounds" format with specific numbers.

=== COMPUTED METRICS ===
Overall Win Rate: {ctx['overall_win_rate']}
Attack Win Rate: {ctx['attack_win_rate']} ({ctx['attack_rounds']} rounds)
Defense Win Rate: {ctx['defense_win_rate']} ({ctx['defense_rounds']} rounds)
Site Bias: {ctx['site_bias']}
First Blood Rate: {ctx['first_kill_rate']} | Conversion: {ctx['first_kill_conversion']}
Trade Efficiency: {ctx['trade_efficiency']} ({ctx['trade_kills']}/{ctx['trade_opportunities']})
Pistol: {ctx['pistol_win_rate']} | Eco: {ctx['eco_win_rate']} | Full Buy: {ctx['full_buy_win_rate']}
Maps: {ctx['map_breakdown']}

OUTPUT (use markdown, be SPECIFIC with percentages):

## Attack Patterns
- **Site Distribution**: "X% of attacks target A-Site, Y% target B-Site"
- **Pistol Executes**: Based on {ctx['pistol_win_rate']} pistol rate, describe their Round 1 attack (e.g., "70% 5-man fast B")
- **Default Setup**: Describe their likely 4-1 or 3-2 spread before execute
- **Execute Timing**: Based on first blood {ctx['first_kill_rate']}, classify tempo (Fast <30s / Default 30-60s / Slow >60s)

## Defense Patterns
- **Setup Style**: Based on {ctx['defense_win_rate']}, do they anchor or retake?
- **Default Formation**: "1-3-1 with Sentinel mid" or "2-1-2 site anchor" style
- **Rotation Tendency**: Based on site bias, over-rotate or hold positions?
- **Aggression Level**: Any early push tendencies?

## Economic Behavior
- **Pistol Priority**: What does {ctx['pistol_win_rate']} suggest about their pistol strat?
- **Eco Rounds**: {ctx['eco_win_rate']} indicates rush or spread save?
- **Force Buy Discipline**: Based on full buy {ctx['full_buy_win_rate']}, do they force or save?
- **Bonus Rounds**: Post-pistol win behavior

## Map-Specific Tendencies
For each map in {ctx['map_breakdown']}:
- Site preference on that map
- Unique strategies observed"""

        # TENDENCIES PROMPT - Granular player analysis
        tendencies_prompt = f"""You are a VCT Data Analyst. Generate HIGHLY SPECIFIC player intelligence.
{format_rules}
TEAM: {team_name}
CONSTRAINT: Use "Player X has Y% rate" format. Be specific.

=== PLAYER METRICS ===
Primary Opener: {ctx['primary_opener']}
Top Multi-Killer: {ctx['top_multi_killer']}
Best Clutcher: {ctx['best_clutcher']}
Weak Link: {ctx['weak_link']}
Weapon Profile: {ctx['weapon_profile']}

=== COMBAT STATS ===
Trade Efficiency: {ctx['trade_efficiency']}
First Kill Rate: {ctx['first_kill_rate']}
First Kill Conversion: {ctx['first_kill_conversion']}

OUTPUT (markdown with player names and specific stats):

## Player Scouting Cards

### 🎯 Entry Fragger (Primary Threat)
**{ctx['primary_opener'].split('(')[0].strip() if '(' in ctx['primary_opener'] else 'Primary Opener'}**
- **First Duel Stats**: {ctx['primary_opener']}
- **Threat Level**: CRITICAL - Neutralize to drop their win probability
- **Typical Positions**: [Infer from agent - e.g., "Jett: A-Short aggro peek"]
- **Counter Protocol**: Hold off-angles, use utility to deny entry

### 🔻 Weak Link (Isolation Target)
**{ctx['weak_link'].split('(')[0].strip() if '(' in ctx['weak_link'] else 'Weak Link'}**
- **Stats**: {ctx['weak_link']}
- **Exploitation**: Hunt this player on defense rotations
- **Attack Protocol**: Force isolated 1v1 duels

### 🏆 Clutch King (Avoid 1vX)
**{ctx['best_clutcher'].split('(')[0].strip() if '(' in ctx['best_clutcher'] else 'Best Clutcher'}**
- **Clutch Stats**: {ctx['best_clutcher']}
- **Danger**: HIGH in post-plant scenarios
- **Protocol**: ALWAYS double-peek, never give 1v1 clutch opportunities

### 💥 Multi-Kill Threat
**{ctx['top_multi_killer'].split('(')[0].strip() if '(' in ctx['top_multi_killer'] else 'Top Multi-Killer'}**
- **Stats**: {ctx['top_multi_killer']}
- **Pattern**: Chains kills in tight spaces
- **Counter**: Spread out, avoid stacking

## Team-Wide Patterns
- **Trade Discipline**: {ctx['trade_efficiency']} - {"Strong trading, avoid isolated peeks" if float(ctx['trade_efficiency'].rstrip('%')) >= 40 else "They die isolated - take aggressive 1v1s"}
- **Opening Duel**: {ctx['first_kill_rate']} first blood rate
- **Conversion**: {ctx['first_kill_conversion']} - {"Capitalize on 5v4s" if float(ctx['first_kill_conversion'].rstrip('%')) >= 70 else "They throw advantages - stay patient"}

## Weapon Tendencies
{ctx['weapon_profile']}
- Identify primary weapon users and counter-buy accordingly"""

        # COMPOSITIONS PROMPT - Detailed setup analysis
        compositions_prompt = f"""You are a VCT Data Analyst. Provide SPECIFIC composition breakdowns.
{format_rules}
TEAM: {team_name}
CONSTRAINT: Reference pick rates. Describe actual setups.

=== COMPOSITION DATA ===
Agent Core: {ctx['agent_core']}
Role Distribution: {ctx['role_balance']}

=== TACTICAL DATA ===
Site Preferences: {ctx['site_bias']}
Attack Win Rate: {ctx['attack_win_rate']}
Defense Win Rate: {ctx['defense_win_rate']}
Map Breakdown: {ctx['map_breakdown']}

OUTPUT (markdown with specific compositions):

## Most-Played Compositions
Based on agent pick rates, list their likely comps:

### Primary Comp (Most Frequent)
**Agents**: [List 5 agents based on highest pick rates]
**Pick Rate**: ~X% of matches
**Best Map**: [Infer from map data]
**Style**: [Describe playstyle this comp enables]

### Secondary Comp (Alternate)
**Agents**: [List alternate comp]
**When Used**: [Map/situation specific]

## Default Setups

### Defense Formation
Based on {ctx['role_balance']}:
- **A-Site**: [Agent] anchors, [Agent] plays retake
- **B-Site**: [Agent] holds, [Agent] rotates
- **Mid Control**: [Agent] contests mid in X% of rounds

### Attack Formation
- **Default Spread**: Describe 4-1 or 3-1-1 spread
- **Entry Order**: [Duelist] → [Initiator] → [Support]
- **Lurk Tendency**: Does their comp suggest a lurker?

## Utility Sequences

### A-Site Execute
1. [Initiator] opens with [ability]
2. [Controller] smokes [locations]
3. [Duelist] entries

### B-Site Execute
1. [Ability sequence for B]

## Agent Synergies
- **Flash + Entry Combo**: [Skye + Jett combo description]
- **Smoke + Molly Combo**: [Controller + Sentinel combo]
- **Info Gathering**: [Initiator recon patterns]

## Role Balance Analysis
{ctx['role_balance']}
- **Classification**: {"Duelist-heavy (aggressive)" if 'duelist' in ctx['role_balance'].lower() and '30%' in ctx['role_balance'] else "Balanced" if 'controller' in ctx['role_balance'].lower() else "Utility-heavy (methodical)"}
- **Implication**: What this enables strategically"""

        # HOW TO WIN PROMPT - Comprehensive game plan
        how_to_win_prompt = f"""You are a VCT Head Coach creating the MASTER GAME PLAN.
{format_rules}
OPPONENT: {team_name}
Generate a COMPLETE strategic briefing with IF/THEN protocols.

=== SCOUTING INTELLIGENCE ===
Win Rate: {ctx['overall_win_rate']} | Attack: {ctx['attack_win_rate']} | Defense: {ctx['defense_win_rate']}
Site Bias: {ctx['site_bias']}
First Blood: {ctx['first_kill_rate']} rate, {ctx['first_kill_conversion']} conversion
Trade Efficiency: {ctx['trade_efficiency']}
Pistol: {ctx['pistol_win_rate']} | Eco: {ctx['eco_win_rate']} | Full Buy: {ctx['full_buy_win_rate']}
Maps: {ctx['map_breakdown']}

Key Players:
- Primary Threat: {ctx['primary_opener']}
- Weak Link: {ctx['weak_link']}
- Clutch Danger: {ctx['best_clutcher']}

=== COUNTER KNOWLEDGE ===
{counter_text}

OUTPUT (comprehensive markdown game plan):

## 🎯 Win Condition Summary
**Primary Win Condition**: [1 sentence - what we must do to win]
**Loss Condition**: [1 sentence - what to avoid]

## 📋 Round-by-Round Protocol

### Pistol Round (Round 1)
Their pistol win rate: {ctx['pistol_win_rate']}
- **OUR ATTACK**: [Specific pistol strat based on their defense]
- **OUR DEFENSE**: [Setup to counter their pistol aggression]
- **Target**: Focus [player] first

### Anti-Eco (Rounds 2-3)
Their eco win rate: {ctx['eco_win_rate']}
- **IF WE WIN PISTOL**: {"Play long range, avoid close angles" if float(ctx['eco_win_rate'].rstrip('%')) > 20 else "Push aggressively, they won't convert"}
- **IF WE LOSE PISTOL**: [Save or force based on their bonus conversion]

### Gun Rounds (4+)
- **OUR ATTACK FOCUS**: {"B-Site" if 'A' in ctx['site_bias'] and float(ctx['site_bias'].split('A-Site:')[1].split('%')[0].strip()) > 55 else "A-Site"} - they stack the other
- **OUR DEFENSE SETUP**: Stack their preferred attack site

## 🔄 Critical IF/THEN Decision Tree

### Opening Duel Outcomes
- **IF** we get first blood → {{"Play aggressive 5v4" if float(ctx['first_kill_conversion'].rstrip('%')) < 70 else "Play methodical, they convert well"}}
- **IF** we lose first blood → {"Trade immediately" if float(ctx['trade_efficiency'].rstrip('%')) >= 40 else "Reset and regroup"}

### Man-Advantage Scenarios  
- **5v4**: {"Disciplined execute - they trade" if float(ctx['trade_efficiency'].rstrip('%')) >= 40 else "Push fast - they die isolated"}
- **4v5**: {ctx['first_kill_conversion']} conversion means {"Save economy" if float(ctx['first_kill_conversion'].rstrip('%')) >= 70 else "Try retake - they throw leads"}

## 👤 Player Assignments

### HUNT PROTOCOL
**Target**: {ctx['primary_opener'].split('(')[0].strip() if '(' in ctx['primary_opener'] else 'Primary Opener'}
- Assign your best aimer to contest
- Utility to deny their entry angle
- Expected position: [Infer from agent/role]

### EXPLOIT PROTOCOL  
**Target**: {ctx['weak_link'].split('(')[0].strip() if '(' in ctx['weak_link'] else 'Weak Link'}
- Force isolated 1v1 duels
- Hunt on rotations

### AVOID PROTOCOL
**Target**: {ctx['best_clutcher'].split('(')[0].strip() if '(' in ctx['best_clutcher'] else 'Clutch Player'}
- NEVER give 1v1 clutch
- Always double-peek post-plant

## ⏱️ Timing Protocols

### First 30 Seconds
- {"Expect early aggression" if float(ctx['first_kill_rate'].rstrip('%')) > 50 else "They play default"}
- Our response: [Counter to their opening]

### Mid-Round (30-60s)
- Watch for their execute timing
- Rotation trigger points

### Late Round (60s+)  
- {"They rush plant" if float(ctx['attack_win_rate'].rstrip('%')) > 55 else "They play for picks"}
- Post-plant protocol: [Based on clutch data]

## 💰 Economy Warfare

### When THEY'RE on Eco
{ctx['eco_win_rate']} eco rate means:
- {"RESPECT - play long range, no hero plays" if float(ctx['eco_win_rate'].rstrip('%')) > 20 else "PUNISH - push aggressively"}

### When WE'RE Outgunned
Their full buy win rate: {ctx['full_buy_win_rate']}
- {"Play for hero plays - they convert" if float(ctx['full_buy_win_rate'].rstrip('%')) > 55 else "Play disciplined - they're beatable"}

## 🗺️ Map-Specific Notes
{ctx['map_breakdown']}
- Prioritize their weakest map
- Avoid their strongest map in veto"""

        # COUNTERS PROMPT - Dedicated counter-strategy analysis
        counters_prompt = f"""You are a VCT Analyst specializing in counter-strategies. Generate SPECIFIC counters.
{format_rules}
OPPONENT: {team_name}

=== AGENT DATA ===
Agent Core: {ctx['agent_core']}
Role Distribution: {ctx['role_balance']}

=== VULNERABILITY DATA ===
Site Bias: {ctx['site_bias']}
Primary Opener: {ctx['primary_opener']}
Trade Efficiency: {ctx['trade_efficiency']}
Eco Win Rate: {ctx['eco_win_rate']}

=== COUNTER KNOWLEDGE ===
{counter_text}

OUTPUT FORMAT (markdown with specific agent counters):

## Agent-Specific Counters

For each agent in their core, provide:
- **Counter Agent**: Which agent hard-counters them
- **Utility Counter**: Specific ability to deny their value  
- **Positional Denial**: Where to play to neutralize

### [Agent 1] Counter Protocol
[Specific counter strategies]

### [Agent 2] Counter Protocol
[Specific counter strategies]

## Map Control Denial
- How to deny their preferred site control
- Timing windows to exploit

## Economic Warfare
- Anti-eco protocol based on {ctx['eco_win_rate']} eco win rate
- Bonus round denial strategies

## Setup Breakers
- How to break their default defense
- Utility to counter their site holds"""

        return {
            "strategies": strategy_prompt,
            "tendencies": tendencies_prompt,
            "compositions": compositions_prompt,
            "counters": counters_prompt,
            "how_to_win": how_to_win_prompt,
        }

    def _compute_insights_cache_key(self, metrics: Dict, team_name: str) -> str:
        """Compute a stable cache key from metrics and team name."""
        # Extract key metrics that affect insights
        key_data = {
            "team": team_name.strip().lower(),
            "win_rate": metrics.get("win_rate"),
            "site_preferences": metrics.get("site_preferences"),
            "first_duel": metrics.get("first_duel"),
            "side_metrics": metrics.get("side_metrics"),
            "combat_metrics": metrics.get("combat_metrics"),
            "round_type_performance": metrics.get("round_type_performance"),
            "player_tendencies": [
                {"player": p.get("player"), "kd_ratio": p.get("kd_ratio")}
                for p in (metrics.get("player_tendencies") or [])[:5]
            ],
            "agent_composition": [
                {"agent": a.get("agent"), "pick_rate": a.get("pick_rate")}
                for a in (metrics.get("agent_composition") or [])[:5]
            ],
        }
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _load_insights_cache(self, cache_key: str) -> Optional[Dict[str, str]]:
        """Load cached insights from memory or disk."""
        # Check memory cache first
        if cache_key in self._insights_memory_cache:
            return self._insights_memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self._insights_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Store in memory for faster access
                    self._insights_memory_cache[cache_key] = data
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_insights_cache(self, cache_key: str, insights: Dict[str, str]) -> None:
        """Save insights to memory and disk cache."""
        self._insights_memory_cache[cache_key] = insights
        cache_file = self._insights_cache_dir / f"{cache_key}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(insights, f, ensure_ascii=False)
        except IOError:
            pass

    def generate_insights(self, metrics: Dict, team_name: str) -> Dict[str, str]:
        total_start = time.perf_counter()
        
        # Check cache first
        cache_key = self._compute_insights_cache_key(metrics, team_name)
        cached = self._load_insights_cache(cache_key)
        if cached:
            logger.info(f"[RAG TIMING] insights_cache_hit: {time.perf_counter() - total_start:.2f}s")
            return cached

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            insights = self._fallback_insights(metrics, team_name)
            self._save_insights_cache(cache_key, insights)
            return insights

        t0 = time.perf_counter()
        prompts = self._build_insight_prompts(metrics, team_name)
        logger.info(f"[RAG TIMING] build_prompts: {time.perf_counter() - t0:.2f}s")
        
        try:
            from groq import Groq
        except Exception:
            Groq = None

        if Groq is not None:
            try:
                client = Groq(api_key=groq_key)
                sections: Dict[str, str] = {}
                for section, prompt in prompts.items():
                    t0 = time.perf_counter()
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=False,
                    )
                    content = ""
                    if completion and completion.choices:
                        message = completion.choices[0].message
                        if message and message.content:
                            content = message.content
                    sections[section] = content
                    logger.info(f"[RAG TIMING] llm_call[{section}]: {time.perf_counter() - t0:.2f}s")
                self._save_insights_cache(cache_key, sections)
                logger.info(f"[RAG TIMING] generate_insights_total: {time.perf_counter() - total_start:.2f}s")
                return sections
            except Exception:
                pass

        try:
            from langchain_groq import ChatGroq
        except Exception:
            insights = self._fallback_insights(metrics, team_name)
            self._save_insights_cache(cache_key, insights)
            return insights

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            groq_api_key=groq_key,
            streaming=True,
        )

        sections: Dict[str, str] = {}
        for section, prompt in prompts.items():
            result = llm.invoke(prompt)
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            sections[section] = content

        self._save_insights_cache(cache_key, sections)
        return sections

    def generate_insights_stream(self, metrics: Dict, team_name: str):
        total_start = time.perf_counter()
        
        # Check cache first - if cached, yield all at once (fast path)
        cache_key = self._compute_insights_cache_key(metrics, team_name)
        cached = self._load_insights_cache(cache_key)
        if cached:
            logger.info(f"[RAG STREAM TIMING] insights_cache_hit: {time.perf_counter() - total_start:.2f}s")
            for section, text in cached.items():
                yield section, text
            return

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            fallback = self._fallback_insights(metrics, team_name)
            self._save_insights_cache(cache_key, fallback)
            for section, text in fallback.items():
                yield section, text
            return

        t0 = time.perf_counter()
        prompts = self._build_insight_prompts(metrics, team_name)
        logger.info(f"[RAG STREAM TIMING] build_prompts: {time.perf_counter() - t0:.2f}s")
        
        # Collect all insights for caching
        all_insights: Dict[str, str] = {}
        
        try:
            from groq import Groq
        except Exception:
            Groq = None

        if Groq is not None:
            try:
                client = Groq(api_key=groq_key)
                for section, prompt in prompts.items():
                    t0 = time.perf_counter()
                    streamed = ""
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=True,
                    )
                    for chunk in completion:
                        if not chunk or not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        text = ""
                        if delta and delta.content:
                            text = delta.content
                        if text:
                            streamed += text
                            yield section, text
                    if not streamed:
                        completion = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            stream=False,
                        )
                        content = ""
                        if completion and completion.choices:
                            message = completion.choices[0].message
                            if message and message.content:
                                content = message.content
                        if content:
                            streamed = content
                            yield section, content
                    all_insights[section] = streamed
                    logger.info(f"[RAG STREAM TIMING] llm_stream[{section}]: {time.perf_counter() - t0:.2f}s")
                # Save to cache after streaming completes
                if all_insights:
                    self._save_insights_cache(cache_key, all_insights)
                logger.info(f"[RAG STREAM TIMING] generate_insights_total: {time.perf_counter() - total_start:.2f}s")
                return
            except Exception:
                pass

        try:
            from langchain_groq import ChatGroq
        except Exception:
            fallback = self._fallback_insights(metrics, team_name)
            self._save_insights_cache(cache_key, fallback)
            for section, text in fallback.items():
                yield section, text
            return

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            groq_api_key=groq_key,
            stream=True,
        )

        for section, prompt in prompts.items():
            streamed = ""
            try:
                for chunk in llm.stream(prompt):
                    if isinstance(chunk, str):
                        text = chunk
                    else:
                        text = getattr(chunk, "content", "") or ""
                    if text:
                        streamed += text
                        yield section, text
            except Exception:
                if not streamed:
                    result = llm.invoke(prompt)
                    content = getattr(result, "content", None)
                    if content is None:
                        content = str(result)
                    if content:
                        streamed = content
                        yield section, content
            all_insights[section] = streamed
        
        # Save to cache after streaming completes
        if all_insights:
            self._save_insights_cache(cache_key, all_insights)

    @staticmethod
    def _event_summary(metrics: Dict) -> str:
        first_duel = metrics.get("first_duel") or {}
        pistol_rounds = metrics.get("pistol_rounds") or {}
        parts: List[str] = []

        if first_duel:
            first_kill_rate = first_duel.get("team_first_kill_rate")
            conversion_rate = first_duel.get("first_kill_conversion_rate")
            avg_time = first_duel.get("avg_time_to_first_kill")
            if first_kill_rate is not None:
                parts.append(f"first-kill rate {first_kill_rate}%")
            if conversion_rate is not None:
                parts.append(f"conversion {conversion_rate}%")
            if avg_time is not None:
                parts.append(f"avg first kill {avg_time}s")

            top_players = first_duel.get("top_first_kill_players") or []
            top_labels = []
            for player in top_players[:2]:
                name = player.get("player")
                count = player.get("first_kills")
                if name and count is not None:
                    top_labels.append(f"{name} ({count})")
            if top_labels:
                parts.append(f"top first killers: {', '.join(top_labels)}")

        if pistol_rounds:
            overall = pistol_rounds.get("overall") or {}
            pistol_bits = []
            if overall.get("win_rate") is not None:
                pistol_bits.append(f"pistol win {overall.get('win_rate')}%")
            if overall.get("first_kill_rate") is not None:
                pistol_bits.append(f"pistol first-kill {overall.get('first_kill_rate')}%")
            if overall.get("rounds") is not None:
                pistol_bits.append(f"n={overall.get('rounds')}")
            if pistol_bits:
                parts.append("pistol rounds: " + ", ".join(pistol_bits))

            by_side = pistol_rounds.get("by_side") or {}
            side_bits = []
            for side in ("attacker", "defender"):
                side_info = by_side.get(side) or {}
                if side_info.get("win_rate") is not None:
                    side_bits.append(f"{side} {side_info.get('win_rate')}%")
            if side_bits:
                parts.append("pistol by side: " + ", ".join(side_bits))

        return "; ".join(parts)

    @staticmethod
    def _top_map(win_rates: Dict[str, float]) -> List[str]:
        if not win_rates:
            return ["Unknown", "0"]
        map_name, pct = max(win_rates.items(), key=lambda x: x[1])
        return [str(map_name), str(round(pct, 1))]

    @staticmethod
    def _top_site(site_preferences: Dict[str, float]) -> List[str]:
        if not site_preferences:
            return ["Unknown", "0"]
        site, pct = max(site_preferences.items(), key=lambda x: x[1])
        return [str(site), str(round(pct, 1))]

    @staticmethod
    def _fallback_insights(metrics: Dict, team_name: str) -> Dict[str, str]:
        """Generate comprehensive Moneyball-style fallback insights when LLM is unavailable."""
        site_pref = metrics.get("site_preferences", {})
        site_a = site_pref.get("A", site_pref.get("a", 0))
        site_b = site_pref.get("B", site_pref.get("b", 0))
        
        side = metrics.get("side_metrics", {})
        atk_wr = side.get("attack_win_rate", 0)
        def_wr = side.get("defense_win_rate", 0)
        
        first_duel = metrics.get("first_duel", {})
        fk_rate = first_duel.get("team_first_kill_rate", 0)
        fk_conv = first_duel.get("first_kill_conversion_rate", 0)
        
        combat = metrics.get("combat_metrics", {})
        trade_eff = combat.get("trade_efficiency", 0)
        
        round_perf = metrics.get("round_type_performance", {})
        pistol_wr = round_perf.get("pistol", {}).get("win_rate", 0)
        eco_wr = round_perf.get("eco", {}).get("win_rate", 0)
        full_buy_wr = round_perf.get("full_buy", {}).get("win_rate", 0)
        
        opening_duels = combat.get("opening_duels", [])
        opener_name = "Unknown"
        opener_stat = ""
        if opening_duels:
            opener = max(opening_duels, key=lambda x: x.get("opening_wins", 0) + x.get("opening_losses", 0))
            total = opener.get("opening_wins", 0) + opener.get("opening_losses", 0)
            wr = (opener.get("opening_wins", 0) / total * 100) if total > 0 else 0
            opener_name = opener.get("player", "Unknown")
            opener_stat = f"{wr:.1f}% win rate on {total} duels"
        
        clutch_performers = combat.get("clutch_performers", [])
        clutcher_name = "Unknown"
        clutcher_stat = ""
        if clutch_performers:
            best = max(clutch_performers, key=lambda x: x.get("clutches_won", 0))
            clutcher_name = best.get("player", "Unknown")
            clutcher_stat = f"{best.get('clutches_won', 0)}/{best.get('clutches_faced', 0)} clutches"
        
        players = metrics.get("player_tendencies", [])
        weak_link = "Unknown"
        weak_stat = ""
        if players:
            players_kd = [p for p in players if p.get("kd_ratio") is not None]
            if players_kd:
                wl = min(players_kd, key=lambda x: x.get("kd_ratio", 999))
                weak_link = wl.get("player", "Unknown")
                weak_stat = f"KD: {wl.get('kd_ratio', 0):.2f}"
        
        agents = metrics.get("agent_composition", [])
        agent_core = ', '.join([f"{a.get('agent', '?')} ({a.get('pick_rate', 0):.1f}%)" for a in agents[:5]])
        role_dist = metrics.get("role_distribution", {})
        role_summary = ', '.join([f"{r}: {p:.1f}%" for r, p in role_dist.items()])

        return {
            "strategies": f"""## Attack Patterns
- **Site Distribution**: A-Site {site_a:.1f}%, B-Site {site_b:.1f}%
- **Tempo**: {"Fast (high first blood rate)" if fk_rate > 50 else "Default/Methodical"} - {fk_rate:.1f}% first blood rate
- **Post-FB Conversion**: {fk_conv:.1f}% - {"Strong closers" if fk_conv >= 70 else "Throw leads often"}

## Defense Patterns
- **Side Strength**: Attack {atk_wr:.1f}% vs Defense {def_wr:.1f}%
- **Style**: {"Anchor-heavy" if def_wr > atk_wr else "Retake-focused"}
- **Trade Discipline**: {trade_eff:.1f}%

## Economic Behavior
- **Pistol Win Rate**: {pistol_wr:.1f}%
- **Eco Win Rate**: {eco_wr:.1f}% (league avg ~15%)
- **Full Buy Win Rate**: {full_buy_wr:.1f}%

## Overall
- **Win Rate**: {metrics.get('win_rate', 0):.1f}%""",
            
            "tendencies": f"""## Player Scouting Cards

### 🎯 Entry Fragger (Primary Threat)
**{opener_name}**
- **Stats**: {opener_stat}
- **Threat Level**: CRITICAL
- **Counter**: Hold off-angles, use utility to deny entry

### 🔻 Weak Link (Isolation Target)
**{weak_link}**
- **Stats**: {weak_stat}
- **Exploitation**: Hunt on rotations, force isolated 1v1s

### 🏆 Clutch King (Avoid 1vX)
**{clutcher_name}**
- **Stats**: {clutcher_stat}
- **Protocol**: ALWAYS double-peek, never give 1v1 clutch

## Team Patterns
- **Trade Efficiency**: {trade_eff:.1f}% - {"Strong trading" if trade_eff >= 40 else "Die isolated"}
- **First Blood**: {fk_rate:.1f}% rate, {fk_conv:.1f}% conversion""",
            
            "compositions": f"""## Agent Core
{agent_core}

## Role Distribution
{role_summary}

## Default Setups

### Defense Formation
- **Primary Site**: {"A-Site" if site_a > site_b else "B-Site"} ({max(site_a, site_b):.1f}% preference)
- **Classification**: {"Duelist-heavy" if "duelist" in role_summary.lower() else "Balanced"}

### Attack Formation
- **Target Site**: {"A-Site" if site_a > site_b else "B-Site"} favored
- **Style**: {"Fast execute" if fk_rate > 50 else "Slow default"}""",
            
            "counters": f"""## Agent Counters
Based on their agent pool, prepare counters for:
- {agent_core}

## Site Control Denial
- **Their Preference**: {"A-Site" if site_a > site_b else "B-Site"} ({max(site_a, site_b):.1f}%)
- **Counter**: Stack {"A" if site_a > site_b else "B"} on defense

## Economic Warfare
- **Anti-Eco Protocol**: {"Play long range - {eco_wr:.1f}% eco danger" if eco_wr > 20 else "Push aggressively - low eco threat"}
- **Anti-Pistol**: {pistol_wr:.1f}% pistol rate - {"Respect their pistol" if pistol_wr >= 55 else "Winnable"}

## Setup Breakers
- **Trading**: {"They trade well - avoid isolated peeks" if trade_eff >= 40 else "They die alone - take 1v1s"}""",
            
            "how_to_win": f"""## 🎯 Win Condition Summary
**Primary**: Neutralize {opener_name} early, hunt {weak_link}
**Avoid**: Giving {clutcher_name} clutch opportunities

## 📋 Round Protocol

### Pistol (Round 1)
- Their rate: {pistol_wr:.1f}%
- **Strategy**: {"Play for trades" if pistol_wr >= 55 else "Aggressive entry"}

### Anti-Eco (Rounds 2-3)
- Their eco rate: {eco_wr:.1f}%
- **Protocol**: {"Long range, no hero plays" if eco_wr > 20 else "Push aggressively"}

### Gun Rounds
- **Attack**: Target {"B-Site" if site_a > site_b else "A-Site"} (they stack other)
- **Defense**: Stack their preferred site

## 🔄 IF/THEN Decision Tree

### Opening Duel
- **IF** we get FB (their conv: {fk_conv:.1f}%) → {"Play fast 5v4" if fk_conv < 70 else "Methodical"}
- **IF** we lose FB → {"Trade immediately" if trade_eff >= 40 else "Reset"}

### Man Advantage
- **5v4**: {"Disciplined - they trade" if trade_eff >= 40 else "Push - they die alone"}
- **4v5**: {"Save" if fk_conv >= 70 else "Try retake"}

## 👤 Player Assignments
- **HUNT**: {opener_name}
- **ISOLATE**: {weak_link}
- **AVOID 1v1**: {clutcher_name}

## 💰 Economy
- **Their Eco**: {"RESPECT" if eco_wr > 20 else "PUNISH"}
- **Full Buy**: {full_buy_wr:.1f}% rate"""
        }
