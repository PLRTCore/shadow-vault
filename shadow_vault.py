"""
shadow_vault_full.py
====================
ShadowVault — Persistent Research Memory Store
with scored retrieval, deduplication, pruning, rediscovery,
and Grok Research Migrator — all in one file.

Run:
    python shadow_vault_full.py

COMMAND SET
-----------
  Shadow Vault: Help           Show this command list
  Shadow Vault: Save           Manually save a new entry
  Shadow Vault: Edit           Edit an existing entry by ID
  Shadow Vault: Delete         Delete an entry by ID
  Shadow Vault: Recall         Browse all entries by number, open any to read
  Shadow Vault: Recall Rare    Browse least-accessed entries (rediscovery mode)
  Shadow Vault: Search         Search entries by keyword query
  Shadow Vault: Summarise      Save a session summary
  Shadow Vault: Reinforce      Integrity check, stats, and prune pass
  Shadow Vault: Migrate        Import a Grok JSON export file
  Shadow Vault: Migrate Paste  Paste a Grok JSON block directly
  Shadow Vault: Migrate Dry    Preview a Grok JSON import without saving
  Shadow Vault: Prompt         Print the Grok extraction prompt
  Shadow Vault: Exit           Exit

GROK MIGRATION WORKFLOW
-----------------------
1. Run:  Shadow Vault: Prompt
2. Copy the printed prompt and paste it into your Grok chat.
3. Grok outputs a JSON block. Save it as a .json file.
4. Run:  Shadow Vault: Migrate  and enter the file path when asked.
5. Run:  Shadow Vault: Recall   to browse imported entries.

ENTRY STORAGE FORMAT
--------------------
Migrated entries are stored with a metadata header line so tags and
importance are visible when reading inside Recall:

    [importance: 0.85 | tags: topic, subtopic]
    Full research text...

Tags and importance are ALSO stored as vault metadata so the scoring
engine (relevance, recency, rarity, tag bonus) can use them for
retrieval and rediscovery.

RETRIEVAL SCORING
-----------------
  relevance   0.45  — token overlap with query
  importance  0.23  — entry importance (0.0–1.0)
  recency     0.14  — decay over ~150 days
  rarity      0.10  — favors less-accessed entries
  tag_bonus  +0.08  — flat bonus added when entry tags match query

PRUNING
-------
Entries with importance < 0.35 AND access_count < 3 are pruned
automatically when the vault exceeds max_entries (default 800).
Pruning is capped at 10% of the vault per pass.
Research entries imported via Migrate use the importance value
assigned by Grok, so important findings are never pruned.
"""

import contextlib
import json
import os
import re
import tempfile
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "prune_threshold":    0.35,   # importance < this AND access < 3 → prunable
    "duplicate_threshold": 0.78,  # Jaccard similarity threshold for dedup
    "dedup_window":       300,    # recent entries to check for duplicates
    "max_entries":        800,    # soft cap; triggers pruning when exceeded
    "max_backups":        8,      # rotating backup files to keep
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _multiline_input(prompt: str) -> str:
    print(f"{prompt} (type END on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW VAULT
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowVault:
    """
    Persistent key-value research memory store with scored retrieval,
    Jaccard deduplication, importance-based pruning, and rediscovery.
    """

    def __init__(self, filepath: str = "vault.json",
                 backup_dir: str = "vault_backups") -> None:
        self.filepath   = filepath
        self.backup_dir = backup_dir
        self.entries:        dict                    = {}
        self.summaries:      dict                    = {}
        self.tags:           defaultdict[str, set]   = defaultdict(set)
        self.config:         dict                    = dict(DEFAULT_CONFIG)
        self.recent_entries: deque                   = deque(
            maxlen=self.config.get("dedup_window", 300)
        )
        self._last_backup:   str                     = ""
        self.load()

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> set[str]:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        stop = {
            'the', 'and', 'for', 'with', 'that', 'this', 'from',
            'are', 'was', 'were', 'be', 'have', 'has',
        }
        stems = set()
        for w in text.split():
            if len(w) <= 2 or w in stop:
                continue
            # Apply at most one suffix strip per word
            for suffix in ('ing', 'ed', 'ly', 'es', 's'):
                stripped = w.removesuffix(suffix)
                if stripped != w and len(stripped) >= 3:
                    w = stripped
                    break
            if len(w) >= 3:
                stems.add(w)
        return stems

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _jaccard(self, text1: str, text2: str) -> float:
        t1, t2 = self._tokenize(text1), self._tokenize(text2)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def _is_duplicate(self, new_text: str) -> str | None:
        """
        Check the most recent dedup_window entries for a near-duplicate.
        Returns the existing entry_id if found, else None.
        """
        threshold = self.config.get("duplicate_threshold", 0.78)
        for eid in self.recent_entries:
            if eid in self.entries:
                if self._jaccard(new_text, self.entries[eid]["text"]) >= threshold:
                    return eid
        return None

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_low_value(self, force_persist: bool = False) -> None:
        """
        Delete entries where importance < prune_threshold AND access_count < 3.
        Capped at 10% of vault per call to avoid sudden large purges.
        """
        threshold = self.config.get("prune_threshold", 0.35)
        to_delete = [
            eid for eid, e in list(self.entries.items())
            if e.get("importance", 0.5) < threshold
            and e.get("access_count", 0) < 3
        ]
        if to_delete:
            cap        = max(5, int(len(self.entries) * 0.1))
            delete_set = set(to_delete[:cap])
            for eid in delete_set:
                del self.entries[eid]
                self.tags.pop(eid, None)
            self.recent_entries = deque(
                (x for x in self.recent_entries if x not in delete_set),
                maxlen=self.config.get("dedup_window", 300),
            )
            print(f"[ShadowVault] Pruned {len(delete_set)} low-value entries.")

        if force_persist or to_delete:
            self.persist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist(self) -> None:
        """
        Write vault to disk atomically and rotate backups.

        Uses write-to-temp + os.replace so a crash or power loss mid-write
        never leaves vault.json truncated or corrupt. os.replace is atomic
        on POSIX; on Windows it is as close to atomic as the OS allows.
        """
        Path(self.backup_dir).mkdir(exist_ok=True)
        data = {
            "entries":   self.entries,
            "summaries": self.summaries,
            "tags":      {k: list(v) for k, v in self.tags.items()},
        }
        tmp_path   = None
        tmp_backup = None
        try:
            # Write to a temp file in the same directory, then atomically replace.
            # Same-directory temp ensures os.replace stays on one filesystem.
            vault_dir = Path(self.filepath).parent
            with tempfile.NamedTemporaryFile(
                "w", dir=vault_dir, delete=False, suffix=".tmp"
            ) as tf:
                json.dump(data, tf, indent=2)
                tmp_path = tf.name
            os.replace(tmp_path, self.filepath)
            tmp_path = None  # replaced successfully — no cleanup needed

            # Rotate backups — write at most one per hour, keep max_backups
            max_backups  = self.config.get("max_backups", 8)
            current_hour = datetime.now().strftime("%Y%m%d_%H")
            if self._last_backup != current_hour:
                bpath = (
                    Path(self.backup_dir)
                    / f"vault_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                )
                with tempfile.NamedTemporaryFile(
                    "w", dir=Path(self.backup_dir), delete=False, suffix=".tmp"
                ) as tf:
                    json.dump(data, tf, indent=2)
                    tmp_backup = tf.name
                os.replace(tmp_backup, bpath)
                tmp_backup = None  # replaced successfully — no cleanup needed
                self._last_backup = current_hour

                backups = sorted(Path(self.backup_dir).glob("vault_*.json"))
                if len(backups) > max_backups:
                    for old in backups[:-max_backups]:
                        old.unlink()

        except Exception as e:
            # Clean up any orphaned temp files — suppress OSError in case they
            # were already removed or never created. Done before printing so the
            # original exception is always what's reported.
            for p in filter(None, [tmp_path, tmp_backup]):
                with contextlib.suppress(OSError):
                    os.unlink(p)
            print(f"[ShadowVault] Persist warning ({type(e).__name__}): {e}")

    def load(self) -> None:
        """Load vault from disk, rebuilding recent_entries and tags."""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath) as f:
                data = json.load(f)

            # Support both the old flat format {id: {text,...}} and the
            # new nested format {entries: {...}, summaries: {...}, tags: {...}}
            if "entries" in data:
                self.entries   = data.get("entries", {})
                self.summaries = data.get("summaries", {})
                for eid, taglist in data.get("tags", {}).items():
                    self.tags[eid] = set(taglist)
            else:
                # Legacy flat format — migrate
                for eid, value in data.items():
                    if isinstance(value, str):
                        self.entries[eid] = {
                            "text":         value,
                            "created":      _now(),
                            "last_edited":  _now(),
                            "importance":   0.7,
                            "access_count": 0,
                        }
                    else:
                        self.entries[eid] = value

            # Rebuild recent_entries from most-recently-edited
            window      = self.config.get("dedup_window", 300)
            recent_list = sorted(
                self.entries.items(),
                key=lambda x: x[1].get("last_edited", ""),
                reverse=True,
            )[:window]
            self.recent_entries = deque(
                [eid for eid, _ in recent_list], maxlen=window
            )

            self._prune_low_value(force_persist=False)

        except Exception as e:
            print(f"[ShadowVault] Load warning: {e}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, entry_id: str, text: str,
             tags: list[str] | None = None,
             importance: float = 0.7) -> bool:
        """
        Save a memory entry with optional tags and importance score.

        If a near-duplicate already exists within the dedup window, the
        existing entry's importance is bumped instead of creating a new one.

        importance guide:
          1.0  — core conclusion, central finding
          0.85 — important supporting insight
          0.70 — useful context, methodology note  (default)
          0.55 — minor detail, tangential observation
          0.40 — speculative or low-confidence note

        Returns True on success, False if entry_id or text is empty.
        """
        entry_id = str(entry_id).strip()
        if not entry_id or not text.strip():
            print("Invalid entry ID or empty text.")
            return False

        # Deduplication check
        existing = self._is_duplicate(text)
        if existing:
            self.entries[existing]["last_edited"] = _now()
            self.entries[existing]["importance"]  = min(
                1.0, self.entries[existing].get("importance", 0.7) + 0.08
            )
            print(f"[ShadowVault] Deduplicated — reinforced '{existing}'")
            self.persist()
            return True

        self.entries[entry_id] = {
            "text":         text.strip(),
            "created":      _now(),
            "last_edited":  _now(),
            "importance":   max(0.0, min(1.0, float(importance))),
            "access_count": 0,
        }
        if tags:
            for t in tags:
                self.tags[entry_id].add(str(t).strip().lower())

        self.recent_entries.append(entry_id)

        # Only prune when the vault has exceeded max_entries — not on every save
        if len(self.entries) > self.config.get("max_entries", 800):
            self._prune_low_value(force_persist=True)

        self.persist()
        print(f"[ShadowVault] Saved '{entry_id}'")
        return True

    def edit(self, entry_id: str, new_text: str) -> None:
        entry_id = entry_id.strip()
        if not entry_id or not new_text.strip():
            print("Invalid input.")
            return
        if entry_id not in self.entries:
            print(f"Entry '{entry_id}' not found.")
            return
        self.entries[entry_id]["text"]        = new_text.strip()
        self.entries[entry_id]["last_edited"] = _now()
        self.persist()
        print(f"Entry '{entry_id}' updated.")

    def delete(self, entry_id: str) -> None:
        entry_id = entry_id.strip()
        if entry_id not in self.entries:
            print(f"Entry '{entry_id}' not found.")
            return
        confirm = input(f"Delete '{entry_id}'? (y/n): ").strip().lower()
        if confirm == "y":
            del self.entries[entry_id]
            self.tags.pop(entry_id, None)
            self.recent_entries = deque(
                (x for x in self.recent_entries if x != entry_id),
                maxlen=self.config.get("dedup_window", 300),
            )
            self.persist()
            print(f"Entry '{entry_id}' deleted.")
        else:
            print("Delete cancelled.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _sorted_ids(self) -> list[str]:
        """Alphabetical — matches Recall numbering."""
        return sorted(self.entries.keys())

    def get_relevant_memories(self, query: str, top_k: int = 10,
                               rediscovery_mode: bool = False) -> list[dict]:
        """
        Return up to top_k entries ranked by composite score.
        Returns list of dicts with keys: id, text, score.

        rediscovery_mode=True strongly favors entries with low access counts,
        surfacing forgotten entries that are still relevant to the query.
        """
        if not self.entries or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        query_lower  = query.lower()
        scored       = []
        now          = datetime.now()

        for eid, entry in self.entries.items():
            text = entry["text"]
            if not text.strip():
                continue

            tokens    = self._tokenize(text)
            relevance = (
                len(query_tokens & tokens) / len(query_tokens)
                if query_tokens else 0.0
            )

            try:
                age_days = (now - datetime.fromisoformat(entry["last_edited"])).days
                recency  = max(0.2, 1.0 - (age_days / 150))
            except (ValueError, TypeError):
                recency = 0.6

            importance = entry.get("importance", 0.7)
            access     = entry.get("access_count", 0)

            rarity = (
                1.0 / (1.0 + access * 0.35)       # strongly favors low-access
                if rediscovery_mode
                else max(0.4, 1.0 - (access * 0.08))  # gently favors low-access
            )

            tag_bonus = 0.08 if any(
                tag in query_lower for tag in self.tags.get(eid, set())
            ) else 0.0

            if importance < 0.22 and not rediscovery_mode:
                continue

            score = (
                relevance  * 0.45 +
                importance * 0.23 +
                recency    * 0.14 +
                rarity     * 0.10 +
                tag_bonus
            )
            scored.append((score, eid, text))

        scored.sort(reverse=True, key=lambda x: x[0])

        results = []
        dirty: set[str] = set()
        for score, eid, text in scored[:top_k]:
            results.append({"id": eid, "text": text, "score": score})
            if eid in self.entries:
                self.entries[eid]["access_count"] = (
                    self.entries[eid].get("access_count", 0) + 1
                )
                dirty.add(eid)

        if dirty:
            self.persist()

        return results

    def get_rare_memories(self, query: str, top_k: int = 10) -> list[dict]:
        """Rediscovery mode — surfaces forgotten but relevant entries."""
        return self.get_relevant_memories(query, top_k=top_k, rediscovery_mode=True)

    def search(self, query: str, top_k: int = 12) -> list[tuple[str, str, float]]:
        """
        Return (entry_id, text, relevance) tuples ranked by token overlap.
        Does NOT increment access counts. Useful for inspection.
        """
        if not self.entries or not query.strip():
            return []
        query_tokens = self._tokenize(query)
        results = []
        for eid, entry in self.entries.items():
            tokens    = self._tokenize(entry["text"])
            relevance = (
                len(query_tokens & tokens) / len(query_tokens)
                if query_tokens else 0.0
            )
            results.append((relevance, eid, entry["text"]))
        results.sort(reverse=True, key=lambda x: x[0])
        return [(eid, text, rel) for rel, eid, text in results[:top_k]]

    # ── Summaries ─────────────────────────────────────────────────────────────

    def consolidate(self, summary_text: str, period: str = "session") -> None:
        """Store a high-level summary of a session or time period."""
        sid = f"summary_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.summaries[sid] = {
            "text":    summary_text.strip(),
            "period":  period,
            "created": _now(),
        }
        self.persist()
        print(f"[ShadowVault] Summary '{sid}' saved.")

    def get_summaries(self, top_k: int = 5) -> list[str]:
        """Return the most recent summaries."""
        recent = sorted(
            self.summaries.items(),
            key=lambda x: x[1]["created"],
            reverse=True,
        )
        return [s["text"] for _, s in recent[:top_k]]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "entries":        len(self.entries),
            "summaries":      len(self.summaries),
            "tagged_entries": len(self.tags),
            "recent_window":  len(self.recent_entries),
        }

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(self, rare_mode: bool = False) -> None:
        """
        Interactive browser. Lists entries alphabetically by ID.
        rare_mode=True shows least-accessed entries first for rediscovery.
        Enter a number to read the full body of that entry.
        """
        if not self.entries:
            print("Shadow Vault is currently empty.")
            return

        if rare_mode:
            # Sort by access_count ascending, then alphabetically
            ids = sorted(
                self.entries.keys(),
                key=lambda e: (self.entries[e].get("access_count", 0), e)
            )
            print("\n[Rediscovery Mode — least-accessed entries first]")
        else:
            ids = self._sorted_ids()

        dirty: set[str] = set()   # eids whose access_count changed this session

        while True:
            print("\n--- Shadow Vault: Entries ---")
            for i, eid in enumerate(ids, start=1):
                entry   = self.entries[eid]
                imp     = entry.get("importance", 0.7)
                access  = entry.get("access_count", 0)
                print(f"  {i:03d}.  {eid:<55}  imp:{imp:.2f}  access:{access}")
            print("---")
            choice = input("Enter number to open entry (or blank to exit): ").strip()
            if not choice:
                break
            if choice.isdigit() and 1 <= int(choice) <= len(ids):
                eid   = ids[int(choice) - 1]
                entry = self.entries[eid]
                tags  = sorted(self.tags.get(eid, set()))
                print(f"\n--- Entry: {eid} ---")
                print(entry["text"])
                print(f"\nImportance:  {entry.get('importance', 0.7):.2f}")
                print(f"Tags:        {', '.join(tags) if tags else '—'}")
                print(f"Access:      {entry.get('access_count', 0)}×")
                print(f"Created:     {entry.get('created', 'N/A')}")
                print(f"Last Edited: {entry.get('last_edited', 'N/A')}")
                print("---")
                # Increment access count — persist once on session exit, not per read
                self.entries[eid]["access_count"] = (
                    self.entries[eid].get("access_count", 0) + 1
                )
                dirty.add(eid)
            else:
                print("Invalid selection.")

        # One persist covers all access count updates for this browse session
        if dirty:
            self.persist()

    # ── Reinforce ─────────────────────────────────────────────────────────────

    def reinforce(self) -> None:
        """Full integrity check: disk sync, entry inspection, stats, prune pass."""
        print("\n--- Shadow Vault: Integrity Check ---")

        # Disk sync
        if not os.path.exists(self.filepath):
            print(f"{self.filepath} does not exist on disk.")
        else:
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                print(f"{self.filepath} — valid JSON.")
                disk_ids = set(data.get("entries", data).keys())
                mem_ids  = set(self.entries.keys())
                if disk_ids == mem_ids:
                    print("Disk and in-memory entries are in sync.")
                else:
                    only_disk = disk_ids - mem_ids
                    only_mem  = mem_ids  - disk_ids
                    if only_disk:
                        print(f"On disk but not in memory: {', '.join(sorted(only_disk))}")
                    if only_mem:
                        print(f"In memory but not on disk: {', '.join(sorted(only_mem))}")
            except json.JSONDecodeError as e:
                print(f"{self.filepath} is corrupted: {e}")

        # Entry inspection
        issues = []
        for eid, entry in self.entries.items():
            if not entry.get("text", "").strip():
                issues.append(f"  - '{eid}' has empty text.")
            if "created" not in entry:
                issues.append(f"  - '{eid}' is missing a created date.")
            if "last_edited" not in entry:
                issues.append(f"  - '{eid}' is missing a last_edited date.")

        if issues:
            print(f"Found {len(issues)} issue(s):")
            for issue in issues:
                print(issue)
        else:
            print(f"All {len(self.entries)} entries passed inspection.")

        # Stats
        s = self.stats()
        print(f"\nStats:")
        print(f"  Entries:        {s['entries']}")
        print(f"  Summaries:      {s['summaries']}")
        print(f"  Tagged entries: {s['tagged_entries']}")
        print(f"  Dedup window:   {s['recent_window']}")

        # Prune pass
        self._prune_low_value(force_persist=True)
        print("Integrity check complete.")
        print("---\n")


# ═══════════════════════════════════════════════════════════════════════════════
# GROK EXTRACTION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

GROK_EXTRACTION_PROMPT = """
Please review our entire conversation above and extract all significant research
findings, conclusions, insights, and reference points from YOUR responses only.

Output ONLY a valid JSON array — no explanation, no markdown fences, no preamble.

Each entry must follow this exact schema:
{
  "id": "<short_snake_case_slug describing the finding>",
  "text": "<COMPLETE, UNABBREVIATED text of the finding. Include all detail,
           all supporting evidence, all caveats. Do not summarise or truncate.
           This is the permanent record — write it as a full research note.>",
  "tags": ["<topic>", "<subtopic>", ...],
  "importance": <float 0.0-1.0>
}

Importance guide:
  1.0  — core conclusion, central finding, key definition
  0.85 — important supporting insight or significant data point
  0.70 — useful context, methodology note, secondary finding
  0.55 — minor detail, tangential observation
  0.40 — speculative, uncertain, or low-confidence note

Rules:
- Preserve the complete content of each finding exactly as discussed in the
  conversation. Do not condense, paraphrase, or shorten — this is the
  permanent record and must contain everything.
- Write each entry so it stands alone as a full record. Do not use phrases
  like "as mentioned above", "as discussed", or "see previous response" —
  instead, restate any necessary context directly inside the entry text.
- Do not summarise your reasoning process — only findings and conclusions
- Merge closely related points into one entry rather than splitting them
- Use specific, descriptive IDs (e.g. "protein_folding_alphafold_accuracy")
- Tags should be lowercase, 1-3 words each, 2-5 tags per entry
- text must be COMPLETE — never truncate with "..." or "etc."
- Output nothing except the JSON array
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MigrationError(Exception):
    """Raised for unrecoverable migration failures."""


def _format_entry_text(entry: dict) -> str:
    """
    Compose the full text to store in the vault.
    Writes a metadata header so tags and importance are visible in Recall,
    even though they are also stored as proper vault metadata fields.

        [importance: 0.85 | tags: topic, subtopic]
        Full research text...
    """
    text       = entry["text"].strip()
    tags       = [str(t).strip().lower() for t in entry.get("tags", []) if str(t).strip()]
    importance = float(entry.get("importance", 0.7))

    header_parts = [f"importance: {importance:.2f}"]
    if tags:
        header_parts.append(f"tags: {', '.join(tags)}")
    header = "[" + " | ".join(header_parts) + "]"

    return f"{header}\n{text}"


def _is_valid_slug(s: str) -> bool:
    """
    Loosely validates snake_case slug format.
    Must start with a lowercase letter to avoid purely numeric or
    digit-leading IDs that are harder to read in Recall listings.
    """
    return bool(re.match(r'^[a-z][a-z0-9_]*$', s.strip()))


def _validate_entry(entry: dict, index: int) -> list[str]:
    """Return validation issues. Warnings are tagged '[warn]'."""
    errors = []

    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        errors.append(f"[{index}] Missing or invalid 'id'")
    elif not _is_valid_slug(entry_id):
        errors.append(
            f"[{index}] [warn] 'id' should be snake_case "
            f"(got '{entry_id}'). Proceeding anyway."
        )

    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"[{index}] Missing or invalid 'text'")
    elif text.strip().endswith("...") or text.strip().endswith("…"):
        errors.append(
            f"[{index}] [warn] 'text' appears truncated (ends with '…'). "
            "Re-run extraction with the updated prompt."
        )

    tags = entry.get("tags")
    if not isinstance(tags, list):
        errors.append(f"[{index}] 'tags' must be a list")
    else:
        for ti, tag in enumerate(tags):
            if not isinstance(tag, str) or not tag.strip():
                errors.append(f"[{index}] tag[{ti}] is empty or not a string")

    imp = entry.get("importance")
    if not isinstance(imp, (int, float)) or not (0.0 <= imp <= 1.0):
        errors.append(f"[{index}] 'importance' must be a float 0.0–1.0")

    return errors


def _validate_payload(data: list) -> tuple[list, list, list]:
    """Returns (valid_entries, hard_errors, warnings)."""
    if not isinstance(data, list):
        return [], ["Root JSON value must be an array."], []

    valid, hard_errors, warnings = [], [], []
    for i, entry in enumerate(data):
        errs       = _validate_entry(entry, i)
        entry_hard = [e for e in errs if "[warn]" not in e]
        entry_warn = [e for e in errs if "[warn]" in e]
        hard_errors.extend(entry_hard)
        warnings.extend(entry_warn)
        if not entry_hard:
            valid.append(entry)

    return valid, hard_errors, warnings


def _load_json_file(path: str) -> list:
    p = Path(path.strip())
    if not p.exists():
        raise MigrationError(f"File not found: {path}")
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise MigrationError(f"Invalid JSON in {path}: {e}") from e


def _load_json_paste() -> list:
    print("Paste your Grok JSON output below.")
    print("Type END on a new line when done:\n")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    raw = "\n".join(lines).strip()
    # Strip markdown code fences — handles ```json, ``` json, ``` etc.
    raw = re.sub(r'^```[a-zA-Z]*\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise MigrationError(f"Could not parse pasted JSON: {e}") from e


def _report_validation(
    raw_data: list,
    valid_entries: list,
    hard_errors: list[str],
    warnings: list[str],
) -> bool:
    """
    Print validation results and return True if ingestion should proceed.
    Extracted from _run_migration for clarity and testability.
    """
    if hard_errors:
        print(f"\n⚠️  Validation errors ({len(hard_errors)}):")
        for e in hard_errors:
            print(f"   {e}")
    if warnings:
        print(f"\n⚡ Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"   {w}")

    if not valid_entries:
        print("No valid entries to import. Aborting.")
        return False

    skipped_validation = len(raw_data) - len(valid_entries)
    print(
        f"\n✓ {len(valid_entries)} valid "
        f"entr{'y' if len(valid_entries) == 1 else 'ies'} ready"
        + (f"  ({skipped_validation} skipped — validation errors)"
           if skipped_validation else "")
    )
    return True


def _preview_entry(
    eid: str,
    text: str,
    entry: dict,
    vault: ShadowVault,
    counts: dict[str, int],
) -> None:
    """
    Print what would happen to one entry in dry-run mode.
    Mutates counts dict (keys: saved, skipped) in place.
    Extracted from _run_migration for clarity and testability.
    """
    raw_preview = entry["text"].strip()
    preview     = f"{raw_preview[:100]}{'…' if len(raw_preview) > 100 else ''}"
    importance  = float(entry.get("importance", 0.7))
    tags        = [str(t).strip().lower() for t in entry.get("tags", []) if str(t).strip()]

    if eid in vault.entries and vault.entries[eid]["text"] == text:
        print(f"  [SKIP]   '{eid}' — identical content already exists.")
        counts["skipped"] += 1
    elif eid in vault.entries:
        print(f"  [UPDATE] '{eid}' — ID exists, content differs  "
              f"imp={importance:.2f}  tags={tags}")
        print(f"           {preview}")
        counts["saved"] += 1
    else:
        n_approx = len(vault.entries) + counts["saved"] + 1
        print(f"  [SAVE]   #{n_approx:03d}  '{eid}'  "
              f"imp={importance:.2f}  tags={tags}")
        print(f"           {preview}")
        counts["saved"] += 1


def _ingest_entries(
    valid_entries: list,
    vault: ShadowVault,
) -> dict[str, int]:
    """
    Save validated entries into the vault and return outcome counts.
    Extracted from _run_migration for clarity and testability.
    Returns dict with keys: saved, skipped, errors.
    """
    counts: dict[str, int] = {"saved": 0, "skipped": 0, "errors": 0}

    for entry in valid_entries:
        eid        = entry["id"].strip()
        text       = _format_entry_text(entry)
        tags       = [str(t).strip().lower() for t in entry.get("tags", []) if str(t).strip()]
        importance = float(entry.get("importance", 0.7))

        if eid in vault.entries and vault.entries[eid]["text"] == text:
            print(f"  [SKIP]  '{eid}' — identical content already exists.")
            counts["skipped"] += 1
            continue

        action      = "UPDATE" if eid in vault.entries else "SAVE"
        size_before = len(vault.entries)
        ok          = vault.save(eid, text, tags=tags or None, importance=importance)

        if not ok:
            print(f"  [ERROR] vault.save() rejected '{eid}'.")
            counts["errors"] += 1
        elif len(vault.entries) == size_before:
            print(f"  [DEDUP]  '{eid}' — merged into existing near-duplicate.")
            counts["saved"] += 1
        else:
            n = vault._sorted_ids().index(eid) + 1
            print(f"  [{action}]  #{n:03d}  '{eid}'")
            counts["saved"] += 1

    return counts


def _run_migration(vault: ShadowVault, raw_data: list, dry_run: bool = False) -> None:
    """
    Validate and ingest a list of raw Grok entries into the vault.
    Coordinates _report_validation, _preview_entry, and _ingest_entries.
    """
    valid_entries, hard_errors, warnings = _validate_payload(raw_data)

    if not _report_validation(raw_data, valid_entries, hard_errors, warnings):
        return

    if dry_run:
        print("\n[DRY RUN — nothing will be written]\n")
        counts: dict[str, int] = {"saved": 0, "skipped": 0, "errors": 0}
        for entry in valid_entries:
            eid  = entry["id"].strip()
            text = _format_entry_text(entry)
            _preview_entry(eid, text, entry, vault, counts)
    else:
        counts = _ingest_entries(valid_entries, vault)

    print(f"\n{'=' * 50}")
    print("Migration complete." + ("  (dry run)" if dry_run else ""))
    print(f"  Saved:    {counts['saved']}")
    print(f"  Skipped:  {counts['skipped']}")
    if counts.get("errors"):
        print(f"  Errors:   {counts['errors']}")
    if not dry_run and counts["saved"]:
        print(f"  Vault:    {len(vault.entries)} entries total")
        print("  Run 'Shadow Vault: Recall' to browse imported entries.")
    print(f"{'=' * 50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LOOP
# ═══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
╔════════════════════════════════════════════════════════════════╗
║                 SHADOW VAULT — COMMAND SET                     ║
╠════════════════════════════════════════════════════════════════╣
║  VAULT                                                         ║
║  Shadow Vault: Save           Save a new entry manually        ║
║  Shadow Vault: Edit           Edit an entry by ID              ║
║  Shadow Vault: Delete         Delete an entry by ID            ║
║  Shadow Vault: Recall         Browse and read all entries      ║
║  Shadow Vault: Recall Rare    Browse least-accessed entries    ║
║  Shadow Vault: Search         Search entries by keyword        ║
║  Shadow Vault: Summarise      Save a session summary           ║
║  Shadow Vault: Reinforce      Integrity check, stats, prune    ║
║                                                                ║
║  MIGRATION                                                     ║
║  Shadow Vault: Prompt         Print Grok extraction prompt     ║
║  Shadow Vault: Migrate        Import a Grok .json file         ║
║  Shadow Vault: Migrate Paste  Paste JSON directly              ║
║  Shadow Vault: Migrate Dry    Preview import (no write)        ║
║                                                                ║
║  SYSTEM                                                        ║
║  Shadow Vault: Help           Show this command list           ║
║  Shadow Vault: Exit           Exit                             ║
╚════════════════════════════════════════════════════════════════╝
""".strip()


def main() -> None:
    vault = ShadowVault()
    print("=== SHADOW VAULT INITIALIZED ===\n")
    print(HELP_TEXT)
    print()

    while True:
        try:
            command = input("Shadow Vault Command: ").strip().lower()

            # ── System ───────────────────────────────────────────────────────
            if command == "shadow vault: exit":
                print("Goodbye.")
                break

            elif command == "shadow vault: help":
                print(f"\n{HELP_TEXT}\n")

            # ── Vault ────────────────────────────────────────────────────────
            elif command == "shadow vault: recall":
                vault.recall(rare_mode=False)

            elif command == "shadow vault: recall rare":
                vault.recall(rare_mode=True)

            elif command == "shadow vault: search":
                query   = input("Search query: ").strip()
                results = vault.search(query, top_k=12)
                if not results:
                    print("No results found.")
                else:
                    print(f"\n--- Search results for: '{query}' ---")
                    for eid, text, relevance in results:
                        preview = f"{text[:120]}{'…' if len(text) > 120 else ''}"
                        print(f"\n  [{relevance:.2f}]  {eid}")
                        print(f"           {preview}")
                    print("---")

            elif command == "shadow vault: save":
                eid  = input("Entry ID: ").strip()
                text = _multiline_input("Entry text")
                imp_raw = input("Importance (0.0–1.0, default 0.7): ").strip()
                try:
                    importance = float(imp_raw) if imp_raw else 0.7
                except ValueError:
                    importance = 0.7
                tags_raw = input("Tags (comma-separated, or blank): ").strip()
                tags     = [t.strip().lower() for t in tags_raw.split(",") if t.strip()] or None
                vault.save(eid, text, tags=tags, importance=importance)

            elif command == "shadow vault: edit":
                eid = input("Entry ID to edit: ").strip()
                if eid in vault.entries:
                    print(f"\n--- Current Entry: {eid} ---")
                    print(vault.entries[eid]["text"])
                    print("---\n")
                    new_text = _multiline_input("New text")
                    vault.edit(eid, new_text)
                else:
                    print(f"Entry '{eid}' not found.")

            elif command == "shadow vault: delete":
                eid = input("Entry ID to delete: ").strip()
                vault.delete(eid)

            elif command == "shadow vault: summarise":
                period  = input("Period label (e.g. 'session', 'week', or blank): ").strip()
                period  = period if period else "session"
                summary = _multiline_input("Summary text")
                vault.consolidate(summary, period=period)

            elif command == "shadow vault: reinforce":
                vault.reinforce()

            # ── Migration ────────────────────────────────────────────────────
            elif command == "shadow vault: prompt":
                print("\n" + "=" * 60)
                print("GROK EXTRACTION PROMPT — paste into your Grok chat:")
                print("=" * 60 + "\n")
                print(GROK_EXTRACTION_PROMPT)
                print("\n" + "=" * 60 + "\n")

            elif command == "shadow vault: migrate":
                path = input("Path to Grok JSON file: ").strip()
                try:
                    raw_data = _load_json_file(path)
                    _run_migration(vault, raw_data, dry_run=False)
                except MigrationError as e:
                    print(f"ERROR: {e}")

            elif command == "shadow vault: migrate paste":
                try:
                    raw_data = _load_json_paste()
                    _run_migration(vault, raw_data, dry_run=False)
                except MigrationError as e:
                    print(f"ERROR: {e}")

            elif command == "shadow vault: migrate dry":
                path = input("Path to Grok JSON file (dry run): ").strip()
                try:
                    raw_data = _load_json_file(path)
                    _run_migration(vault, raw_data, dry_run=True)
                except MigrationError as e:
                    print(f"ERROR: {e}")

            else:
                print("Unknown command. Type 'Shadow Vault: Help' to see all commands.")

        except KeyboardInterrupt:
            print("\nShadow Vault is Sealed.")
            break


if __name__ == "__main__":
    main()
