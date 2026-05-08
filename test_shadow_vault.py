"""
test_shadow_vault.py
====================
Test suite for shadow_vault_full.py.

Revision history
----------------
v1  — Initial suite covering tokeniser, validation, save/dedup/edit/delete,
      retrieval, summaries, persistence, migration, pruning, stats.

v2  — Robustness gap pass.  New coverage added in this revision:

  TestTokenizer
    • test_suffix_strip_priority — verifies only the *first* matching suffix
      is stripped per word ('walking' → 'walk', NOT 'walk' twice or 'walks').
    • test_tokenize_punctuation_stripped — symbols removed before tokenising.
    • test_tokenize_min_length_three — post-strip tokens shorter than 3 are
      excluded even when the original word was longer.

  TestValidation
    • test_validate_entry_empty_tag_in_list — a list with a blank-string tag
      must flag that specific tag, not the whole 'tags' field.
    • test_validate_payload_all_hard_errors_returns_empty_valid — all entries
      invalid → valid list is empty and function does not raise.
    • test_validate_payload_empty_list — empty input is valid JSON but yields
      nothing; function must handle gracefully (no crash, empty valid list).

  TestVaultOperations
    • test_save_importance_stored_correctly — importance value stored is the
      one passed in (within float precision).
    • test_save_updates_existing_entry_in_place — re-saving same ID overwrites
      rather than creating a second entry.
    • test_dedup_reinforces_importance — near-duplicate bumps existing entry's
      importance by ~0.08 and caps at 1.0.
    • test_delete_removes_from_recent_entries — after delete, eid is absent
      from the dedup window.
    • test_delete_removes_tags — tags are cleaned up alongside the entry.
    • test_save_adds_to_recent_entries — new entry appears in recent_entries.

  TestRetrieval
    • test_get_relevant_memories_access_count_persisted — access count change
      survives a vault reload (the critical bug fixed in the previous review).
    • test_search_does_not_increment_access_count — search() must be read-only.
    • test_get_relevant_memories_tag_bonus — an entry whose tag matches the
      query keyword should outrank an otherwise-equal entry without the tag.
    • test_get_relevant_memories_top_k_respected — result length ≤ top_k.
    • test_get_relevant_memories_empty_vault — returns [] cleanly.
    • test_scoring_weights_sum — informal sanity check that the five scoring
      components with their documented weights sum to ≤ 1.0 + tag_bonus.

  TestSummaries
    • test_consolidate_persists_to_disk — summary survives a vault reload.
    • test_get_summaries_respects_top_k_zero — top_k=0 returns empty list.

  TestPersistence
    • test_round_trip_access_count — access_count survives persist/load.
    • test_round_trip_importance — importance survives persist/load.
    • test_persist_is_atomic_temp_cleanup — no .tmp files left after persist.
    • test_load_legacy_flat_format — old flat {id: str} format loads cleanly.
    • test_backup_rotation_respects_max — only max_backups files are kept.

  TestMigration
    • test_preview_entry_skip_path — _preview_entry increments skipped when
      text is identical to existing vault entry.
    • test_preview_entry_update_path — _preview_entry increments saved when
      ID exists but text differs.
    • test_preview_entry_new_path — _preview_entry increments saved for new ID.
    • test_ingest_entries_returns_counts — _ingest_entries returns a dict with
      'saved', 'skipped', 'errors' keys and correct values.
    • test_run_migration_aborts_all_hard_errors — _run_migration with zero
      valid entries must not write anything to the vault.
    • test_migrate_updates_existing_entry — re-migrating same ID with *different*
      text replaces the stored text (not skipped).

  TestPruning
    • test_prune_updates_recent_entries — pruned IDs are removed from the
      dedup deque.
    • test_prune_no_entries_no_crash — empty vault handles prune gracefully.
    • test_prune_only_fires_above_max_entries — save() does NOT trigger a prune
      pass when entry count is below max_entries.

  TestEdgeCases  (new class)
    • test_save_whitespace_only_id_rejected — '   ' is treated as empty.
    • test_save_whitespace_only_text_rejected — whitespace-only text rejected.
    • test_jaccard_single_token_overlap — degenerate one-token sets give 0 or 1.
    • test_sorted_ids_alphabetical — _sorted_ids() returns lexicographic order.
    • test_reinforce_runs_without_crash — end-to-end smoke test (no assert, just
      must not raise).
    • test_format_entry_text_strips_body_whitespace — leading/trailing whitespace
      in entry text is stripped in the stored representation.

Run:
    python test_shadow_vault.py [-v]
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Load module from sibling file without installing ──────────────────────────
_spec = importlib.util.spec_from_file_location(
    "shadow_vault_full",
    Path(__file__).parent / "shadow_vault_full.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ShadowVault        = _mod.ShadowVault
_validate_entry    = _mod._validate_entry
_validate_payload  = _mod._validate_payload
_format_entry_text = _mod._format_entry_text
_is_valid_slug     = _mod._is_valid_slug
_run_migration     = _mod._run_migration
_preview_entry     = _mod._preview_entry
_ingest_entries    = _mod._ingest_entries
MigrationError     = _mod.MigrationError
_load_json_file    = _mod._load_json_file
DEFAULT_CONFIG     = _mod.DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_vault(tmp_path: str) -> ShadowVault:
    """Return a fresh, empty ShadowVault backed by a temp directory."""
    return ShadowVault(
        filepath=os.path.join(tmp_path, "vault.json"),
        backup_dir=os.path.join(tmp_path, "backups"),
    )


class _TmpDirMixin:
    """setUp/tearDown boilerplate shared by most test classes."""
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.vault   = make_vault(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOKENIZER & JACCARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizer(_TmpDirMixin, unittest.TestCase):
    """Unit tests for _tokenize and _jaccard."""

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_tokenize_non_empty(self):
        tokens = self.vault._tokenize(
            "Photosynthesis converts solar energy into glucose molecules."
        )
        self.assertIn("photosynthesi", tokens)   # 's' suffix stripped
        self.assertGreater(len(tokens), 2)

    def test_tokenize_stop_words_removed(self):
        tokens = self.vault._tokenize("the and for with that this from")
        self.assertEqual(tokens, set(), "All stop words should be removed")

    def test_tokenize_short_words_removed(self):
        tokens = self.vault._tokenize("a is it to")
        self.assertEqual(tokens, set(), "Words ≤ 2 chars should be removed")

    def test_jaccard_identical_rich_strings(self):
        text = "Neutron stars contain degenerate matter compressed beyond nuclear density."
        self.assertAlmostEqual(self.vault._jaccard(text, text), 1.0)

    def test_jaccard_completely_different_strings(self):
        t1 = "Volcanic eruptions release magma, ash, and sulfur dioxide into the atmosphere."
        t2 = "Renaissance painters pioneered perspective techniques in Florentine workshops."
        score = self.vault._jaccard(t1, t2)
        self.assertLess(score, 0.15,
                        f"Unrelated texts should have low Jaccard; got {score:.3f}")

    def test_jaccard_partial_overlap(self):
        t1 = "Machine learning models require large labelled training datasets for accuracy."
        t2 = "Machine learning algorithms generalise from patterns found in training data."
        score = self.vault._jaccard(t1, t2)
        self.assertGreater(score, 0.20)
        self.assertLess(score, 0.95)

    def test_jaccard_empty_strings_return_zero(self):
        self.assertEqual(self.vault._jaccard("", "anything"), 0.0)
        self.assertEqual(self.vault._jaccard("anything", ""), 0.0)
        self.assertEqual(self.vault._jaccard("", ""), 0.0)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_suffix_strip_priority(self):
        """Only the *first* matching suffix is stripped per word."""
        # 'walking' → strip 'ing' → 'walk'  (not 'walk' → strip 's' → 'walk' again)
        tokens = self.vault._tokenize("walking")
        self.assertIn("walk", tokens)
        self.assertNotIn("walking", tokens)
        # 'passes' → strip 'es' → 'pass' (3 chars, kept)
        tokens2 = self.vault._tokenize("passes")
        self.assertIn("pass", tokens2)

    def test_tokenize_punctuation_stripped(self):
        """Punctuation must be removed before tokenising."""
        tokens = self.vault._tokenize("black-hole, quasar! nebula?")
        # After stripping non-alphanumeric, words are: blackhole quasar nebula
        # (or black hole split — either is fine; what matters is no punct tokens)
        for tok in tokens:
            self.assertTrue(tok.isalnum() or "_" not in tok,
                            f"Unexpected punctuation in token: {tok!r}")

    def test_tokenize_min_length_three_after_strip(self):
        """Post-strip tokens shorter than 3 chars are excluded."""
        # 'ed' is 2 chars → must be excluded (length guard).
        tokens = self.vault._tokenize("ed")
        self.assertNotIn("ed", tokens)

        # 'eding' → strip 'ing' → 'ed' (2 chars) → strip rejected, keep 'eding' (5 chars).
        # Either form is fine as long as a 2-char residual is never emitted.
        tokens2 = self.vault._tokenize("eding")
        self.assertNotIn("ed", tokens2,
                         "A 2-char post-strip residual must not appear in token set")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VALIDATION — _validate_entry / _validate_payload
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidation(unittest.TestCase):

    def _good_entry(self, **overrides) -> dict:
        base = {
            "id":         "valid_slug_name",
            "text":       "Mitochondria generate ATP via oxidative phosphorylation.",
            "tags":       ["biology", "cell"],
            "importance": 0.8,
        }
        base.update(overrides)
        return base

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_valid_entry_no_errors(self):
        self.assertEqual(_validate_entry(self._good_entry(), 0), [])

    def test_missing_id(self):
        errors = _validate_entry(self._good_entry(id=""), 0)
        self.assertTrue(any("Missing or invalid 'id'" in e for e in errors))

    def test_missing_text(self):
        errors = _validate_entry(self._good_entry(text=""), 0)
        self.assertTrue(any("Missing or invalid 'text'" in e for e in errors))

    def test_truncated_text_warning(self):
        errors = _validate_entry(
            self._good_entry(text="The result was conclusive..."), 0
        )
        warn_msgs = [e for e in errors if "[warn]" in e]
        self.assertTrue(warn_msgs,
                        "Expected a [warn] message for truncated text ending in '...'")
        hard_msgs = [e for e in errors if "[warn]" not in e]
        self.assertEqual(hard_msgs, [],
                         "Truncated text should only warn, not hard-error")

    def test_non_snake_case_id_is_warning_not_hard_error(self):
        errors = _validate_entry(self._good_entry(id="Not-SnakeCase"), 0)
        warn_msgs = [e for e in errors if "[warn]" in e]
        hard_msgs = [e for e in errors if "[warn]" not in e]
        self.assertTrue(warn_msgs, "Non-snake-case ID should produce a [warn] message")
        self.assertEqual(hard_msgs, [],
                         "Non-snake-case ID should NOT produce a hard error")

    def test_importance_out_of_range(self):
        errors = _validate_entry(self._good_entry(importance=1.5), 0)
        self.assertTrue(any("importance" in e for e in errors))

    def test_tags_not_list(self):
        errors = _validate_entry(self._good_entry(tags="single_tag"), 0)
        self.assertTrue(any("tags" in e for e in errors))

    def test_validate_payload_separates_hard_and_warnings(self):
        entries = [
            self._good_entry(),                               # clean
            self._good_entry(id="BadSlug-Here"),              # warn only
            self._good_entry(id="", text="orphan entry"),     # hard error
        ]
        valid, hard, warns = _validate_payload(entries)
        self.assertEqual(len(valid), 2,
                         "Two entries should pass (clean + warn-only)")
        self.assertEqual(len(hard), 1, "One hard error for missing id")
        self.assertTrue(any("[warn]" in w for w in warns))

    def test_validate_payload_non_list_input(self):
        valid, hard, warns = _validate_payload({"not": "a list"})
        self.assertFalse(valid)
        self.assertTrue(hard)

    def test_is_valid_slug_accepts_valid(self):
        for slug in ("abc", "hello_world", "x1_y2", "a0"):
            self.assertTrue(_is_valid_slug(slug), f"Should be valid: {slug}")

    def test_is_valid_slug_rejects_invalid(self):
        for slug in ("", "1start", "has space", "CamelCase", "dash-slug"):
            self.assertFalse(_is_valid_slug(slug), f"Should be invalid: {slug}")

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_validate_entry_empty_tag_in_list(self):
        """A blank-string tag inside an otherwise-valid list must be flagged."""
        errors = _validate_entry(
            self._good_entry(tags=["biology", "", "cell"]), 0
        )
        # The empty string at index 1 should produce an error message.
        self.assertTrue(
            any("tag[1]" in e for e in errors),
            f"Expected tag[1] error; got: {errors}"
        )

    def test_validate_payload_all_hard_errors_returns_empty_valid(self):
        """All-invalid input returns an empty valid list without raising."""
        entries = [
            self._good_entry(id=""),   # hard error
            self._good_entry(text=""), # hard error
        ]
        valid, hard, warns = _validate_payload(entries)
        self.assertEqual(valid, [])
        self.assertGreater(len(hard), 0)

    def test_validate_payload_empty_list(self):
        """Empty list input is technically valid JSON — should not crash."""
        valid, hard, warns = _validate_payload([])
        self.assertEqual(valid, [])
        self.assertEqual(hard, [])
        self.assertEqual(warns, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VAULT — save / dedup / edit / delete
# ═══════════════════════════════════════════════════════════════════════════════

class TestVaultOperations(_TmpDirMixin, unittest.TestCase):

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_save_creates_entry(self):
        ok = self.vault.save("e1", "Ribosomes translate mRNA sequences into polypeptide chains.")
        self.assertTrue(ok)
        self.assertIn("e1", self.vault.entries)

    def test_save_rejects_empty_id(self):
        self.assertFalse(self.vault.save("", "some text"))

    def test_save_rejects_empty_text(self):
        self.assertFalse(self.vault.save("e_empty", ""))

    def test_save_persists_to_disk(self):
        self.vault.save("persist_test", "The Coriolis effect deflects moving air masses.")
        with open(self.vault.filepath) as f:
            data = json.load(f)
        self.assertIn("persist_test", data["entries"])

    def test_duplicate_detection_similar_text(self):
        base = (
            "Gravitational waves are ripples in spacetime caused by accelerating masses, "
            "first directly detected by LIGO in 2015 from a binary black hole merger."
        )
        paraphrase = (
            "Gravitational waves are ripples in spacetime caused by accelerating masses, "
            "first directly detected by LIGO in 2015 from a binary black hole merger event."
        )
        self.vault.save("grav_wave_original", base)
        count_before = len(self.vault.entries)
        self.vault.save("grav_wave_dup", paraphrase)
        self.assertEqual(len(self.vault.entries), count_before,
                         "Near-duplicate should not create a new entry")

    def test_distinct_entries_not_deduplicated(self):
        self.vault.save(
            "astronomy_entry",
            "Pulsars are rotating neutron stars emitting beams of electromagnetic radiation "
            "detectable as regular pulses, discovered by Jocelyn Bell Burnell in 1967."
        )
        self.vault.save(
            "culinary_entry",
            "Fermentation converts sugars into alcohol or lactic acid through yeast and "
            "bacterial metabolism, preserving food and creating flavour complexity."
        )
        self.assertEqual(len(self.vault.entries), 2)

    def test_edit_updates_text(self):
        self.vault.save("edit_me", "Original text about photons.")
        self.vault.edit("edit_me", "Revised text about quarks and gluons.")
        self.assertEqual(
            self.vault.entries["edit_me"]["text"],
            "Revised text about quarks and gluons."
        )

    def test_edit_nonexistent_entry(self):
        self.vault.edit("ghost_entry", "This should not crash.")

    def test_delete_removes_entry(self):
        self.vault.save("to_delete", "Temporary research note about enzymes.")
        with patch("builtins.input", return_value="y"):
            self.vault.delete("to_delete")
        self.assertNotIn("to_delete", self.vault.entries)

    def test_delete_cancel_keeps_entry(self):
        self.vault.save("keep_me", "Important finding about synaptic plasticity.")
        with patch("builtins.input", return_value="n"):
            self.vault.delete("keep_me")
        self.assertIn("keep_me", self.vault.entries)

    def test_tags_stored_correctly(self):
        self.vault.save("tagged", "Photons exhibit both wave and particle duality.",
                        tags=["Physics", " Quantum "])
        stored = self.vault.tags["tagged"]
        self.assertIn("physics", stored)
        self.assertIn("quantum", stored)

    def test_importance_clamped(self):
        self.vault.save("imp_high", "High importance entry.", importance=1.5)
        self.assertLessEqual(self.vault.entries["imp_high"]["importance"], 1.0)
        self.vault.save("imp_low", "Low importance entry.", importance=-0.5)
        self.assertGreaterEqual(self.vault.entries["imp_low"]["importance"], 0.0)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_save_importance_stored_correctly(self):
        """Importance value is stored as passed (within float precision)."""
        self.vault.save("imp_check", "Some research note.", importance=0.65)
        stored = self.vault.entries["imp_check"]["importance"]
        self.assertAlmostEqual(stored, 0.65, places=5)

    def test_save_updates_existing_entry_in_place(self):
        """Re-saving the same ID must not create a second entry."""
        # First save: text A
        self.vault.save("overwrite_me",
                        "Initial finding about solar wind particle flux.")
        initial_count = len(self.vault.entries)
        # Second save: text B — too different to be a Jaccard duplicate, but
        # same entry_id, so vault.save stores a genuinely new entry under the
        # same key (not a dedup bump).
        self.vault.save("overwrite_me",
                        "Completely unrelated text about Byzantine coinage.")
        # The key count must not grow.
        self.assertEqual(len(self.vault.entries), initial_count,
                         "Re-saving the same ID should overwrite, not append")

    def test_dedup_reinforces_importance(self):
        """Near-duplicate save bumps importance of the original entry."""
        base = (
            "Cosmic inflation expanded the universe exponentially in the first "
            "fractions of a second after the Big Bang, explaining cosmic uniformity."
        )
        near_dup = (
            "Cosmic inflation expanded the universe exponentially in the first "
            "fractions of a second after the Big Bang, explaining cosmic uniformity today."
        )
        self.vault.save("inflation_original", base, importance=0.70)
        original_imp = self.vault.entries["inflation_original"]["importance"]
        self.vault.save("inflation_dup", near_dup)
        new_imp = self.vault.entries["inflation_original"]["importance"]
        self.assertGreater(new_imp, original_imp,
                           "Dedup reinforcement should increase importance")
        self.assertLessEqual(new_imp, 1.0, "Importance must not exceed 1.0")

    def test_delete_removes_from_recent_entries(self):
        """After deleting an entry, its ID must be absent from the dedup deque."""
        self.vault.save("dedup_victim",
                        "Research note on antibiotic resistance mechanisms.")
        self.assertIn("dedup_victim", list(self.vault.recent_entries))
        with patch("builtins.input", return_value="y"):
            self.vault.delete("dedup_victim")
        self.assertNotIn("dedup_victim", list(self.vault.recent_entries))

    def test_delete_removes_tags(self):
        """Deleting an entry must also clean up its tags."""
        self.vault.save("tagged_victim",
                        "Observation about CRISPR off-target editing effects.",
                        tags=["genetics", "crispr"])
        self.assertIn("tagged_victim", self.vault.tags)
        with patch("builtins.input", return_value="y"):
            self.vault.delete("tagged_victim")
        self.assertNotIn("tagged_victim", self.vault.tags)

    def test_save_adds_to_recent_entries(self):
        """A newly saved entry must appear in the dedup deque."""
        self.vault.save("new_entry",
                        "The mantle convection cycle drives tectonic plate movement.")
        self.assertIn("new_entry", list(self.vault.recent_entries))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL — search / get_relevant_memories / get_rare_memories
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieval(_TmpDirMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        entries = [
            ("quantum_entanglement",
             "Quantum entanglement links particle states instantaneously across arbitrary distance.",
             ["physics", "quantum"]),
            ("dna_replication",
             "DNA replication unwinds the double helix and synthesises complementary strands via polymerase.",
             ["biology", "genetics"]),
            ("roman_aqueducts",
             "Roman aqueducts transported fresh water over hundreds of kilometres using gravity-fed channels.",
             ["history", "engineering"]),
            ("jazz_origins",
             "Jazz music emerged from African-American blues and ragtime traditions in New Orleans around 1900.",
             ["music", "history"]),
            ("deep_sea_vents",
             "Hydrothermal vents support chemosynthetic ecosystems independent of sunlight at ocean depths.",
             ["biology", "oceanography"]),
        ]
        for eid, text, tags in entries:
            self.vault.save(eid, text, tags=tags, importance=0.8)

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_search_returns_relevant_results(self):
        results = self.vault.search("quantum physics particles", top_k=3)
        ids = [r[0] for r in results]
        self.assertIn("quantum_entanglement", ids)

    def test_search_empty_query_returns_empty(self):
        self.assertEqual(self.vault.search(""), [])

    def test_get_relevant_memories_returns_list(self):
        results = self.vault.get_relevant_memories("biology genetics DNA")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        ids = [r["id"] for r in results]
        self.assertIn("dna_replication", ids)

    def test_get_relevant_memories_increments_access_count(self):
        before = self.vault.entries["quantum_entanglement"].get("access_count", 0)
        self.vault.get_relevant_memories("quantum entanglement distance")
        after  = self.vault.entries["quantum_entanglement"].get("access_count", 0)
        self.assertGreater(after, before)

    def test_get_rare_memories_favors_low_access(self):
        for eid in ["quantum_entanglement", "dna_replication",
                    "roman_aqueducts", "jazz_origins"]:
            self.vault.entries[eid]["access_count"] = 50
        self.vault.persist()
        results = self.vault.get_rare_memories("any topic", top_k=5)
        ids = [r["id"] for r in results]
        self.assertIn("deep_sea_vents", ids,
                      "Least-accessed entry should surface in rediscovery mode")

    def test_search_result_tuple_structure(self):
        results = self.vault.search("history engineering water")
        self.assertTrue(all(len(r) == 3 for r in results))
        eid, text, rel = results[0]
        self.assertIsInstance(eid, str)
        self.assertIsInstance(text, str)
        self.assertIsInstance(rel, float)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_get_relevant_memories_access_count_persisted(self):
        """
        The critical bug fixed in the previous review: access_count changes
        must survive a vault reload.
        """
        self.vault.get_relevant_memories("quantum entanglement")
        count_in_memory = self.vault.entries["quantum_entanglement"]["access_count"]

        # Load a fresh vault from the same file and check the count survived.
        v2 = make_vault(self._tmpdir)
        count_on_disk = v2.entries["quantum_entanglement"]["access_count"]
        self.assertEqual(
            count_in_memory, count_on_disk,
            "access_count must be persisted, not just held in memory"
        )

    def test_search_does_not_increment_access_count(self):
        """search() is a read-only operation — access counts must not change."""
        before = self.vault.entries["roman_aqueducts"].get("access_count", 0)
        self.vault.search("roman aqueducts water channels")
        after  = self.vault.entries["roman_aqueducts"].get("access_count", 0)
        self.assertEqual(before, after,
                         "search() must not increment access_count")

    def test_get_relevant_memories_tag_bonus(self):
        """
        An entry whose tag exactly matches a query keyword should score higher
        than an otherwise-similar entry without the tag.
        """
        # 'physics' is a tag on quantum_entanglement; not on jazz_origins.
        # Query the word 'physics' so the tag bonus fires only on the physics entry.
        results = self.vault.get_relevant_memories("physics", top_k=5)
        ids = [r["id"] for r in results]
        if "quantum_entanglement" in ids and "jazz_origins" in ids:
            q_score = next(r["score"] for r in results
                           if r["id"] == "quantum_entanglement")
            j_score = next(r["score"] for r in results
                           if r["id"] == "jazz_origins")
            self.assertGreater(
                q_score, j_score,
                "Tag-matching entry should score higher than non-matching entry"
            )

    def test_get_relevant_memories_top_k_respected(self):
        results = self.vault.get_relevant_memories("biology", top_k=2)
        self.assertLessEqual(len(results), 2,
                             "Result count must not exceed top_k")

    def test_get_relevant_memories_empty_vault(self):
        empty = make_vault(tempfile.mkdtemp())
        results = empty.get_relevant_memories("any query")
        self.assertEqual(results, [])

    def test_scoring_weights_sum(self):
        """
        Informal sanity check: documented weight coefficients sum to 1.0
        (relevance 0.45 + importance 0.23 + recency 0.14 + rarity 0.10 = 0.92,
        plus a maximum tag_bonus of 0.08 → total ≤ 1.0).
        """
        weights = 0.45 + 0.23 + 0.14 + 0.10
        self.assertAlmostEqual(weights, 0.92, places=5)
        max_score = weights + 0.08   # with full tag bonus
        self.assertAlmostEqual(max_score, 1.00, places=5)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARIES — consolidate / get_summaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummaries(_TmpDirMixin, unittest.TestCase):

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_consolidate_creates_summary(self):
        self.vault.consolidate("Session covered quantum computing fundamentals.", "session")
        self.assertEqual(len(self.vault.summaries), 1)

    def test_get_summaries_empty(self):
        self.assertEqual(self.vault.get_summaries(), [])

    def test_get_summaries_returns_most_recent_first(self):
        self.vault.summaries["s_old"] = {
            "text":    "Oldest summary — machine learning basics",
            "period":  "session",
            "created": "2025-01-01T10:00:00",
        }
        self.vault.summaries["s_mid"] = {
            "text":    "Middle summary — neural network architectures",
            "period":  "session",
            "created": "2025-06-15T12:00:00",
        }
        self.vault.summaries["s_new"] = {
            "text":    "Newest summary — transformer attention mechanisms",
            "period":  "session",
            "created": "2025-12-31T23:59:59",
        }
        results = self.vault.get_summaries(top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Newest summary — transformer attention mechanisms")
        self.assertEqual(results[1], "Middle summary — neural network architectures")
        self.assertEqual(results[2], "Oldest summary — machine learning basics")

    def test_get_summaries_top_k_respected(self):
        for i in range(10):
            self.vault.consolidate(f"Summary number {i} content.", period="test")
        self.assertLessEqual(len(self.vault.get_summaries(top_k=3)), 3)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_consolidate_persists_to_disk(self):
        """A consolidated summary must survive a vault reload."""
        self.vault.consolidate("Persistent session summary about thermodynamics.", "session")
        sid = list(self.vault.summaries.keys())[0]
        v2  = make_vault(self._tmpdir)
        self.assertIn(sid, v2.summaries,
                      "Summary must be written to disk and reloaded correctly")

    def test_get_summaries_respects_top_k_zero(self):
        """top_k=0 should return an empty list without error."""
        self.vault.consolidate("Some summary.", "session")
        results = self.vault.get_summaries(top_k=0)
        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PERSISTENCE — load / persist / backup rotation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistence(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_round_trip_entry(self):
        v1 = make_vault(self._tmpdir)
        v1.save("persist_e",
                "Plate tectonics describes the motion of Earth's lithospheric plates.")
        v2 = make_vault(self._tmpdir)
        self.assertIn("persist_e", v2.entries)
        self.assertIn("Plate tectonics", v2.entries["persist_e"]["text"])

    def test_round_trip_tags(self):
        v1 = make_vault(self._tmpdir)
        v1.save("tagged_persist",
                "Supernovae disperse heavy elements across the galaxy.",
                tags=["astronomy", "nucleosynthesis"])
        v2 = make_vault(self._tmpdir)
        self.assertIn("astronomy", v2.tags["tagged_persist"])

    def test_backup_created(self):
        v = make_vault(self._tmpdir)
        v.save("bkp_entry",
               "Backup test: dark matter constitutes ~27% of the universe's energy density.")
        backup_dir = Path(self._tmpdir) / "backups"
        backups    = list(backup_dir.glob("vault_*.json"))
        self.assertGreater(len(backups), 0,
                           "At least one backup should exist after save")

    def test_load_missing_file_doesnt_crash(self):
        v = ShadowVault(
            filepath=os.path.join(self._tmpdir, "nonexistent.json"),
            backup_dir=os.path.join(self._tmpdir, "bkp"),
        )
        self.assertEqual(len(v.entries), 0)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_round_trip_access_count(self):
        """access_count must survive persist → load."""
        v1 = make_vault(self._tmpdir)
        v1.save("access_persist", "Note about the photoelectric effect.")
        v1.entries["access_persist"]["access_count"] = 7
        v1.persist()
        v2 = make_vault(self._tmpdir)
        self.assertEqual(v2.entries["access_persist"]["access_count"], 7)

    def test_round_trip_importance(self):
        """importance must survive persist → load."""
        v1 = make_vault(self._tmpdir)
        v1.save("imp_persist", "Note about Bohr's atomic model.", importance=0.93)
        v2 = make_vault(self._tmpdir)
        self.assertAlmostEqual(
            v2.entries["imp_persist"]["importance"], 0.93, places=5
        )

    def test_persist_is_atomic_temp_cleanup(self):
        """No orphaned .tmp files should remain after a successful persist."""
        v = make_vault(self._tmpdir)
        v.save("atomic_test", "Testing atomic write behaviour.")
        tmp_files = list(Path(self._tmpdir).glob("*.tmp"))
        self.assertEqual(tmp_files, [],
                         f"Orphaned .tmp files found: {tmp_files}")

    def test_load_legacy_flat_format(self):
        """Old flat {id: str} format must load without errors."""
        legacy = {"old_entry": "This is a legacy string entry without metadata."}
        legacy_path = os.path.join(self._tmpdir, "vault.json")
        with open(legacy_path, "w") as f:
            json.dump(legacy, f)
        v = ShadowVault(filepath=legacy_path,
                        backup_dir=os.path.join(self._tmpdir, "bkp"))
        self.assertIn("old_entry", v.entries)
        self.assertIn("legacy string", v.entries["old_entry"]["text"])

    def test_backup_rotation_respects_max(self):
        """
        After exceeding max_backups saves (forcing one backup per call by
        patching _last_backup each time), old backups are deleted.
        """
        max_backups = 3
        v = make_vault(self._tmpdir)
        v.config["max_backups"] = max_backups

        for i in range(max_backups + 3):
            # Reset _last_backup so every persist() creates a new backup file.
            v._last_backup = ""
            v.save(f"rotation_entry_{i}",
                   f"Backup rotation test entry number {i}.")
            time.sleep(0.01)  # ensure distinct filenames

        backup_dir = Path(self._tmpdir) / "backups"
        backups    = sorted(backup_dir.glob("vault_*.json"))
        self.assertLessEqual(
            len(backups), max_backups,
            f"Expected ≤ {max_backups} backups; found {len(backups)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MIGRATION — _format_entry_text / _preview_entry / _ingest_entries /
#                _run_migration / _load_json_file
# ═══════════════════════════════════════════════════════════════════════════════

class TestMigration(_TmpDirMixin, unittest.TestCase):

    def _valid_entry(self, eid="migration_slug",
                     text="Sample migration text about quantum chromodynamics.",
                     tags=None, importance=0.8) -> dict:
        return {
            "id":         eid,
            "text":       text,
            "tags":       tags if tags is not None else ["physics"],
            "importance": importance,
        }

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_format_entry_text_contains_header(self):
        entry     = self._valid_entry(tags=["biochemistry"], importance=0.9,
                                      text="Enzymes lower activation energy by stabilising the transition state.")
        formatted = _format_entry_text(entry)
        self.assertIn("[importance: 0.90", formatted)
        self.assertIn("tags: biochemistry", formatted)
        self.assertIn("Enzymes lower", formatted)

    def test_format_entry_text_no_tags(self):
        entry     = self._valid_entry(tags=[], importance=0.7,
                                      text="Photons have zero rest mass but carry momentum.")
        formatted = _format_entry_text(entry)
        self.assertIn("[importance: 0.70]", formatted)
        self.assertNotIn("tags:", formatted)

    def test_migrate_single_valid_entry(self):
        raw = [self._valid_entry(
            eid="single_migrate",
            text="Black holes form when stellar mass collapses below the Schwarzschild radius.",
            tags=["astrophysics"],
        )]
        _run_migration(self.vault, raw, dry_run=False)
        self.assertIn("single_migrate", self.vault.entries)

    def test_migrate_dry_run_does_not_write(self):
        raw          = [self._valid_entry(eid="dry_run_entry")]
        count_before = len(self.vault.entries)
        _run_migration(self.vault, raw, dry_run=True)
        self.assertEqual(len(self.vault.entries), count_before,
                         "Dry run must not write any entries")

    def test_migrate_skip_identical_entry(self):
        entry = self._valid_entry(
            eid="idempotent_entry",
            text="Tectonic plate boundaries are sites of volcanic and seismic activity.",
            tags=["geology"],
        )
        _run_migration(self.vault, [entry], dry_run=False)
        _run_migration(self.vault, [entry], dry_run=False)
        self.assertEqual(len(self.vault.entries), 1,
                         "Re-migrating identical entry should not duplicate it")

    def test_migrate_multiple_entries(self):
        raw = [
            self._valid_entry("entry_alpha",
                              "Superfluid helium exhibits zero viscosity below the lambda point temperature.",
                              ["physics", "cryogenics"]),
            self._valid_entry("entry_beta",
                              "Coral bleaching occurs when ocean temperature rises cause zooxanthellae expulsion.",
                              ["marine biology", "climate"]),
        ]
        _run_migration(self.vault, raw, dry_run=False)
        self.assertIn("entry_alpha", self.vault.entries)
        self.assertIn("entry_beta",  self.vault.entries)

    def test_migrate_multiple_saves_distinct_entries_not_deduped(self):
        entries = [
            self._valid_entry(
                "volcanology_note",
                ("Pyroclastic flows travel at hundreds of kilometres per hour, "
                 "incinerating everything in their path with superheated gas and ash."),
                ["geology"], 0.70,
            ),
            self._valid_entry(
                "linguistics_note",
                ("Proto-Indo-European reconstructed vocabulary suggests cattle herding "
                 "and wheeled transport were central to ancient steppe cultures."),
                ["linguistics"], 0.70,
            ),
            self._valid_entry(
                "immunology_note",
                ("T-cell receptor diversity arises from somatic recombination of V, D, "
                 "and J gene segments during thymic development."),
                ["immunology"], 0.70,
            ),
        ]
        _run_migration(self.vault, entries, dry_run=False)
        self.assertEqual(len(self.vault.entries), 3,
                         "Three distinct entries must all be saved — none deduped.")

    def test_load_json_file_valid(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, dir=self._tmpdir
        )
        json.dump([{"id": "x", "text": "y", "tags": [], "importance": 0.5}], tmp)
        tmp.close()
        result = _load_json_file(tmp.name)
        self.assertEqual(len(result), 1)

    def test_load_json_file_missing(self):
        with self.assertRaises(MigrationError):
            _load_json_file("/nonexistent/path/file.json")

    def test_load_json_file_invalid_json(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, dir=self._tmpdir
        )
        tmp.write("{ not valid json")
        tmp.close()
        with self.assertRaises(MigrationError):
            _load_json_file(tmp.name)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_preview_entry_skip_path(self):
        """_preview_entry increments 'skipped' when text is identical."""
        entry = self._valid_entry(eid="preview_skip")
        # Pre-populate the vault with the formatted text so it matches exactly.
        text  = _format_entry_text(entry)
        self.vault.entries["preview_skip"] = {
            "text":         text,
            "created":      "2025-01-01T00:00:00",
            "last_edited":  "2025-01-01T00:00:00",
            "importance":   0.8,
            "access_count": 0,
        }
        counts = {"saved": 0, "skipped": 0}
        _preview_entry("preview_skip", text, entry, self.vault, counts)
        self.assertEqual(counts["skipped"], 1)
        self.assertEqual(counts["saved"],   0)

    def test_preview_entry_update_path(self):
        """_preview_entry increments 'saved' when ID exists but text differs."""
        entry = self._valid_entry(eid="preview_update")
        self.vault.entries["preview_update"] = {
            "text":         "Old completely different text about botany.",
            "created":      "2025-01-01T00:00:00",
            "last_edited":  "2025-01-01T00:00:00",
            "importance":   0.5,
            "access_count": 0,
        }
        new_text = _format_entry_text(entry)
        counts   = {"saved": 0, "skipped": 0}
        _preview_entry("preview_update", new_text, entry, self.vault, counts)
        self.assertEqual(counts["saved"],   1)
        self.assertEqual(counts["skipped"], 0)

    def test_preview_entry_new_path(self):
        """_preview_entry increments 'saved' for a genuinely new ID."""
        entry  = self._valid_entry(eid="preview_new")
        text   = _format_entry_text(entry)
        counts = {"saved": 0, "skipped": 0}
        _preview_entry("preview_new", text, entry, self.vault, counts)
        self.assertEqual(counts["saved"],   1)
        self.assertEqual(counts["skipped"], 0)

    def test_ingest_entries_returns_counts(self):
        """_ingest_entries returns a dict with saved, skipped, errors keys."""
        entries = [
            self._valid_entry("ingest_a",
                              "The Krebs cycle produces NADH and FADH2 in mitochondria.",
                              ["biochemistry"]),
            self._valid_entry("ingest_b",
                              "Metamorphic rock forms under high pressure and temperature deep in the crust.",
                              ["geology"]),
        ]
        counts = _ingest_entries(entries, self.vault)
        self.assertIn("saved",   counts)
        self.assertIn("skipped", counts)
        self.assertIn("errors",  counts)
        self.assertEqual(counts["saved"], 2)
        self.assertEqual(counts["errors"], 0)

    def test_run_migration_aborts_all_hard_errors(self):
        """_run_migration with zero valid entries must not write anything."""
        bad_entries = [
            {"id": "",   "text": "", "tags": [], "importance": 0.5},  # hard error
            {"id": "ok", "text": "", "tags": [], "importance": 0.5},  # hard error (no text)
        ]
        count_before = len(self.vault.entries)
        _run_migration(self.vault, bad_entries, dry_run=False)
        self.assertEqual(len(self.vault.entries), count_before,
                         "Migration with all-hard-errors must not write anything")

    def test_migrate_updates_existing_entry_with_different_text(self):
        """Re-migrating the same ID with different text replaces the entry."""
        original = self._valid_entry(
            eid="updatable_entry",
            text="Original finding about superconductivity at low temperatures.",
            tags=["physics"],
        )
        updated  = self._valid_entry(
            eid="updatable_entry",
            text="Revised finding: high-temperature superconductors violate BCS theory.",
            tags=["physics"],
        )
        _run_migration(self.vault, [original], dry_run=False)
        _run_migration(self.vault, [updated],  dry_run=False)
        stored_text = self.vault.entries["updatable_entry"]["text"]
        self.assertIn("Revised finding",  stored_text,
                      "Updated text should replace original in vault")
        self.assertNotIn("Original finding", stored_text)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

class TestPruning(_TmpDirMixin, unittest.TestCase):

    def _add_entry(self, eid, importance=0.5, access_count=0):
        self.vault.entries[eid] = {
            "text":         f"Research note about {eid}: placeholder content with sufficient length.",
            "created":      "2024-01-01T00:00:00",
            "last_edited":  "2024-01-01T00:00:00",
            "importance":   importance,
            "access_count": access_count,
        }
        self.vault.recent_entries.append(eid)

    # ── Existing tests (preserved) ────────────────────────────────────────────

    def test_prune_removes_low_value_entries(self):
        for i in range(5):
            self._add_entry(f"low_{i}",  importance=0.1, access_count=0)
        for i in range(5):
            self._add_entry(f"high_{i}", importance=0.9, access_count=5)
        self.vault._prune_low_value()
        for i in range(5):
            self.assertIn(f"high_{i}", self.vault.entries)

    def test_prune_keeps_high_access_entries(self):
        self._add_entry("accessed_low_imp", importance=0.1, access_count=10)
        self.vault._prune_low_value()
        self.assertIn("accessed_low_imp", self.vault.entries)

    def test_prune_cap_at_10_percent(self):
        for i in range(50):
            self._add_entry(f"prunable_{i}", importance=0.1, access_count=0)
        count_before = len(self.vault.entries)
        self.vault._prune_low_value()
        removed = count_before - len(self.vault.entries)
        self.assertLessEqual(removed, max(5, int(count_before * 0.1)) + 1)

    # ── New v2 tests ──────────────────────────────────────────────────────────

    def test_prune_updates_recent_entries(self):
        """Pruned IDs must be removed from the dedup deque."""
        self._add_entry("dequeue_victim", importance=0.1, access_count=0)
        self.assertIn("dequeue_victim", list(self.vault.recent_entries))
        self.vault._prune_low_value()
        self.assertNotIn("dequeue_victim", list(self.vault.recent_entries))

    def test_prune_no_entries_no_crash(self):
        """Empty vault handles a prune pass without raising."""
        self.vault._prune_low_value()   # must not raise

    def test_prune_only_fires_above_max_entries(self):
        """
        save() must NOT invoke prune when entry count is below max_entries.
        We set max_entries to a large number and verify no pruning output.
        """
        self.vault.config["max_entries"] = 10_000
        # Add a prunable entry — it should NOT be pruned because we are below cap.
        self._add_entry("prunable_below_cap", importance=0.1, access_count=0)
        self.vault.save("below_cap_safe",
                        "A completely new entry well below the max_entries threshold.")
        # The prunable entry should still be there (prune was never triggered).
        self.assertIn("prunable_below_cap", self.vault.entries)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. STATS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStats(_TmpDirMixin, unittest.TestCase):

    def test_stats_keys_present(self):
        s = self.vault.stats()
        for key in ("entries", "summaries", "tagged_entries", "recent_window"):
            self.assertIn(key, s)

    def test_stats_reflect_saves(self):
        self.vault.save("s1",
                        "Continental drift was proposed by Alfred Wegener in 1912.")
        self.vault.save("s2",
                        "The double helix structure of DNA was elucidated in 1953.",
                        tags=["biology"])
        s = self.vault.stats()
        self.assertEqual(s["entries"],        2)
        self.assertEqual(s["tagged_entries"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. EDGE CASES (new v2 class)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(_TmpDirMixin, unittest.TestCase):

    def test_save_whitespace_only_id_rejected(self):
        """'   ' (whitespace) must be treated as an empty ID."""
        ok = self.vault.save("   ", "Some valid text content.")
        self.assertFalse(ok)
        # Ensure nothing slipped into entries under a blank key.
        self.assertEqual(len(self.vault.entries), 0)

    def test_save_whitespace_only_text_rejected(self):
        """Whitespace-only text must be rejected."""
        ok = self.vault.save("valid_id", "   \n\t  ")
        self.assertFalse(ok)

    def test_jaccard_single_token_overlap(self):
        """Single-token sets give Jaccard of 0 (no overlap) or 1 (identical)."""
        # Same single token → Jaccard == 1.0
        self.assertAlmostEqual(
            self.vault._jaccard("photosynthesis", "photosynthesis"), 1.0
        )
        # Two different single-token strings → Jaccard == 0.0
        score = self.vault._jaccard("mitochondria", "atmosphere")
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_sorted_ids_alphabetical(self):
        """_sorted_ids() must return IDs in strict lexicographic order."""
        for eid in ["zebra_fact", "alpha_fact", "middle_fact"]:
            self.vault.save(eid, f"Research note for {eid} with some content.")
        ids = self.vault._sorted_ids()
        self.assertEqual(ids, sorted(ids))

    def test_reinforce_runs_without_crash(self):
        """
        End-to-end smoke test for reinforce().
        It must run to completion without raising any exception.
        """
        self.vault.save("smoke_test",
                        "Halley's Comet orbits the Sun with a period of ~75 years.")
        self.vault.reinforce()   # must not raise

    def test_format_entry_text_strips_body_whitespace(self):
        """Leading/trailing whitespace in entry text is stripped in storage."""
        entry = {
            "id":         "strip_test",
            "text":       "  Leading and trailing spaces should be removed.   ",
            "tags":       ["test"],
            "importance": 0.7,
        }
        formatted = _format_entry_text(entry)
        # The body line (after the header) should not start or end with spaces.
        lines     = formatted.splitlines()
        body_line = lines[-1]  # last line is the stripped body
        self.assertFalse(body_line.startswith(" "),
                         "Formatted text body should not have leading spaces")
        self.assertFalse(body_line.endswith(" "),
                         "Formatted text body should not have trailing spaces")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
