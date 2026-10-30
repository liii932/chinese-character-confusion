"""Focused checks for full-pool scope, stable labels, and explicit relation provenance."""
import unittest

from pilot.render_samples import checked_input
from training.prepare_round2 import make_table


class FullPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = make_table()
        cls.original, _ = checked_input()

    def test_all_source_characters_once_and_old_ids_retained(self):
        rows = self.table["characters"]
        self.assertEqual(len(rows), 1140)
        self.assertEqual([r["class_id"] for r in rows], list(range(1140)))
        self.assertEqual([r["char"] for r in rows[:81]], [r["char"] for r in self.original["characters"]])
        self.assertEqual({r["char"] for r in rows},
            {c for g in self.table["source_groups"] for c in g["characters"]})
        self.assertEqual(len({r["char"] for r in rows}), len(rows))
        # Both overlapping groups must contribute provenance, never another class.
        by_char = {r["char"]: r for r in rows}
        self.assertEqual(len(by_char["免"]["source_group_ids"]), 2)
        self.assertIn("丏", by_char)
        self.assertIn("尢", by_char)

    def test_no_clique_or_transitive_expansion(self):
        pairs = {frozenset((p["a"], p["b"])) for p in self.table["candidate_pairs"]}
        self.assertEqual(len(pairs), 251)
        explicit = {frozenset(g["characters"]) for g in self.table["source_groups"] if len(g["characters"]) == 2}
        self.assertEqual(pairs, explicit)
        self.assertIn(frozenset(("兔", "免")), pairs)
        self.assertIn(frozenset(("免", "奂")), pairs)
        self.assertNotIn(frozenset(("兔", "奂")), pairs)
        self.assertEqual(self.table["pairs"], self.original["pairs"])
        # The reversed repeated source group is represented once, with both sources.
        repeated = next(p for p in self.table["candidate_pairs"] if {p["a"], p["b"]} == {"嵌", "崭"})
        self.assertEqual(len(repeated["source_group_ids"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
