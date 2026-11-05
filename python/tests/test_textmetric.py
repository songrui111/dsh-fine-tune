"""Tests for the text metric used on answers that have no exact target.

The calibration anchors are the point of this file: identical answers must score
1.0 and unrelated fluent text must score 0.0. Those four numbers are what the old
"non-empty passes" metric failed, and they are what makes the panel's percentage
mean anything.

Strings are written as escapes on purpose. This suite originally failed with every
Chinese literal arriving as '?', which looked like a metric bug and was actually a
transport encoding problem in the test harness — the escapes make the test
independent of how the file is saved or piped.
"""

import unittest

from fnt_worker.textmetric import content_tokens, score_text_answer, score_text_split, shape_of

REFERENCE = "\u897f\u5357\u5927\u5b66\u7684\u6821\u8bad\u662f\u542b\u5f18\u5149\u5927\u7ee7\u5f80\u5f00\u6765\u3002"  # 西南大学的校训是含弘光大继往开来。
UNRELATED = "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u6211\u6253\u7b97\u51fa\u95e8\u6563\u6b65"  # 今天天气很好我打算出门散步
PARTIAL = "\u6821\u8bad\u662f\u542b\u5f18\u5149\u5927"  # 校训是含弘光大


class CalibrationAnchors(unittest.TestCase):
    """What right and wrong look like, so a regression is caught immediately."""

    def test_identical_scores_one(self):
        self.assertEqual(score_text_answer(REFERENCE, REFERENCE)["f1"], 1.0)

    def test_unrelated_scores_zero(self):
        self.assertEqual(score_text_answer(UNRELATED, REFERENCE)["f1"], 0.0)

    def test_empty_scores_zero(self):
        self.assertEqual(score_text_answer("", REFERENCE)["f1"], 0.0)
        self.assertEqual(score_text_answer("   \n ", REFERENCE)["f1"], 0.0)

    def test_a_truncated_reference_is_still_on_topic(self):
        # Everything it says is in the reference, so coverage is saturated.
        self.assertGreater(score_text_answer(PARTIAL, REFERENCE)["f1"], 0.9)


class Structure(unittest.TestCase):
    def test_markdown_shape_is_read(self):
        shape = shape_of("## \u7ed3\u8bba\n\n- a\n- b\n\n```json\n{}\n```\n")
        self.assertEqual(shape["headings"], 1)
        self.assertEqual(shape["bullets"], 2)
        self.assertEqual(shape["code_fence"], 2)

    def test_same_words_wrong_form_scores_lower(self):
        reference = "## \u7ed3\u8bba\n\n- \u7b2c\u4e00\u70b9\n- \u7b2c\u4e8c\u70b9\n"
        with_form = score_text_answer("## \u7ed3\u8bba\n\n- \u7b2c\u4e00\u70b9\n- \u7b2c\u4e8c\u70b9\n", reference)
        without = score_text_answer("\u7ed3\u8bba \u7b2c\u4e00\u70b9 \u7b2c\u4e8c\u70b9", reference)
        self.assertGreater(with_form["shape"], without["shape"])

    def test_padding_lowers_precision(self):
        faithful = score_text_answer(REFERENCE, REFERENCE)
        padded = score_text_answer(REFERENCE + "\u53e6\u5916\u6211\u8fd8\u559c\u6b22\u4e2d\u56fd\u83dc" * 5, REFERENCE)
        self.assertLess(padded["precision"], faithful["precision"])


class Tokenising(unittest.TestCase):
    def test_latin_and_cjk(self):
        tokens = content_tokens("PyTorch \u8bad\u7ec3 model")
        self.assertIn("pytorch", tokens)
        self.assertIn("model", tokens)
        # CJK becomes bigrams, so a single character alone produces none.
        self.assertIn("\u8bad\u7ec3", tokens)


class SplitAggregation(unittest.TestCase):
    def test_keeps_per_case_detail(self):
        cases = [{"expected": REFERENCE}, {"expected": REFERENCE}]
        split = score_text_split(cases, [REFERENCE, UNRELATED])
        self.assertEqual(split["cases"], 2)
        self.assertEqual(split["perCase"][0]["f1"], 1.0)
        self.assertEqual(split["perCase"][1]["f1"], 0.0)
        # Half on topic; the aggregate must not hide that behind a single number.
        self.assertEqual(split["onTopicRate"], 0.5)

    def test_missing_outputs_are_treated_as_empty(self):
        split = score_text_split([{"expected": REFERENCE}], [])
        self.assertEqual(split["cases"], 1)
        self.assertEqual(split["perCase"][0]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
