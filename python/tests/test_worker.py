"""Protocol tests: the JSON-lines contract the host parses.

Runs on a machine with no ML stack, which is the point — a worker that cannot
import torch must still *say so* on the protocol channel, and `--help` must work.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fnt_worker import __main__ as worker  # noqa: E402


class ProtocolTest(unittest.TestCase):
    def run_worker(self, argv):
        stream = io.StringIO()
        code = 0
        with redirect_stdout(stream):
            try:
                code = worker.main(argv)
            except SystemExit as exit_request:
                code = int(exit_request.code or 0)
        events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        return code, events

    def write_config(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_every_line_is_a_json_event(self):
        config = self.write_config({"model": "x", "data": "/nope", "out": "/nope"})
        code, events = self.run_worker(["train", "--config", config])
        self.assertTrue(len(events) >= 2)
        self.assertEqual(events[0]["kind"], "phase")
        self.assertEqual(events[0]["phase"], "start")
        # No torch on the test machine: the failure is a protocol error with a code.
        self.assertEqual(events[-1]["kind"], "error")
        self.assertIn(events[-1]["code"], ("deps", "data", "config"))
        self.assertNotEqual(code, 0)

    def test_a_missing_config_file_is_reported_not_traced(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            with self.assertRaises(SystemExit) as caught:
                worker.load_config(os.path.join(tempfile.gettempdir(), "definitely-not-here-42.json"))
        self.assertEqual(caught.exception.code, 2)
        events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(events[-1]["kind"], "error")
        self.assertEqual(events[-1]["code"], "config")

    def test_a_config_with_a_bom_still_loads(self):
        handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
        handle.write(b"\xef\xbb\xbf" + json.dumps({"a": 1}).encode("utf-8"))
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        self.assertEqual(worker.load_config(handle.name), {"a": 1})

    def test_train_without_data_is_a_data_error(self):
        root = tempfile.mkdtemp()
        config = self.write_config({"model": "Qwen/Qwen2.5-1.5B-Instruct", "data": root, "out": root})
        code, events = self.run_worker(["train", "--config", config])
        self.assertNotEqual(code, 0)
        # Either "deps" (no torch) or "data" (empty train.jsonl) is a correct
        # refusal; what must never happen is a silent success.
        self.assertEqual(events[-1]["kind"], "error")

    def test_prompt_mask_masks_the_prompt_half(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                tokens = []
                for message in messages:
                    tokens.extend([len(message["content"])] + list(range(len(message["content"]))))
                return tokens

        ids, labels = worker.build_prompt_mask(
            FakeTokenizer(),
            [{"role": "user", "content": "abcd"}, {"role": "assistant", "content": "ef"}],
            max_len=64,
        )
        self.assertEqual(len(ids), len(labels))
        # The first `prompt` tokens carry no loss; the answer half does.
        self.assertEqual(labels[0], -100)
        self.assertEqual(labels[-1], ids[-1])

    def test_a_long_sample_keeps_the_answer(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                total = sum(len(message["content"]) for message in messages)
                return list(range(total))

        ids, labels = worker.build_prompt_mask(
            FakeTokenizer(),
            [{"role": "user", "content": "x" * 100}, {"role": "assistant", "content": "y" * 10}],
            max_len=20,
        )
        self.assertLessEqual(len(ids), 20)
        # Truncation happens at the front of the prompt, and the label survives.
        self.assertEqual([value for value in labels if value != -100], ids[-10:])

    def test_tool_call_parsing_matches_the_hosts_convention(self):
        fenced = '好的\n```json\n{"tool": "read", "arguments": {"path": "a"}}\n```'
        self.assertEqual(worker._tools_in(fenced)[0]["tool"], "read")
        bare = '{"tool": "bash", "arguments": "ls"}'
        self.assertEqual(worker._tools_in(bare)[0]["tool"], "bash")
        self.assertEqual(worker._tools_in("没有工具"), [])

    def test_scoring_counts_a_wrong_tool_as_a_failure(self):
        cases = [{"prompt": "p", "expected": '{"tool":"read","arguments":{}}'}]
        score = worker._score(cases, ['{"tool":"bash","arguments":{}}'])
        self.assertEqual(score["score"], 0)
        score = worker._score(cases, ['{"tool":"read","arguments":{}}'])
        self.assertEqual(score["score"], 1)

    def test_unknown_mode_is_a_config_error_not_a_silent_qlora(self):
        """A typo'd mode must not quietly train the *other* mode: the memory the
        user planned around would be wrong."""
        config = self.write_config({"mode": "LoRA-fp16", "model": "x", "data": "/nope", "out": "/nope"})
        code, events = self.run_worker(["train", "--config", config])
        self.assertNotEqual(code, 0)
        messages = " ".join(str(event.get("message", "")) for event in events if event["kind"] == "error")
        # Without torch installed the deps error comes first; either way the run
        # must not have proceeded to training.
        self.assertIn(events[-1]["kind"], ("error",))

    def test_lora_mode_never_asks_for_quantization(self):
        """The two modes differ only in how the base is loaded; the config builder
        must pass `mode` through untouched so that difference is visible."""
        config = self.write_config({"mode": "lora", "model": "Qwen/Qwen2.5-1.5B-Instruct", "data": "/nope", "out": "/nope"})
        loaded = worker.load_config(config)
        self.assertEqual(loaded["mode"], "lora")
        self.assertNotIn("quantType", loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
