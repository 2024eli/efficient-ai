"""CPU-only checks for scoring and leakage-relevant prompt structure."""
import tempfile
import unittest
from pathlib import Path
from benchmark import answer, messages, summarize


class BenchmarkTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(answer('work: 7\nFINAL: 1,200.00'), '1.2E+3')
        self.assertEqual(answer('work\n#### 1200', gold=True), '1.2E+3')
        self.assertEqual(answer('FINAL: -2.5'), '-2.5')
        self.assertIsNone(answer('I tried 2 then 42'))
        self.assertIsNone(answer('<think>FINAL: 4</think>unfinished 4'))
        self.assertEqual(answer('FINAL: 2\nFINAL: 3'), '3')

    def test_prompt(self):
        examples = [{'question': 'training question', 'answer': 'work\n#### 2'}]
        prompt = messages('held-out question', examples)
        self.assertEqual([x['role'] for x in prompt], ['system','user','assistant','user'])
        self.assertEqual(prompt[-1]['content'], 'held-out question')
        self.assertEqual(prompt[2]['content'], 'work\nFINAL: 2')
        self.assertEqual(len(messages('held-out question', [])), 2)

    def test_summary_counts_invalid_answers_as_wrong(self):
        common = dict(model='model', shots=0, test_id=1, generation_seconds=2,
                      gpu_joules=10, input_tokens=20, output_tokens=5,
                      hit_token_limit=False)
        rows = [dict(common, correct=True, parse_failure=False),
                dict(common, correct=False, parse_failure=True)]
        with tempfile.TemporaryDirectory() as directory:
            summarize(rows, Path(directory))
            import csv
            with (Path(directory)/'summary.csv').open() as handle:
                result = list(csv.DictReader(handle))[0]
            self.assertEqual(float(result['accuracy']), .5)
            self.assertEqual(float(result['gpu_joules_per_correct']), 20)
            self.assertEqual(int(result['unique_questions']), 1)


if __name__ == '__main__':
    unittest.main()
