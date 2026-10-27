import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from lora_energy import boxed_answer


class LoraTests(unittest.TestCase):
    def test_boxed_parsing(self):
        self.assertEqual(boxed_answer('so \\boxed{18}'), '18')
        self.assertEqual(boxed_answer('\\boxed{1,200}'), '1.2E+3')
        self.assertEqual(boxed_answer('\\boxed{\\$18.00}.'), '18')
        self.assertEqual(boxed_answer('\\boxed{18 \\text{ dollars}}'), '18')
        self.assertEqual(boxed_answer('\\boxed{5} then \\boxed{7}'), '7')
        self.assertEqual(boxed_answer('\\boxed{-3.5}'), '-3.5')
        self.assertIsNone(boxed_answer('the answer is 18'))
        self.assertIsNone(boxed_answer('\\boxed{3, 4}'))
        self.assertIsNone(boxed_answer('\\boxed{\\frac{1}{2}}'))
        self.assertIsNone(boxed_answer('\\boxed{18'))
        self.assertIsNone(boxed_answer('<think>\\boxed{18}'))
        # Stray close tag (observed from the adapter with thinking off) must not hide the answer.
        self.assertEqual(boxed_answer('so \\boxed{72}\n</think>\n\nDone.'), '72')
        self.assertEqual(boxed_answer('work\n</think>\n\nTherefore \\boxed{72}.'), '72')
        self.assertEqual(boxed_answer('<think>\\boxed{1}</think>\\boxed{2}'), '2')

    def test_end_to_end_analysis(self):
        rows = []
        for i in range(12):
            for repeat in range(2):
                for phase in ['natural', 'fixed', 'forward']:
                    for c in ['base', 'lora'] + (['8b'] if phase != 'forward' else []):
                        joules = {'base': 10, 'lora': 11, '8b': 40}[c] + repeat * .1
                        rows.append(dict(phase=phase, condition=c, repeat=repeat, id=i, seconds=1.,
                            gpu_joules=joules, output_tokens=100, passes=20, response_sha256='x',
                            correct=(c != 'base' or i % 3 == 0) if phase == 'natural' else None,
                            parse_failure=False, hit_token_limit=False))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'manifest.json').write_text(json.dumps(dict(complete=True, args=dict(fixed_tokens=256))))
            (path / 'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            subprocess.run([sys.executable, str(Path(__file__).with_name('lora_analyze.py')), directory],
                           check=True, stdout=subprocess.DEVNULL)
            report = json.loads((path / 'lora-report.json').read_text())
            self.assertAlmostEqual(report['fixed']['energy_overhead_lora_vs_base']['percent'], 1 / 10.05 * 100, 3)
            groups = {r['group']: r['questions'] for r in report['natural']['table_b']}
            self.assertEqual(groups['both correct'], 4)
            self.assertEqual(groups['base wrong, LoRA correct'], 8)
            policy = {r['policy']: r for r in report['natural']['policies']}
            # Oracle uses base on the 4 questions base solves: saves 4/12 of the 1 J overhead.
            self.assertEqual(policy['oracle base/lora']['accuracy'], 1)
            self.assertAlmostEqual(policy['oracle min-energy matching always-lora accuracy']['mean_gpu_joules'],
                                   (4 * 10.05 + 8 * 11.05) / 12)
            self.assertEqual(len((path / 'per-question.csv').read_text().splitlines()), 13)


if __name__ == '__main__':
    unittest.main()
