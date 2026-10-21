import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from prepare import split_rows
from collect import parse, prompt_for


class ScreenTests(unittest.TestCase):
    def test_duplicate_grouping(self):
        rows=[dict(domain='math',question=f'question {i}',target='1') for i in range(6)]
        rows.append(dict(rows[0]))
        result=split_rows(rows,[2,2,2],42)
        self.assertEqual(len({r['id'] for r in result}),6)
        self.assertEqual([r['split'] for r in result].count('test'),2)
        with self.assertRaises(ValueError):
            split_rows(rows,[3,2,2],42)

    def test_scoring(self):
        self.assertEqual(parse('work\nFINAL: 1,200',{'domain':'math'}),'1.2E+3')
        self.assertEqual(parse('FINAL: b',{'domain':'science','allowed':['A','B','C']}),'B')
        self.assertIsNone(parse('FINAL: 7',{'domain':'science','allowed':['A','B']}))
        self.assertEqual(parse('FINAL: True',{'domain':'logic','allowed':['true','false']}),'true')
        self.assertIsNone(parse('<think>\nFINAL: 42',{'domain':'math'}))
        item=dict(domain='math',question='question',target='secret')
        self.assertNotIn('secret',str(prompt_for(item)))

    def test_policy_selection(self):
        from analyze import select_threshold,oracle_choice
        candidates=[dict(threshold=2.,accuracy=.8,gpu_joules=10),
                    dict(threshold=.5,accuracy=.79,gpu_joules=5),
                    dict(threshold=0.,accuracy=.5,gpu_joules=2)]
        self.assertEqual(select_threshold(candidates,.02),.5)
        small=dict(accuracy=0,gpu_joules=1)
        large=dict(accuracy=1,gpu_joules=10)
        self.assertIs(oracle_choice(small,large),large)
        self.assertEqual(oracle_choice(dict(accuracy=1,gpu_joules=1),large)['gpu_joules'],1)

    def test_end_to_end_analysis(self):
        items=[]
        rows=[]
        for domain in ['math','science','logic']:
            for split in ['train','validation','test']:
                for i in range(6):
                    key=f'{domain}-{split}-{i}'
                    items.append(dict(id=key,domain=domain,split=split,
                        question=f'{domain} question {i} '+('easy simple' if i%2 else 'hard difficult')))
                    for model in ['Qwen/Qwen3-1.7B','Qwen/Qwen3-8B']:
                        small='1.7B' in model
                        rows.append(dict(id=key,model=model,repeat=0,correct=bool(i%2) if small else True,
                            gpu_joules=2 if small else 10,seconds=.1 if small else .3,
                            parse_failure=False,hit_token_limit=False))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'manifest.json').write_text(json.dumps(dict(args=dict(smoke=False,repeats=1))))
            (path/'data.json').write_text(json.dumps(dict(items=items)))
            (path/'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),directory],check=True)
            report=json.loads((path/'routing-report.json').read_text())
            self.assertIn('selected_threshold',report)
            self.assertEqual(len((path/'test-routes.jsonl').read_text().splitlines()),18)
            self.assertEqual(len((path/'routing-summary.csv').read_text().splitlines()),25)


if __name__=='__main__':
    unittest.main()
