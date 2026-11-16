import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from common import canonical,gold,key,messages,parse,FirstTokenTimer,SMALL,LARGE,SHOTS
from prepare import build_plan
from analyze import load_complete,select_shots,METRICS


def fixture_plan():
    train=[dict(question=f'Training problem {i}',answer=f'work\n#### {i}') for i in range(64)]
    test=[dict(question=f'Test problem {i}',answer=f'work\n#### {i}') for i in range(12)]
    return build_plan(train,test,2,2)


class ICLTests(unittest.TestCase):
    def test_numeric_scoring(self):
        self.assertEqual(gold('work\n#### 1,200'), '1200')
        self.assertEqual(parse('calculation FINAL: 16'), '16')
        self.assertEqual(parse('FINAL: <answer> 1200'), '1200')
        self.assertEqual(parse('FINAL: $31'), '31')
        self.assertIsNone(parse('FINAL: $31',False))
        self.assertIsNone(parse('FINAL: <answer>'))
        self.assertIsNone(parse('FINAL: 3.333...'))
        self.assertIsNone(parse('I got 42'))
        self.assertIsNone(parse('<think>\nFINAL: 42'))
        self.assertIsNone(canonical('NaN'))

    def test_nonoverlap_and_nested_examples(self):
        plan=fixture_plan()
        sets=[set(r['ids']) for r in plan['demonstrations']]
        self.assertEqual(len(set.union(*sets)),48)
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertEqual(len({r['id'] for r in plan['items']}),4)
        examples=plan['demonstrations'][0]['items']
        two=messages('held-out',examples[:2]);four=messages('held-out',examples[:4])
        self.assertEqual(two[:-1],four[:len(two)-1])
        self.assertEqual(two[-1],{'role':'user','content':'held-out'})
        self.assertEqual(len(messages('held-out',[])),2)

    def test_exclusions(self):
        train=[dict(question=f'Train {i}',answer='#### 1') for i in range(64)]
        test=[dict(question=f'Test {i}',answer='#### 1') for i in range(6)]
        excluded={key(test[0]['question'])}
        plan=build_plan(train,test,2,2,excluded)
        self.assertNotIn(next(iter(excluded)),{r['id'] for r in plan['items']})

    def test_first_token_excludes_prompt(self):
        calls=[]
        timer=FirstTokenTimer(lambda:calls.append('sync'),lambda:123.0)
        timer.put('prompt')
        self.assertIsNone(timer.first_time)
        timer.put('token1');timer.put('token2')
        self.assertEqual(timer.first_time,123.0)
        self.assertEqual(calls,['sync'])

    def test_validation_selection(self):
        rows=[dict(shots=0,correct=.6,gpu_joules=100),dict(shots=4,correct=.8,gpu_joules=150)]
        self.assertEqual(select_shots(rows,.81,.02),4)
        self.assertIsNone(select_shots(rows,.95,.02))

    def test_complete_run_and_analysis(self):
        plan=fixture_plan()
        rows=[]
        for item in plan['items']:
            for rep,example in enumerate(plan['demonstrations']):
                for model,shots in [(SMALL,SHOTS),(LARGE,[0])]:
                    for shot in shots:
                        values=dict(correct=(model==LARGE or shot>=4),gpu_joules=100+shot,
                            request_seconds=1.,ttft_seconds=.1,input_tokens=20+shot*50,
                            output_tokens=20,parse_failure=False,strict_parse_failure=False,
                            hit_token_limit=False)
                        rows.append(dict(id=item['id'],split=item['split'],model=model,shots=shot,
                            repetition=rep,example_seed=example['seed'],**values))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(args=dict(smoke=False))))
            (root/'plan.json').write_text(json.dumps(plan))
            raw='\n'.join(json.dumps(r) for r in rows)+'\n'
            (root/'measurements.jsonl').write_text(raw)
            self.assertEqual(len(load_complete(root)[2]),72)
            subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),
                            directory,'--bootstrap','50'],check=True,capture_output=True,text=True)
            report=json.loads((root/'analysis.json').read_text())
            self.assertEqual(report['selection']['near_large'],4)
            self.assertTrue((root/'paired_differences.csv').is_file())
            (root/'measurements.jsonl').write_text('\n'.join(raw.splitlines()[:-1]))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                load_complete(root)


if __name__=='__main__':
    unittest.main()
