"""Recheck a smoke run with the current parser; preserve original measurements."""
import collections
import json
import sys
from pathlib import Path
from collect import parse, parse_strict

run=Path(sys.argv[1])
manifest=json.loads((run/'manifest.json').read_text())
if not manifest['args']['smoke']:
    raise SystemExit('This diagnostic is restricted to smoke runs.')
items={r['id']:r for r in json.loads((run/'data.json').read_text())['items']}
groups=collections.defaultdict(list)
rescored=[]
for line in (run/'measurements.jsonl').read_text().splitlines():
    row=json.loads(line)
    item=items[row['id']]
    prediction=parse(row['response'],item)
    row.update(prediction=prediction,correct=prediction==item['target'],
               parse_failure=prediction is None,
               strict_parse_failure=parse_strict(row['response'],item) is None)
    groups[(row['model'],row['domain'])].append(row)
    rescored.append(row)
for (model,domain),rows in sorted(groups.items()):
    print(model,domain,'n=',len(rows),'correct=',sum(r['correct'] for r in rows),
          'parse failures=',sum(r['parse_failure'] for r in rows),
          'strict format failures=',sum(r['strict_parse_failure'] for r in rows),
          'token limits=',sum(r['hit_token_limit'] for r in rows))
    for r in rows:
        if r['parse_failure']:
            print('UNPARSED:',r['id'],r['response'])
with (run/'measurements-rescored.jsonl').open('w') as f:
    for row in rescored:
        f.write(json.dumps(row)+'\n')
print('Original measurements unchanged; wrote measurements-rescored.jsonl')
