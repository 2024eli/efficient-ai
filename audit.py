"""Print format and truncation audit for a smoke or full measurement run."""
import argparse
import collections
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('run')
a=p.parse_args()
groups=collections.defaultdict(list)
for line in (Path(a.run)/'measurements.jsonl').read_text().splitlines():
    r=json.loads(line)
    groups[(r['model'],r['domain'])].append(r)
for (model,domain),rows in sorted(groups.items()):
    print(model,domain,'n=',len(rows),'correct=',sum(r['correct'] for r in rows),
          'parse failures=',sum(r['parse_failure'] for r in rows),
          'token limits=',sum(r['hit_token_limit'] for r in rows))
    for r in rows:
        if r['parse_failure'] or r['hit_token_limit']:
            print(r['id'],r['response'])
