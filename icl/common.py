"""Dependency-light data, scoring and timing utilities."""
import hashlib
import re
import time
from decimal import Decimal, InvalidOperation

SMALL = 'Qwen/Qwen3-1.7B'
LARGE = 'Qwen/Qwen3-8B'
SHOTS = [0, 2, 4, 8, 16]
SYSTEM = ('Solve the math problem. Show a concise calculation. End with FINAL: followed by '
          'one number, without currency symbols, units, or angle brackets. '
          'Use digits for the number; do not leave a placeholder.')
NUMBER = r'[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?'


def key(question):
    return hashlib.sha256(' '.join(question.lower().split()).encode()).hexdigest()


def canonical(value):
    try:
        number = Decimal(value.replace(',', '').strip())
        return format(number.normalize(), 'f') if number.is_finite() else None
    except InvalidOperation:
        return None


def gold(answer):
    tail = answer.rsplit('####', 1)
    if len(tail) != 2 or not re.fullmatch(NUMBER, tail[1].strip()):
        raise ValueError('Unexpected GSM8K gold answer')
    return canonical(tail[1])


def parse(response, tolerant=True):
    if '<think>' in response and '</think>' not in response:
        return None
    response = response.split('</think>')[-1]
    # Only use explicitly marked final answers; never use the last reasoning number.
    if tolerant:
        response = re.sub(r'(?i)(FINAL:\s*)<answer>\s*', r'\1', response)
        response = re.sub(r'(?i)(FINAL:\s*)\$\s*', r'\1', response)
        pattern = rf'(?i)\bFINAL:\s*({NUMBER})\s*\.?\s*$'
    else:
        pattern = rf'(?im)^\s*FINAL:\s*({NUMBER})\s*\.?\s*$'
    matches = list(re.finditer(pattern, response))
    return canonical(matches[-1].group(1)) if matches else None


def messages(question, demonstrations):
    result = [{'role': 'system', 'content': SYSTEM}]
    for item in demonstrations:
        reasoning = item['answer'].rsplit('####', 1)[0]
        reasoning = re.sub(r'<<[^>]*>>', '', reasoning).strip()
        result.extend([{'role': 'user', 'content': item['question']},
                       {'role': 'assistant', 'content': reasoning + '\nFINAL: ' + gold(item['answer'])}])
    result.append({'role': 'user', 'content': question})
    return result


class FirstTokenTimer:
    """generate() calls put once for the prompt, then once per generated token.

    Timestamp is token-ID availability at a CPU callback, not visible text delivery.
    """
    def __init__(self, sync, clock=time.perf_counter):
        self.sync, self.clock = sync, clock
        self.prompt_seen = False
        self.first_time = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_time is None:
            self.sync()
            self.first_time = self.clock()

    def end(self):
        pass
