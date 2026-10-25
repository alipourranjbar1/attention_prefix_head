"""Record greedy responses, then compare after restarting with confidence enabled.

Run with --output baseline.json against flag-off server. Restart with flag on,
then use --baseline baseline.json --output confidence.json. No NPU mocks here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8216')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--num-prompts', type=int, default=16)
    args = parser.parse_args()
    if args.concurrency < 1 or args.num_prompts < 1:
        raise SystemExit('Concurrency and num-prompts must be positive')
    if args.output.exists():
        raise SystemExit('Choose a new output file')
    with urlopen(args.url + '/v1/models', timeout=30) as response:
        model = json.load(response)['data'][0]['id']
    prompts = ['The capital of France is', 'Explain why the sky is blue.',
               'Write a Python function that computes Fibonacci numbers.',
               'Count from one to ten:', 'The capital of France is']
    def generate(index):
        prompt = prompts[index % len(prompts)]
        payload = dict(model=model, prompt=prompt, temperature=0, seed=42,
                       max_tokens=1 if index == 3 else 64, return_token_ids=True)
        start = time.monotonic()
        request = Request(args.url + '/v1/completions',
                          data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
        choice = result['choices'][0]
        if not isinstance(choice.get('token_ids'), list):
            raise RuntimeError('Server did not return token IDs for parity checking')
        return dict(prompt=prompt, token_ids=choice['token_ids'],
                    finish_reason=choice['finish_reason'], usage=result['usage'],
                    elapsed_seconds=time.monotonic()-start)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(generate, range(args.num_prompts)))
    args.output.write_text(json.dumps(dict(url=args.url, model=model, rows=rows), indent=2))
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())['rows']
        compare = lambda data: [(x['prompt'], x['token_ids'], x['finish_reason']) for x in data]
        if compare(baseline) != compare(rows):
            raise SystemExit('Greedy token parity FAILED; inspect saved responses')
        print('Greedy token parity passed. Inspect server truncation logs separately.')
    print(f'Responses saved: {args.output}')


if __name__ == '__main__':
    main()
