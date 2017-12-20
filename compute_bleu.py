import argparse
import math
import sys

from collections import Counter

def read_refs(filename):
  refs = {}
  with open(filename) as f:
    for line in f:
      # Allow either plain text or
      # sent_id ||| ref plain text
      # formats
      if '|||' in line:
        parts = line.split('|||', 1)
        sent_id = parts[0].strip()
        ref = parts[1].strip()
      else:
        sent_id = str(len(refs))
        ref = line.strip()
      refs[sent_id] = ref
  return refs

def find_ngrams(sent, n):
  words = sent.split()
  ngrams = []
  for i in range(len(words) - n + 1):
    ngrams.append(tuple(words[i : i + n]))
  return Counter(ngrams)

def compute_bleu(hyp, ref, n):
  log_precisions = []
  for i in range(1, n + 1):
    hyp_ngrams = find_ngrams(hyp, i)
    ref_ngrams = find_ngrams(ref, i)
    matches = hyp_ngrams & ref_ngrams
    num_matches = sum(matches.itervalues())
    if num_matches == 0:
      num_matches = 1.e-10

    log_precision = math.log(num_matches) - math.log(len(hyp_ngrams))
    log_precisions.append(log_precision) 

  hyp_len = len(hyp.split())
  ref_len = len(ref.split())
  if hyp_len == 0:
    return 0.0 if ref_len > 0 else 1.0

  if hyp_len < ref_len:
    log_bp = 1.0 - 1.0 * ref_len / hyp_len
  else:
    log_bp = 0.0

  log_bleu = sum(log_precisions) / len(log_precisions) + log_bp
  return math.exp(log_bleu)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('refs')
  parser.add_argument('-n', type=int, default=4, help='BLEU n-gram order')
  args = parser.parse_args()

  refs = read_refs(args.refs)

  for line in sys.stdin:
    parts = [part.strip() for part in line.split('|||')]
    sent_id = parts[0]
    hyp = parts[1]
    ref = refs[sent_id]
    bleu_score = compute_bleu(hyp, ref, args.n)
    parts.append(str(bleu_score))
    print ' ||| '.join(parts)

if __name__ == '__main__':
  main()
