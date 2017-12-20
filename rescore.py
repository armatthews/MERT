import argparse
import sys

from collections import defaultdict

def dot(features, weights):
  score = 0.0
  for key in set(features.keys()) & set(weights.keys()):
    score += features[key] * weights[key]
  return score

def parse_features(feat_string):
  features = {}
  parts = feat_string.split()
  for part in parts:
    key, value = part.split('=')
    features[key] = float(value)
  return features

def rescore_nbest(stream, weights):
  prev_sent_id = None
  best_score = None
  best_line = None

  for line in stream:
    parts = [part.strip() for part in line.split('|||')]
    assert len(parts) >= 3
    sent_id = parts[0]
    feat_string = parts[2]
    features = parse_features(feat_string)
    score = dot(features, weights)

    if prev_sent_id is not None and sent_id != prev_sent_id:
      print best_line.strip()
      best_score = None
      best_line = None
    prev_sent_id = sent_id
      
    if best_score is None or score > best_score:
      best_score = score
      best_line = line

  if prev_sent_id is not None:
    print best_line.strip()

def parse_weights_file(filename):
  weights = defaultdict(float)
  with open(filename) as f:
    for line in f:
      key, value = line.split()
      weights[key] = float(value)
  return weights

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('weights')
  args = parser.parse_args()

  weights = parse_weights_file(args.weights)
  rescore_nbest(sys.stdin, weights)

if __name__ == '__main__':
  main()
