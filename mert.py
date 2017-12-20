import argparse
import math
import random
import sys

from collections import namedtuple, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial

Line = namedtuple('Line', 'm, b')
def find_intersection(line1, line2):
  if line1.m == line2.m:
    return None
  return -(line1.b - line2.b) / (line1.m - line2.m)

class LineCollection:
  def __init__(self):
    self.lines = []

  def add(self, line):
    self.lines.append(line)

  def find_intersections(self):
    """finds intersection points of each pair of lines adjacent in the collection
    there will be exactly N-1 such points"""
    intersections = []
    for i in range(1, len(self.lines)):
      line1 = self.lines[i - 1]
      line2 = self.lines[i]
      intersection = find_intersection(line1, line2)
      intersections.append(intersection)
    assert len(intersections) == len(self.lines) - 1
    return intersections

  def find_all_intersections(self):
    """returns all intersection points between two lines in the collection.
    there are in general N*(N-1)/2 such points"""
    intersections = []
    for i in range(len(self.lines)):
      for j in range(i + 1, len(self.lines)):
        line1 = self.lines[i]
        line2 = self.lines[j]
        intersection = find_intersection(line1, line2)
        if intersection is not None:
          intersections.append((i, j, intersection))
    return intersections

  def plot(self):
    intersections = self.find_all_intersections()
    intersections = [v for (i, j, v) in intersections]
    intersections = sorted(intersections)
    if len(intersections) == 0:
      x_min = -1
      x_max = 1
    elif len(intersections) == 1:
      x_min = intersections[0] - 1
      x_max = intersections[1] + 1
    else:
      x_min = intersections[0]
      x_max = intersections[-1]
      dist = x_max - x_min
      x_min -= 0.1 * dist
      x_max += 0.1 * dist
    
    for m, b in self.lines:
      y1 = m * x_min + b
      y2 = m * x_max + b
      plt.plot((x_min, x_max), (y1, y2), 'k-')

    plt.gca().set_xlim(x_min, x_max)
    plt.show()

  def plot_envelope(self):
    envelope = self.find_upper_envelope()
    env_lc.plot()
 
  def find_upper_envelope(self):
    # Start by solving the dual problem: the convex hull
    # If there are two or fewer points, all of them are on the hull
    # (and the scipy library barfs), so we special case this.
    # We will consider each line to instead be a 2-d point (m, b)
    points = self.lines
    assert len(points) > 0
    if len(points) == 1:
      vertices = [0]
    elif len(points) == 2:
      if points[0][0] > points[1][0]:
        vertices = [0, 1]
      else:
        vertices = [1, 0]
    elif are_collinear(points):
      # If the points are all collinear then the convex hull is composed of the
      # one with the lowest slope and the one with the highest slope, and that's it
      # TODO: In case of ties, pick the one with the highest b?
      slope_ordered = sorted(list(range(len(points))), key=lambda i: points[i])
      vertices = [slope_ordered[0], slope_ordered[-1]]
    else:
      hull = scipy.spatial.ConvexHull(points)
      vertices = hull.vertices

    # Now start with the point with the highest slope
    assert len(points) > 0
    highest_slope_index = 0
    for i in range(1, len(vertices)):
      if points[vertices[i]][0] > points[vertices[highest_slope_index]][0]:
        highest_slope_index = i
   
    # Take lines on the convex hull in order of decreasing slope
    # When the slopes start increasing again, we've hit the bottom
    # part of the convex hull and can quit
    envelope = []
    prev_slope = float('inf')
    for i in range(0, len(vertices)):
      j = (highest_slope_index + i) % len(vertices)
      point = points[vertices[j]]
      slope = point[0]
      if slope > prev_slope:
        break
      envelope.append(vertices[j])
      prev_slope = slope

    # Reverse the list, so we get points with increasing slopes instead of decreasing
    envelope = list(reversed(envelope))

    # Pack the relevant lines into a new envelope object
    return Envelope([self.lines[i] for i in envelope], envelope)

class Envelope(LineCollection):
  def __init__(self, lines, indices):
    self.lines = lines
    self.indices = indices
    self.intersections = self.find_intersections()
    assert len(self.lines) == len(self.indices)
    assert len(self.lines) > 0

  def add(self, line):
    assert False, 'Cannot add to an envelope!'

  def get_hyp_at(self, x):
    for i in range(len(self.intersections)):
      # index represents the index of the hyp in the original
      # set of lines, not just the envelope.
      # this index will be returned for values up to "intersection"
      index = self.indices[i]
      intersection = self.intersections[i]
      if x < intersection:
        return index
    return self.indices[-1]

class HypothesisSet:
  def __init__(self):
    self.line_collection = None
    self.hyps = []
    self.feats = []
    self.metric_scores = []

  def add(self, hyp, feats, metric_score):
    self.hyps.append(hyp)
    self.feats.append(feats)
    self.metric_scores.append(metric_score)

  def compute_lines(self, weights, direction):
    line_collection = LineCollection()
    for hyp_feats in self.feats:
      m = dot(hyp_feats, direction)
      b = dot(hyp_feats, weights)
      line_collection.add(Line(m, b))
    self.line_collection = line_collection

  def get_all_feature_names(self):
    feature_names = set()
    for feat_dict in self.feats:
      feature_names |= set(feat_dict.keys())
    return feature_names

class KbestList:
  def __init__(self):
    self.hyp_sets = defaultdict(HypothesisSet)

  @staticmethod
  def from_file(filename):
    kbest_list = KbestList()
    kbest_list.read(filename)
    return kbest_list

  @staticmethod
  def parse_features(feat_string):
    features = {}
    parts = feat_string.split()
    for part in parts:
      key, value = part.split('=')
      features[key] = float(value)
    return features

  def read(self, filename):
    with open(filename) as f:
      for line in f:
        parts = [part.strip() for part in line.split('|||')]
        assert len(parts) >= 4
        sent_id, hyp_string, feat_string = parts[:3]
        metric_score = float(parts[-1])
        features = KbestList.parse_features(feat_string)
        self.hyp_sets[sent_id].add(hyp_string, features, metric_score)

  def get_all_feature_names(self):
    feature_names = set()
    for hyp_set in self.hyp_sets.itervalues():
      feature_names |= hyp_set.get_all_feature_names()
    return feature_names

  @staticmethod
  def get_interesting_points(intersections):
    if len(intersections) == 0:
      yield 0.0
      return

    yield intersections[0] - 1.0
    for i in range(1, len(intersections)):
      yield (intersections[i] + intersections[i - 1]) / 2.0
    yield intersections[-1] + 1.0

  def mert(self, weights, direction):
    envelopes = {}
    all_intersections = list()
    for sent_id, hyp_set in self.hyp_sets.iteritems():
      hyp_set.compute_lines(weights, direction)
      envelope = hyp_set.line_collection.find_upper_envelope()
      envelopes[sent_id] = envelope  
      all_intersections += envelope.intersections

    all_intersections = sorted(all_intersections)

    best_x = None
    best_metric_score = None
    for x in KbestList.get_interesting_points(all_intersections):
      total_metric_score = 0.0
      for sent_id, hyp_set in self.hyp_sets.iteritems():
        hyp_index = envelopes[sent_id].get_hyp_at(x)
        metric_score = hyp_set.metric_scores[hyp_index]
        total_metric_score += metric_score
      if best_x is None or total_metric_score > best_metric_score:
        best_x = x
        best_metric_score = total_metric_score

    print >>sys.stderr, 'Best x:', best_x
    print >>sys.stderr, 'Metric score:', best_metric_score / len(self.hyp_sets)
    new_weights = {k: weights[k] + best_x * v for (k, v) in direction.iteritems()}
    return new_weights

def slope(p1, p2):
  if p1[0] == p2[0]:
    return float('inf') if p1[1] != p2[1] else 0.0
  return (p1[1] - p2[1]) / (p1[0] - p2[0])

def are_collinear(points):
  if len(points) <= 2:
    return True

  a = points[0]
  b = points[1]
  m = slope(a, b)
  for c in points[2:]:
    n = slope(a, c)
    if n != m:
      return False

  return True

def choose_direction(features):
  r = {key: 2 * random.random() - 1.0 for key in features}
  length = sum(v**2 for v in r.values())
  d = {k: v / length for (k, v) in r.iteritems()}
  return d

def parse_weights_file(filename):
  weights = defaultdict(float)
  with open(filename) as f:
    for line in f:
      key, value = line.split()
      weights[key] = float(value)
  return weights

def dump_weights_file(weights):
  for key, value in weights.iteritems():
    print key, value

def dot(features, weights):
  score = 0.0
  for key in set(features.keys()) & set(weights.keys()):
    score += features[key] * weights[key]
  return score

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-k', '--kbest', required=True, help='K-best list to learn from')
  parser.add_argument('-w', '--initial_weights', required=False, help='Initial weights file (default is all zeros)')
  args = parser.parse_args()

  if args.initial_weights:
    initial_weights = parse_weights_file(args.initial_weights)
  else:
    initial_weights = defaultdict(float)

  kbest = KbestList.from_file(args.kbest)
  direction = choose_direction(kbest.get_all_feature_names())
  new_weights = kbest.mert(initial_weights, direction)
  dump_weights_file(new_weights)

if __name__ == '__main__':
  main()
