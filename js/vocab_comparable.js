// ============================================================================
// The "comparable @ 16k" vocabulary-size formula.
//
// Depth 16,000, RAW ranks 1..depth, NO blocklist/eligibility filtering at
// all: a deliberately simple cross-language index computable from a saved
// sitting's fitted curve {a,b} alone, so every language and every past
// sitting gets it with no extra data. Works as a browser <script>
// (window.IT.*) and as a Node require() (module.exports), so the tests load
// this same file rather than re-typing the formula.
'use strict';

// p_know(x) = 1/(1+exp(-(a - b*x))), x = log10(rank). A local sigmoid clamp —
// deliberately NOT shared with the estimator block's own `sigmoid` (which
// stays local there: test-estimator.js exercises it directly by name via
// M.sigmoid). Duplicating this four-line clamp is cheaper than threading a
// cross-file dependency for it.
function vocabComparableSigmoid(z) {
  if (z > 40) return 1;
  if (z < -40) return 0;
  return 1 / (1 + Math.exp(-z));
}
function pKnow(a, b, x) { return vocabComparableSigmoid(a - b * x); }

var COMPARE_DEPTH = 16000;
function comparableEstimate(curve, depth) {
  depth = depth || COMPARE_DEPTH;
  var sum = 0;
  for (var r = 1; r <= depth; r++) {
    sum += pKnow(curve.a, curve.b, Math.log10(r));
  }
  return Math.round(sum);
}

if (typeof window !== 'undefined') {
  window.IT = window.IT || {};
  window.IT.pKnow = pKnow;
  window.IT.comparableEstimate = comparableEstimate;
  window.IT.COMPARE_DEPTH = COMPARE_DEPTH;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { pKnow: pKnow, comparableEstimate: comparableEstimate, COMPARE_DEPTH: COMPARE_DEPTH };
}
