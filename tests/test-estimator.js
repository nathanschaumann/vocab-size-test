// Plain node, no deps. Extracts the real estimator + band-allocation +
// sampling-exclusion blocks out of index.html between the TEST-EXTRACT
// markers and evaluates them in a vm context (a lightweight way to unit-test
// code that lives inside a single-file app).
//
// Run: node tests/test-estimator.js
'use strict';
var fs = require('fs');
var path = require('path');
var vm = require('vm');

var ROOT = path.join(__dirname, '..');
var src = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');

function slice(tag) {
  var b = 'TEST-EXTRACT:' + tag + ':BEGIN';
  var e = 'TEST-EXTRACT:' + tag + ':END';
  var i = src.indexOf(b), j = src.indexOf(e);
  if (i < 0 || j < 0) throw new Error('missing TEST-EXTRACT markers for ' + tag);
  return src.slice(src.indexOf('\n', i) + 1, src.lastIndexOf('\n', j)) + '\n';
}

var estimatorSrc = slice('estimator');
var bandAllocSrc = slice('bandAlloc');
var samplingSrc = slice('sampling');

var EXPORT_NAMES = [
  'MAX_RANK', 'F_CLAMP_MAX', 'PARAM_BOUNDS', 'sigmoid', 'clampNum', 'clampProb',
  'pKnow', 'projectParams', 'negLogLikDirect', 'negLogLikBounded', 'nelderMead',
  'computeFalseAlarmRate', 'fitModel', 'computeSize', 'computeSizeEligible',
  'eligibleIndexForRank',
  'COMPARE_DEPTH', 'comparableEstimate',
  'mulberry32', 'bootstrapCI', 'fitEstimator',
  'BAND_EDGES', 'bandLoHi', 'bandMidpointX', 'allocateBandCounts', 'MAX_MC_PER_BAND',
  'sittingLooksInvalid', 'INVALID_F_THRESHOLD',
  'NO_REUSE_DAYS', 'historyWithinDays', 'excludedYesNoLemmas', 'excludedPseudoIds',
  'excludedItemIds', 'sampleWithoutReplacement', 'selectYesNoRealLemmas',
  'selectPseudowords', 'selectMcItems',
];

var fullSrc = estimatorSrc + '\n' + bandAllocSrc + '\n' + samplingSrc
  + '\nthis.__exports = {' + EXPORT_NAMES.join(', ') + '};\n';

// pKnow/comparableEstimate/COMPARE_DEPTH were moved out of the estimator
// block into js/vocab_comparable.js (see that file's own header). The extracted
// block now aliases them off `window.IT` rather than defining them itself,
// so this sandbox needs a real `window` before the slice runs: `sandbox`
// IS the vm context's global object, and pointing `window` at itself (set
// BEFORE vm.createContext, so it's part of the context's initial globals)
// means the shared file's own `window.IT = window.IT || {}` lands directly
// on this same sandbox — no separate "module" wiring needed.
var sandbox = {};
sandbox.window = sandbox;
vm.createContext(sandbox);
var sharedSrc = fs.readFileSync(path.join(ROOT, 'js', 'vocab_comparable.js'), 'utf8');
vm.runInContext(sharedSrc, sandbox, { filename: 'js/vocab_comparable.js' });
vm.runInContext(fullSrc, sandbox, { filename: 'index.html (TEST-EXTRACT slice)' });
var M = sandbox.__exports;

var PASS = 0, FAIL = 0;
function ok(cond, msg) { if (cond) { PASS++; } else { FAIL++; console.log('  FAIL: ' + msg); } }
function approx(a, b, tol, msg) {
  ok(Math.abs(a - b) <= tol, msg + ' (got ' + a + ', want within ' + tol + ' of ' + b + ')');
}

/* ---------------------------------------------------------------------------
   Synthetic data generator — mirrors the app's own item counts (105 real
   yes/no + 35 pseudowords + 60 MC) so recovery is checked at realistic
   sample sizes, per the task brief.
   ------------------------------------------------------------------------ */
function genDataset(trueA, trueB, trueF, trueG, opts) {
  opts = opts || {};
  var rng = M.mulberry32(opts.seed || 1);
  var bandEdges = M.BAND_EDGES;
  function randRankInBand(bi) {
    var lo = bi === 0 ? 1 : bandEdges[bi - 1] + 1;
    var hi = bandEdges[bi];
    return lo + Math.floor(rng() * (hi - lo + 1));
  }
  var perBand = opts.perBandYesNo != null ? opts.perBandYesNo : 15;
  var yesnoReal = [];
  for (var bi = 0; bi < 7; bi++) {
    for (var k = 0; k < perBand; k++) {
      var rank = randRankInBand(bi);
      var x = Math.log10(rank);
      var p = M.pKnow(trueA, trueB, x);
      var q = p + (1 - p) * trueF;
      yesnoReal.push({ rank: rank, claimed: rng() < q });
    }
  }
  var nPseudo = opts.nPseudo != null ? opts.nPseudo : 35;
  var pseudo = [];
  for (var i = 0; i < nPseudo; i++) pseudo.push({ claimed: rng() < trueF });

  var nMc = opts.nMc != null ? opts.nMc : 60;
  var mc = [];
  for (var m = 0; m < nMc; m++) {
    var bi2 = m % 7;
    var rank2 = randRankInBand(bi2);
    var x2 = Math.log10(rank2);
    var p2 = M.pKnow(trueA, trueB, x2);
    var pCorrect = p2 + (1 - p2) * (1 - trueG) * 0.25;
    var pIdk = (1 - p2) * trueG;
    var u = rng();
    var outcome;
    if (u < pCorrect) outcome = 'correct';
    else if (u < pCorrect + pIdk) outcome = 'idk';
    else outcome = 'wrong';
    mc.push({ rank: rank2, outcome: outcome });
  }
  return { yesnoReal: yesnoReal, pseudo: pseudo, mc: mc };
}

/* ---------------------------------------------------------------------------
   (a) Synthetic recovery — one representative, hand-verified scenario at
   realistic item counts. A gradual, realistic-shaped curve (not a
   near-step-function edge case — those are covered separately in (c)).
   ------------------------------------------------------------------------ */
(function testRecovery() {
  var trueA = 3.2, trueB = 1.1, trueF = 0.1, trueG = 0.4;
  var trueSize = M.computeSize(trueA, trueB);
  var data = genDataset(trueA, trueB, trueF, trueG, { seed: 6, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var fit = M.fitEstimator(data, { rng: M.mulberry32(1006), bootstrapReps: 200 });
  var ratio = fit.size / trueSize;
  ok(Math.abs(ratio - 1) <= 0.12,
    '(a) size recovery within ~12% of true (true=' + trueSize.toFixed(0) + ', fit=' + fit.size.toFixed(0) + ', ratio=' + ratio.toFixed(3) + ')');
  ok(isFinite(fit.a) && isFinite(fit.b) && isFinite(fit.g) && isFinite(fit.f),
    '(a) fitted params are finite');
})();

/* ---------------------------------------------------------------------------
   (b) More knowledge -> bigger estimate, on paired synthetic sets.
   ------------------------------------------------------------------------ */
(function testMonotonicity() {
  var dataLo = genDataset(1.0, 1.3, 0.1, 0.4, { seed: 7, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var dataHi = genDataset(5.0, 1.3, 0.1, 0.4, { seed: 7, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var fitLo = M.fitEstimator(dataLo, { rng: M.mulberry32(1), bootstrapReps: 0 });
  var fitHi = M.fitEstimator(dataHi, { rng: M.mulberry32(2), bootstrapReps: 0 });
  ok(fitHi.size > fitLo.size,
    '(b) more knowledge (paired synthetic sets) gives a bigger estimate (lo=' + fitLo.size.toFixed(0) + ', hi=' + fitHi.size.toFixed(0) + ')');
})();

/* ---------------------------------------------------------------------------
   (c) Edge cases — all-correct / all-IDK, degenerate f. Finite and ordered.
   ------------------------------------------------------------------------ */
(function testEdgeCases() {
  var allKnownYn = [], allKnownMc = [], allKnownPseudo = [];
  for (var i = 0; i < 40; i++) allKnownYn.push({ rank: 1 + i * 500, claimed: true });
  for (i = 0; i < 30; i++) allKnownMc.push({ rank: 1 + i * 500, outcome: 'correct' });
  for (i = 0; i < 20; i++) allKnownPseudo.push({ claimed: false });
  var fitAllKnown = M.fitEstimator({ yesnoReal: allKnownYn, pseudo: allKnownPseudo, mc: allKnownMc },
    { rng: M.mulberry32(3), bootstrapReps: 60 });
  ok(isFinite(fitAllKnown.size) && isFinite(fitAllKnown.ciLo) && isFinite(fitAllKnown.ciHi),
    '(c) all-correct: size + CI are finite');
  ok(fitAllKnown.ciLo <= fitAllKnown.ciHi, '(c) all-correct: CI ordered (lo<=hi)');
  ok(fitAllKnown.size > 25000, '(c) all-correct: size is high (' + fitAllKnown.size.toFixed(0) + ')');

  var allUnkYn = [], allUnkMc = [];
  for (i = 0; i < 40; i++) allUnkYn.push({ rank: 1 + i * 500, claimed: false });
  for (i = 0; i < 30; i++) allUnkMc.push({ rank: 1 + i * 500, outcome: 'idk' });
  var fitAllUnk = M.fitEstimator({ yesnoReal: allUnkYn, pseudo: allKnownPseudo, mc: allUnkMc },
    { rng: M.mulberry32(4), bootstrapReps: 60 });
  ok(isFinite(fitAllUnk.size) && isFinite(fitAllUnk.ciLo) && isFinite(fitAllUnk.ciHi),
    '(c) all-idk: size + CI are finite');
  ok(fitAllUnk.ciLo <= fitAllUnk.ciHi, '(c) all-idk: CI ordered (lo<=hi)');
  ok(fitAllUnk.size < 5000, '(c) all-idk: size is low (' + fitAllUnk.size.toFixed(0) + ')');

  // f close to 1 (near-universal pseudoword false alarms) must not produce NaN.
  var fNearOneYn = [], fNearOnePseudo = [], fNearOneMc = [];
  for (i = 0; i < 40; i++) fNearOneYn.push({ rank: 1 + i * 500, claimed: i % 2 === 0 });
  for (i = 0; i < 20; i++) fNearOnePseudo.push({ claimed: true });
  for (i = 0; i < 30; i++) fNearOneMc.push({ rank: 1 + i * 500, outcome: (i % 3 === 0 ? 'correct' : (i % 3 === 1 ? 'wrong' : 'idk')) });
  var fitFNearOne = M.fitEstimator({ yesnoReal: fNearOneYn, pseudo: fNearOnePseudo, mc: fNearOneMc },
    { rng: M.mulberry32(5), bootstrapReps: 60 });
  ok(isFinite(fitFNearOne.size) && isFinite(fitFNearOne.ciLo) && isFinite(fitFNearOne.ciHi),
    '(c) f-near-1: size + CI are finite');
  ok(fitFNearOne.ciLo <= fitFNearOne.ciHi, '(c) f-near-1: CI ordered (lo<=hi)');
})();

/* ---------------------------------------------------------------------------
   (d) Bootstrap CI: finite, ordered, and brackets the point estimate.
   ------------------------------------------------------------------------ */
(function testBootstrapCI() {
  var trueA = 3.2, trueB = 1.1, trueF = 0.1, trueG = 0.4;
  var data = genDataset(trueA, trueB, trueF, trueG, { seed: 6, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var fit = M.fitEstimator(data, { rng: M.mulberry32(1006), bootstrapReps: 200 });
  ok(isFinite(fit.ciLo) && isFinite(fit.ciHi), '(d) CI bounds are finite');
  ok(fit.ciLo <= fit.ciHi, '(d) CI is ordered (lo<=hi)');
  ok(fit.ciLo <= fit.size + 1e-6 && fit.size <= fit.ciHi + 1e-6,
    '(d) CI brackets the point estimate (lo=' + fit.ciLo.toFixed(0) + ' size=' + fit.size.toFixed(0) + ' hi=' + fit.ciHi.toFixed(0) + ')');
})();

/* ---------------------------------------------------------------------------
   (e) 68% CI coverage — loose check over 100 synthetic replications.
   Reduced bootstrap reps (60, vs the app's real 200) purely to keep this
   node test's own runtime reasonable; the app itself always uses 200 (see
   the "timing" self-check below, run at the real 200-rep default).

   OPT-IN ONLY, NOT PART OF THE DEFAULT WIRED-IN RUN (2026-09-02, audit item
   61). 100 replications x a real bootstrap fit is genuinely expensive — a
   single `fitEstimator` call measured 4.6-9.2s in THIS repo's vm-sandboxed
   harness (see testTiming's own comment on why the sandbox is unusually
   slow), so REPS=100 costs several minutes to over ten depending on machine
   load, dwarfing the ENTIRE REST of the suite (~625s serial, per the audit,
   across 258 files). That is fine for an occasional deliberate statistical
   re-validation and wrong for a file that is supposed to run on every
   `scripts/run-tests.sh`. Thinning REPS or BOOT_REPS instead of gating was
   tried and rejected: the 55-80% acceptance band is already sized for
   REPS=100's sampling noise (SE~4.7pp there; at REPS=30 SE~8.5pp, which
   would make the check itself flaky rather than fast). So the real check
   stays intact, full-strength, just off by default:
     IT_SLOW_ESTIMATOR_COVERAGE=1 node vocab/scripts/test-estimator.js
   fitOpts.seed skips fitModel's 420-point grid search on each of the 100
   synthetic refits (already-tested elsewhere by (a)-(d) above, which use
   the default un-seeded path) — a real, measured ~40% cut with no loss of
   what THIS check is actually about (coverage, not grid-search accuracy).
   ------------------------------------------------------------------------ */
if (process.env.IT_SLOW_ESTIMATOR_COVERAGE === '1') {
  (function testCoverage() {
    var trueA = 3.2, trueB = 1.1, trueF = 0.1, trueG = 0.4;
    var trueSize = M.computeSize(trueA, trueB);
    var REPS = 100, BOOT_REPS = 60;
    var covered = 0;
    for (var rep = 0; rep < REPS; rep++) {
      var data = genDataset(trueA, trueB, trueF, trueG, { seed: 5000 + rep, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
      var fit = M.fitEstimator(data, {
        rng: M.mulberry32(9000 + rep), bootstrapReps: BOOT_REPS,
        fitOpts: { seed: [trueA, trueB, trueG] }
      });
      if (fit.ciLo <= trueSize && trueSize <= fit.ciHi) covered++;
    }
    var pct = covered / REPS;
    ok(pct >= 0.55 && pct <= 0.80,
      '(e) 68% CI covers the true size in a plausible range across 100 replications (got ' + (pct * 100).toFixed(0) + '%)');
  })();
} else {
  console.log('  (e) 68% CI coverage: SKIPPED by default (slow — opt in with IT_SLOW_ESTIMATOR_COVERAGE=1)');
}

/* ---------------------------------------------------------------------------
   (f) Band allocation returns exactly 60, with a minimum of 2 per band.
   ------------------------------------------------------------------------ */
(function testBandAllocation() {
  var weights = [0.05, 0.15, 0.25, 0.24, 0.15, 0.08, 0.03];
  var alloc = M.allocateBandCounts(weights, 60, 2);
  ok(alloc.length === 7, '(f) allocation has 7 bands');
  var sum = alloc.reduce(function (s, x) { return s + x; }, 0);
  ok(sum === 60, '(f) allocation sums to exactly 60 (got ' + sum + ')');
  ok(alloc.every(function (x) { return x >= 2; }), '(f) every band gets at least 2 (got ' + JSON.stringify(alloc) + ')');

  // Degenerate weights (all zero, and a single extreme outlier) must still
  // sum to exactly 60 with the min-2 floor honored.
  var allocZero = M.allocateBandCounts([0, 0, 0, 0, 0, 0, 0], 60, 2);
  ok(allocZero.reduce(function (s, x) { return s + x; }, 0) === 60, '(f) zero-weight allocation still sums to 60');
  ok(allocZero.every(function (x) { return x >= 2; }), '(f) zero-weight allocation still honors min 2');

  var allocOneHot = M.allocateBandCounts([0, 0, 0, 100, 0, 0, 0], 60, 2);
  ok(allocOneHot.reduce(function (s, x) { return s + x; }, 0) === 60, '(f) one-hot allocation still sums to 60');
  ok(allocOneHot.every(function (x) { return x >= 2; }), '(f) one-hot allocation still honors min 2');
  ok(allocOneHot[3] > allocOneHot[0], '(f) one-hot allocation concentrates on the weighted band');
})();

/* ---------------------------------------------------------------------------
   Bonus: band midpoint + sampling-exclusion sanity (small, cheap, catches
   regressions in the "// TEST-EXTRACT:sampling" block since (a)-(f) above
   don't exercise it directly).
   ------------------------------------------------------------------------ */
(function testSamplingHelpers() {
  approx(M.bandMidpointX(1, 500), (Math.log10(1) + Math.log10(500)) / 2, 1e-9, 'bandMidpointX band 1');
  var loHi = M.bandLoHi(0);
  ok(loHi[0] === 1 && loHi[1] === 500, 'bandLoHi(0) is [1,500]');
  var loHi3 = M.bandLoHi(3);
  ok(loHi3[0] === 2001 && loHi3[1] === 4000, 'bandLoHi(3) is [2001,4000]');

  var history = [
    { at: new Date().toISOString(), language: 'es', usedYesNoLemmas: ['casa'], usedPseudoIds: ['pw-es-0001'], usedItemIds: ['mc-es-0001'] },
    { at: new Date(Date.now() - 200 * 86400000).toISOString(), language: 'es', usedYesNoLemmas: ['agua'], usedPseudoIds: ['pw-es-0002'], usedItemIds: ['mc-es-0002'] },
  ];
  var excl = M.excludedYesNoLemmas(history, 'es', 180);
  ok(excl.has('casa'), 'excludedYesNoLemmas includes a recent entry');
  ok(!excl.has('agua'), 'excludedYesNoLemmas excludes an entry older than 180 days');

  var pool = [{ id: 'pw-1' }, { id: 'pw-2' }, { id: 'pw-3' }];
  var excludedSet = new Set(['pw-1', 'pw-2', 'pw-3']);
  var sel = M.selectPseudowords(pool, excludedSet, 3, M.mulberry32(1));
  ok(sel.items.length === 3, 'selectPseudowords degrades gracefully when the whole pool is excluded');
  ok(sel.reusedItems === true, 'selectPseudowords flags reusedItems when it had to relax the exclusion');

  var freshPool = [{ id: 'pw-4' }, { id: 'pw-5' }];
  var freshSel = M.selectPseudowords(freshPool, new Set(), 2, M.mulberry32(1));
  ok(freshSel.reusedItems === false, 'selectPseudowords does not flag reuse when the exclusion was never violated');
})();

/* ---------------------------------------------------------------------------
   Timing self-check: the app's real bootstrap (200 reps) must run well
   under ~2s, per the design (must run in under ~2s in JS).
   ------------------------------------------------------------------------ */
(function testTiming() {
  var trueA = 3.2, trueB = 1.1, trueF = 0.1, trueG = 0.4;
  var data = genDataset(trueA, trueB, trueF, trueG, { seed: 6, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var t0 = Date.now();
  M.fitEstimator(data, { rng: M.mulberry32(1), bootstrapReps: 200 });
  var elapsed = Date.now() - t0;
  // The "<~2s" budget is for the REAL browser environment. This
  // harness runs the extracted code inside a vm.Context sandbox (the house
  // extraction convention), and V8 cannot JIT cross-context calls nearly as
  // well as same-context code — real in-browser timing is ~200-300ms
  // (verified separately, outside any vm context), well under the <2s
  // budget. So this check uses a generous sandbox-only ceiling to catch a
  // genuine algorithmic regression (e.g. an accidental O(n^2) blowup)
  // without being flaky on the vm overhead itself.
  //
  // CEILING RAISED 15000 -> 60000ms (2026-09-02, audit item 61): the prior
  // 15000ms ceiling was tuned on a machine 15-20x slower than plain node:
  // in the environment that actually runs this suite, three clean back-to-
  // back samples of this exact call came back 13015 / 22719 / 24253ms —
  // 45-80x plain-node speed, not 15-20x, and already past the old ceiling
  // more often than not. 60000ms keeps real margin (>2x the worst sample
  // seen) while still catching an actual blowup, which would land in the
  // minutes, not tens of seconds.
  ok(elapsed < 60000, 'full fit + 200-rep bootstrap completes in bounded time even inside the slow vm sandbox (took ' + elapsed + 'ms; real in-browser timing is ~200-300ms, well under the <2s budget)');
})();

/* ---------------------------------------------------------------------------
   Eligible-ranks size (blocklist-aware): the headline must count ONLY
   non-blocklisted ranks. Added 2026-08-22 after the real es blocklist came
   back 31.5% of the list — summing over all 30k ranks would count junk
   ranks as knowable words.
   ------------------------------------------------------------------------ */
(function testEligibleSize() {
  var a = 3.0, b = 1.2;
  // All ranks eligible == the plain computeSize, exactly.
  var allXs = [];
  for (var r = 1; r <= M.MAX_RANK; r++) allXs.push(Math.log10(r));
  approx(M.computeSizeEligible(a, b, allXs), M.computeSize(a, b), 1e-6,
    'computeSizeEligible over all ranks equals computeSize');

  // A subset: equals the hand-summed value, and is strictly smaller.
  var subXs = [Math.log10(10), Math.log10(5000), Math.log10(25000)];
  var manual = 0;
  subXs.forEach(function (x) { manual += 1 / (1 + Math.exp(-(a - b * x))); });
  approx(M.computeSizeEligible(a, b, subXs), manual, 1e-9, 'subset size equals hand-computed sum');
  ok(M.computeSizeEligible(a, b, subXs) < M.computeSize(a, b), 'subset size strictly below full size');

  // fitEstimator threads eligibleXs through: size and BOTH CI bounds are
  // bounded by the eligible count, eligibleCount is reported, and the same
  // data with a smaller eligible set yields a smaller size.
  var data = genDataset(3.2, 1.1, 0.1, 0.4, { seed: 11, perBandYesNo: 15, nMc: 60, nPseudo: 35 });
  var eligible = [];
  for (var r2 = 1; r2 <= M.MAX_RANK; r2++) if (r2 % 3 !== 0) eligible.push(Math.log10(r2)); // ~2/3 eligible
  var full = M.fitEstimator(data, { rng: M.mulberry32(2), bootstrapReps: 40 });
  var elig = M.fitEstimator(data, { rng: M.mulberry32(2), bootstrapReps: 40, eligibleXs: eligible });
  ok(elig.size < full.size, 'eligible-ranks size below full-ranks size on identical data');
  ok(elig.size <= eligible.length, 'size bounded by eligible count');
  ok(elig.ciLo <= elig.size && elig.size <= elig.ciHi, 'point estimate inside CI (eligible mode)');
  ok(elig.ciHi <= eligible.length + 1e-9, 'CI upper bound bounded by eligible count');
  ok(elig.eligibleCount === eligible.length, 'eligibleCount reported');
  ok(full.eligibleCount === M.MAX_RANK, 'eligibleCount defaults to MAX_RANK without a blocklist');
})();


/* ---------------------------------------------------------------------------
   eligibleIndexForRank — the rank -> eligible-word-index mapping the
   per-sitting curve chart's re-indexed x-axis is built on (2026-08-24,
   the author: "could the graph itself just go up to the true number of words in
   the pool?"). Edge cases: below the first eligible rank -> 0, at/past the
   last eligible rank -> the full eligible count, and monotone non-decreasing
   across an increasing sequence of ranks (including ranks that fall exactly
   ON an eligible rank, and ranks far past MAX_RANK).
   ------------------------------------------------------------------------ */
(function testEligibleIndexForRank() {
  // Five eligible ranks: 2, 5, 20, 500, 4000 (everything else blocklisted).
  var xs = [Math.log10(2), Math.log10(5), Math.log10(20), Math.log10(500), Math.log10(4000)];

  ok(M.eligibleIndexForRank(xs, 1) === 0, 'rank below the first eligible rank -> 0');
  ok(M.eligibleIndexForRank(xs, 2) === 1, 'rank AT the first eligible rank -> 1 (inclusive)');
  ok(M.eligibleIndexForRank(xs, 4) === 1, 'rank between two eligible ranks holds at the lower count');
  ok(M.eligibleIndexForRank(xs, 5) === 2, 'rank AT the second eligible rank -> 2');
  ok(M.eligibleIndexForRank(xs, 4000) === xs.length, 'rank AT the last eligible rank -> the full eligible count (inclusive)');
  ok(M.eligibleIndexForRank(xs, 30000) === xs.length, 'rank past the last eligible rank -> the full eligible count');
  ok(M.eligibleIndexForRank(xs, 1000000) === xs.length, 'a rank far past MAX_RANK still -> the full eligible count, no throw');

  // Empty eligibleXs (a fully-blocklisted / not-yet-loaded language) never
  // throws and always reads as index 0.
  ok(M.eligibleIndexForRank([], 1) === 0, 'empty eligibleXs -> 0 at rank 1');
  ok(M.eligibleIndexForRank([], 30000) === 0, 'empty eligibleXs -> 0 at any rank');

  // Monotone non-decreasing across an increasing sequence of ranks, including
  // several ranks that land exactly ON an eligible entry.
  var prev = -1;
  [1, 2, 3, 4, 5, 6, 19, 20, 21, 499, 500, 501, 3999, 4000, 4001, 999999].forEach(function (r) {
    var v = M.eligibleIndexForRank(xs, r);
    ok(v >= prev, 'monotone non-decreasing at rank ' + r + ' (got ' + v + ', prev ' + prev + ')');
    prev = v;
  });

  // Realistic scale: half of MAX_RANK eligible (every even rank blocklisted
  // out) — index count matches a hand count, and doubles roughly in step.
  var half = [];
  for (var r2 = 1; r2 <= M.MAX_RANK; r2 += 2) half.push(Math.log10(r2)); // 1,3,5,...,29999
  ok(M.eligibleIndexForRank(half, 1) === 1, 'half-density list: rank 1 (odd, eligible) -> index 1');
  ok(M.eligibleIndexForRank(half, 100) === 50, 'half-density list: rank 100 -> 50 eligible odd ranks below it');
  ok(M.eligibleIndexForRank(half, M.MAX_RANK) === half.length, 'half-density list: MAX_RANK -> full eligible count');
})();

/* ---------------------------------------------------------------------------
   comparableEstimate — the cross-language "comparable score" (2026-08-24).
   Exact regression values against two of two synthetic curves, computed independently outside
   this harness and pinned here so a future change to the summation can't
   drift silently. Also: monotonic in depth, and the b=0 degenerate case
   collapses to a flat rate times depth.
   ------------------------------------------------------------------------ */
(function testComparableEstimate() {
  ok(M.COMPARE_DEPTH === 16000, 'COMPARE_DEPTH is 16000 (got ' + M.COMPARE_DEPTH + ')');

  approx(M.comparableEstimate({ a: 8.8, b: 1.85 }, 16000), 13433, 0,
    'comparableEstimate regression value #1 (synthetic curve #1)');
  approx(M.comparableEstimate({ a: 8.9, b: 1.75 }, 16000), 14306, 0,
    'comparableEstimate regression value #2 (synthetic curve #2)');

  // Monotonically non-decreasing in depth: every added rank contributes
  // p_know >= 0, so summing further ranks can never lower the total.
  var curve = { a: 3.0, b: 1.2 };
  var prev = 0;
  [1, 10, 100, 500, 1000, 4000, 8000, 16000].forEach(function (d) {
    var v = M.comparableEstimate(curve, d);
    ok(v >= prev, 'comparableEstimate is non-decreasing in depth (depth=' + d + ': got ' + v + ', prev=' + prev + ')');
    prev = v;
  });

  // b=0: p_know is the SAME sigmoid(a) at every rank (no steepness), so the
  // sum collapses to depth * sigmoid(a).
  var aFlat = 1.5, depthFlat = 16000;
  var flatCurve = { a: aFlat, b: 0 };
  approx(M.comparableEstimate(flatCurve, depthFlat), Math.round(depthFlat * M.sigmoid(aFlat)), 0,
    'comparableEstimate with b=0 collapses to round(depth * sigmoid(a))');

  // Default depth (no second argument) falls back to COMPARE_DEPTH.
  var withDefault = M.comparableEstimate({ a: 8.8, b: 1.85 });
  var withExplicit = M.comparableEstimate({ a: 8.8, b: 1.85 }, M.COMPARE_DEPTH);
  ok(withDefault === withExplicit,
    'comparableEstimate defaults depth to COMPARE_DEPTH when omitted (got ' + withDefault + ', want ' + withExplicit + ')');
})();

/* ---------------------------------------------------------------------------
   Availability/cap-aware allocation + the invalid-sitting guard
   (2026-08-23, after the author's inverted pt sitting).
   ------------------------------------------------------------------------ */
(function testAllocAvailabilityAndCap() {
  // Degenerate weights concentrated on band 0 (the corrupted-sitting shape):
  // without the cap this piled ~48 onto band 0; with avail 40 the shortfall
  // used to vanish (52-item sitting). Now: band 0 capped at 20, total still 60.
  var w = [1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6];
  var avail = [40, 60, 90, 120, 140, 140, 110];
  var out = M.allocateBandCounts(w, 60, 2, avail, M.MAX_MC_PER_BAND);
  ok(out.reduce(function (s, x) { return s + x; }, 0) === 60, 'cap case: total is exactly 60 (got ' + out.join(',') + ')');
  ok(out[0] === 20, 'cap case: hot band capped at MAX_MC_PER_BAND (got ' + out[0] + ')');
  ok(out.every(function (x) { return x >= 2; }), 'cap case: min 2 everywhere');

  // Availability shortage in the hot band redistributes rather than vanishing.
  var out2 = M.allocateBandCounts(w, 60, 2, [8, 60, 90, 120, 140, 140, 110], M.MAX_MC_PER_BAND);
  ok(out2.reduce(function (s, x) { return s + x; }, 0) === 60, 'avail case: total is exactly 60 (got ' + out2.join(',') + ')');
  ok(out2[0] === 8, 'avail case: band 0 limited to its 8 available');

  // Total capacity below 60: allocate everything available, never more.
  var out3 = M.allocateBandCounts(w, 60, 2, [3, 3, 3, 3, 3, 3, 3], M.MAX_MC_PER_BAND);
  ok(out3.reduce(function (s, x) { return s + x; }, 0) === 21, 'shortage case: sums to total capacity 21');
  // Backward compat: the old 3-arg call still returns exactly 60 with min 2.
  var out4 = M.allocateBandCounts([1, 2, 3, 4, 3, 2, 1], 60, 2);
  ok(out4.reduce(function (s, x) { return s + x; }, 0) === 60 && out4.every(function (x) { return x >= 2; }),
    'compat: 3-arg call still sums to 60 with min 2');
})();

(function testSittingLooksInvalid() {
  ok(M.sittingLooksInvalid(35, 35) === true, 'guard: 35/35 fakes claimed is invalid');
  ok(M.sittingLooksInvalid(35, 21) === true, 'guard: 60% claimed is invalid (threshold inclusive)');
  ok(M.sittingLooksInvalid(35, 20) === false, 'guard: 57% claimed is not invalid');
  ok(M.sittingLooksInvalid(35, 2) === false, 'guard: honest low claims are fine');
  ok(M.sittingLooksInvalid(5, 5) === false, 'guard: tiny pools never trip it');
  ok(M.sittingLooksInvalid(0, 0) === false, 'guard: empty pool never trips it');
})();


/* ---------------------------------------------------------------------------
   Per-language band edges (2026-08-23, for Armenian's honest sub-30k depth):
   bandLoHi takes an optional edges array; the bare call keeps the default.
   ------------------------------------------------------------------------ */
(function testCustomBandEdges() {
  var hyEdges = [500, 1000, 2000, 4000, 8000, 14000, 20000];
  ok(M.bandLoHi(0)[0] === 1 && M.bandLoHi(0)[1] === 500, 'default edges band 0 unchanged');
  ok(M.bandLoHi(6)[1] === 30000, 'default edges band 6 tops at 30000');
  ok(M.bandLoHi(5, hyEdges)[0] === 8001 && M.bandLoHi(5, hyEdges)[1] === 14000, 'custom edges band 5 = 8001-14000');
  ok(M.bandLoHi(6, hyEdges)[0] === 14001 && M.bandLoHi(6, hyEdges)[1] === 20000, 'custom edges band 6 = 14001-20000');
})();

console.log('\n' + PASS + ' passed, ' + FAIL + ' failed');
if (FAIL) process.exit(1);
