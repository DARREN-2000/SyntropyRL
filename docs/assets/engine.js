/* syntropyrl engine - browser port
 *
 * This is a deliberate, faithful port of src/syntropyrl/{stats,core,simulate,
 * detectors}.py so the playground needs no backend. Every threshold below is
 * the same number the Python package uses. If you change one here, change it
 * there too (CONTRIBUTING.md says so, and it matters: a playground that
 * disagrees with the package is worse than no playground).
 *
 * No dependencies, no build step, no framework. Same rule as the Python side.
 */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------ rng
  // mulberry32: small, fast, deterministic across browsers.
  function Rng(seed) {
    this.s = (seed >>> 0) || 1;
    this._spare = null;
  }
  Rng.prototype.next = function () {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    var t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Rng.prototype.gauss = function (mu, sigma) {
    if (this._spare !== null) {
      var v = this._spare;
      this._spare = null;
      return mu + sigma * v;
    }
    var u1 = Math.max(this.next(), 1e-12);
    var u2 = this.next();
    var r = Math.sqrt(-2 * Math.log(u1));
    this._spare = r * Math.sin(2 * Math.PI * u2);
    return mu + sigma * r * Math.cos(2 * Math.PI * u2);
  };
  Rng.prototype.choice = function (arr) {
    return arr[Math.floor(this.next() * arr.length) % arr.length];
  };

  // ---------------------------------------------------------------- stats
  var S = {};
  S.finite = function (xs) {
    var out = [];
    for (var i = 0; i < xs.length; i++) {
      var v = xs[i];
      if (typeof v === "number" && isFinite(v)) out.push(v);
    }
    return out;
  };
  S.mean = function (xs) {
    var f = S.finite(xs);
    if (!f.length) return null;
    var t = 0;
    for (var i = 0; i < f.length; i++) t += f[i];
    return t / f.length;
  };
  S.std = function (xs) {
    var f = S.finite(xs);
    if (f.length < 2) return 0;
    var m = S.mean(f), t = 0;
    for (var i = 0; i < f.length; i++) t += (f[i] - m) * (f[i] - m);
    return Math.sqrt(t / (f.length - 1));
  };
  S.median = function (xs) {
    var f = S.finite(xs).slice().sort(function (a, b) { return a - b; });
    if (!f.length) return null;
    var mid = Math.floor(f.length / 2);
    return f.length % 2 ? f[mid] : (f[mid - 1] + f[mid]) / 2;
  };
  S.slope = function (xs) {
    var f = S.finite(xs);
    var n = f.length;
    if (n < 3) return 0;
    var mx = (n - 1) / 2, my = S.mean(f), num = 0, den = 0;
    for (var i = 0; i < n; i++) {
      num += (i - mx) * (f[i] - my);
      den += (i - mx) * (i - mx);
    }
    return den === 0 ? 0 : num / den;
  };
  S.pearson = function (xs, ys) {
    var ax = [], ay = [];
    for (var i = 0; i < Math.min(xs.length, ys.length); i++) {
      if (isFinite(xs[i]) && isFinite(ys[i])) { ax.push(xs[i]); ay.push(ys[i]); }
    }
    if (ax.length < 3) return null;
    var mx = S.mean(ax), my = S.mean(ay), num = 0, dx = 0, dy = 0;
    for (var j = 0; j < ax.length; j++) {
      var a = ax[j] - mx, b = ay[j] - my;
      num += a * b; dx += a * a; dy += b * b;
    }
    if (dx <= 0 || dy <= 0) return null;
    return num / Math.sqrt(dx * dy);
  };
  S.relativeChange = function (xs, head, tail) {
    var f = S.finite(xs);
    if (f.length < head + tail) return null;
    var a = S.mean(f.slice(0, head)), b = S.mean(f.slice(-tail));
    if (a === null || b === null) return null;
    return { from: a, to: b, delta: b - a, rel: Math.abs(a) > 1e-9 ? (b - a) / Math.abs(a) : null };
  };
  S.splitHalves = function (xs) {
    var f = S.finite(xs);
    if (f.length < 6) return null;
    var h = Math.floor(f.length / 2);
    return { first: S.mean(f.slice(0, h)), second: S.mean(f.slice(h)) };
  };
  var WORD_RE = /[a-z0-9']+/g;
  S.words = function (text) { return (text || "").toLowerCase().match(WORD_RE) || []; };
  S.ngrams = function (text, n) {
    var w = S.words(text), out = [];
    for (var i = 0; i + n <= w.length; i++) out.push(w.slice(i, i + n).join(" "));
    return out;
  };
  S.ngramShare = function (texts, gram, n) {
    n = n || 4;
    if (!texts.length) return 0;
    var hits = 0;
    for (var i = 0; i < texts.length; i++) {
      if (S.ngrams(texts[i], n).indexOf(gram) !== -1) hits++;
    }
    return hits / texts.length;
  };
  S.duplicateShare = function (texts) {
    if (texts.length < 2) return 0;
    var seen = {}, dup = 0;
    for (var i = 0; i < texts.length; i++) {
      var k = (texts[i] || "").trim();
      if (seen[k]) dup++; else seen[k] = 1;
    }
    return dup / texts.length;
  };
  /* Find the n-gram whose PRESENCE predicts higher reward, ranked by a
     Welch-style standardized effect. Selecting on the raw gap alone finds a
     "winner" in healthy runs by chance - we learned that the hard way, and the
     t >= 6 gate in RLD-031 is the fix. */
  S.rewardCorrelatedNgram = function (texts, rewards, n, candidates, minGroup) {
    n = n || 4; candidates = candidates || 40; minGroup = minGroup || 5;
    if (texts.length !== rewards.length || texts.length < 2 * minGroup) return null;
    // Precompute one gram set per text: the naive version re-tokenizes every
    // text for every candidate, which is ~120k string ops per call and the
    // browser feels it at 300 steps.
    var counts = {};
    var sets = [];
    for (var i = 0; i < texts.length; i++) {
      var set = {}, gs = S.ngrams(texts[i], n);
      for (var j = 0; j < gs.length; j++) {
        if (!set[gs[j]]) { set[gs[j]] = 1; counts[gs[j]] = (counts[gs[j]] || 0) + 1; }
      }
      sets.push(set);
    }
    var keys = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    var best = null, tested = 0;
    // Mirrors stats.reward_correlated_ngram in the package. A phrase present in
    // 95% of rollouts is the domain's vocabulary, and the tiny complement left
    // over often has zero reward variance, which fakes an enormous t. Require
    // both sides to be a real share of the batch, and floor each side's variance
    // at 10% of the batch's own reward spread.
    var groupFloor = Math.max(minGroup, Math.ceil(0.15 * texts.length));
    var allStd = S.std(rewards) || 0;
    var varFloor = (0.10 * allStd) * (0.10 * allStd);
    for (var k = 0; k < keys.length && tested < candidates; k++) {
      var gram = keys[k];
      if (counts[gram] < groupFloor || texts.length - counts[gram] < groupFloor) continue;
      tested++;
      var withR = [], without = [];
      for (var t = 0; t < texts.length; t++) {
        (sets[t][gram] ? withR : without).push(rewards[t]);
      }
      var m1 = S.mean(withR), m0 = S.mean(without);
      if (m1 === null || m0 === null) continue;
      var delta = m1 - m0;
      if (delta <= 0) continue;
      var s1 = S.std(withR) || 0, s0 = S.std(without) || 0;
      var se = Math.sqrt(
        Math.max(s1 * s1, varFloor) / withR.length +
        Math.max(s0 * s0, varFloor) / without.length
      );
      var tstat = se > 1e-9 ? delta / se : (delta > 0 ? 999 : 0);
      if (!best || tstat > best.t) {
        best = {
          gram: gram, delta: delta, t: tstat, with: m1, without: m0,
          prevalence: counts[gram] / texts.length, tested: tested,
          nWith: withR.length, nWithout: without.length
        };
      }
    }
    return best;
  };
  S.fmt = function (v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "n/a";
    return v.toFixed(d === undefined ? 3 : d);
  };
  S.pct = function (v) {
    if (v === null || v === undefined || !isFinite(v)) return "n/a";
    return (v * 100).toFixed(1) + "%";
  };

  // ----------------------------------------------------------------- core
  var SEVERITY = { info: 0, warn: 1, critical: 2 };
  var HEALTHY_GAP = 1e-3;

  function Rollout(o) {
    this.reward = o.reward;
    this.completion = o.completion || "";
    this.nTokens = o.nTokens || S.words(this.completion).length;
    this.truncated = !!o.truncated;
    this.correct = o.correct === undefined ? null : o.correct;
    this.groupId = o.groupId === undefined ? null : o.groupId;
    this.rolloutLogprobs = o.rolloutLogprobs || null;
    this.trainerLogprobs = o.trainerLogprobs || null;
  }
  Rollout.prototype.logprobGap = function () {
    if (!this.rolloutLogprobs || !this.trainerLogprobs) return null;
    var n = Math.min(this.rolloutLogprobs.length, this.trainerLogprobs.length);
    if (!n) return null;
    var t = 0;
    for (var i = 0; i < n; i++) t += Math.abs(this.rolloutLogprobs[i] - this.trainerLogprobs[i]);
    return t / n;
  };

  function Step(step, rollouts, metrics) {
    this.step = step;
    this.rollouts = rollouts || [];
    this.metrics = metrics || {};
  }
  Step.prototype.merged = function () {
    var m = {}, k;
    for (k in this.metrics) if (this.metrics.hasOwnProperty(k)) m[k] = this.metrics[k];
    var rs = this.rollouts;
    if (!rs.length) return m;
    var rewards = rs.map(function (r) { return r.reward; });
    var lens = rs.map(function (r) { return r.nTokens; });
    var texts = rs.map(function (r) { return r.completion; });
    if (m.reward_mean === undefined) m.reward_mean = S.mean(rewards);
    if (m.reward_std === undefined) m.reward_std = S.std(rewards);
    if (m.seq_len_mean === undefined) m.seq_len_mean = S.mean(lens);
    if (m.truncated_frac === undefined) {
      m.truncated_frac = rs.filter(function (r) { return r.truncated; }).length / rs.length;
    }
    var graded = rs.filter(function (r) { return r.correct !== null; });
    if (m.accuracy === undefined && graded.length) {
      m.accuracy = graded.filter(function (r) { return r.correct; }).length / graded.length;
    }
    var gaps = [];
    for (var i = 0; i < rs.length; i++) {
      var g = rs[i].logprobGap();
      if (g !== null) gaps.push(g);
    }
    if (m.logprob_gap === undefined && gaps.length) m.logprob_gap = S.mean(gaps);
    // group-relative advantages, GRPO style
    var groups = {};
    for (var j = 0; j < rs.length; j++) {
      var gid = rs[j].groupId === null ? "_" : rs[j].groupId;
      (groups[gid] = groups[gid] || []).push(rs[j].reward);
    }
    var advs = [], degenerate = 0, nGroups = 0;
    for (var gk in groups) {
      if (!groups.hasOwnProperty(gk)) continue;
      nGroups++;
      var vals = groups[gk], gm = S.mean(vals), gs = S.std(vals);
      if (gs < 1e-6) degenerate++;
      for (var v = 0; v < vals.length; v++) advs.push(gs > 1e-9 ? (vals[v] - gm) / gs : 0);
    }
    if (m.advantage_std === undefined) m.advantage_std = S.std(advs);
    if (m.degenerate_group_frac === undefined && nGroups) m.degenerate_group_frac = degenerate / nGroups;
    if (m.duplicate_frac === undefined) m.duplicate_frac = S.duplicateShare(texts);
    return m;
  };
  Step.prototype.topRollouts = function (k) {
    return this.rollouts.slice().sort(function (a, b) { return b.reward - a.reward; }).slice(0, k || 8);
  };

  function RunHistory(windowSize) {
    this.window = windowSize || 250;
    this._steps = [];
    this._series = {};
  }
  RunHistory.prototype.observe = function (step) {
    this._steps.push(step);
    if (this._steps.length > this.window) this._steps.shift();
    var m = step.merged();
    for (var k in m) {
      if (!m.hasOwnProperty(k)) continue;
      var v = m[k];
      if (typeof v !== "number" || !isFinite(v)) continue;
      var arr = this._series[k] = this._series[k] || [];
      arr.push(v);
      if (arr.length > this.window) arr.shift();
    }
  };
  RunHistory.prototype.series = function (name) { return this._series[name] || []; };
  RunHistory.prototype.has = function (name) { return (this._series[name] || []).length > 0; };
  RunHistory.prototype.length = function () { return this._steps.length; };
  RunHistory.prototype.steps = function () { return this._steps; };
  RunHistory.prototype.lastStep = function () { return this._steps[this._steps.length - 1]; };
  RunHistory.prototype.recentRollouts = function (n) {
    var out = [];
    for (var i = this._steps.length - 1; i >= 0 && out.length < n; i--) {
      var rs = this._steps[i].rollouts;
      for (var j = 0; j < rs.length && out.length < n; j++) out.push(rs[j]);
    }
    return out;
  };

  // ------------------------------------------------------------ detectors
  // Each detector: { code, title, family, minSteps, cooldown, requires, check }
  // check(history) returns null or a diagnosis object.
  function dx(det, h, severity, summary, evidence, causes, fixes) {
    return {
      code: det.code, title: det.title, family: det.family,
      severity: severity, summary: summary, evidence: evidence || {},
      causes: causes || [], fixes: fixes || [],
      step: h.lastStep() ? h.lastStep().step : h.length(),
      url: "./atlas.html#" + det.code
    };
  }
  function tail(arr, n) { return arr.slice(Math.max(0, arr.length - n)); }

  var DETECTORS = [
    {
      code: "RLD-014", title: "Trainer/generator logprob divergence", family: "numerics",
      minSteps: 5, cooldown: 60, requires: ["logprob_gap"],
      check: function (h) {
        var gaps = h.series("logprob_gap");
        if (gaps.length < 5) return null;
        var recent = tail(gaps, 10), m = S.mean(recent);
        if (m === null || m <= 3 * HEALTHY_GAP) return null;
        var sev = m > 10 * HEALTHY_GAP ? "critical" : "warn";
        var rs = h.recentRollouts(64).filter(function (r) { return r.logprobGap() !== null; });
        var lenCorr = rs.length >= 8
          ? S.pearson(rs.map(function (r) { return r.nTokens; }), rs.map(function (r) { return r.logprobGap(); }))
          : null;
        var causes = [
          "bf16/fp16 generation compared against a different-precision recomputation",
          "Sampling parameters applied in the rollout path but not the training path",
          "Tensor or sequence parallel reductions changing summation order"
        ];
        if (lenCorr !== null && lenCorr > 0.4) {
          causes.unshift("Gap grows with sequence length (r=" + S.fmt(lenCorr, 2) +
            "), which points at chunked prefill or paged attention numerics");
        }
        return dx(this, h, sev,
          "Mean |logprob| gap between generator and trainer is " + S.fmt(m, 5) +
          ", " + (m / HEALTHY_GAP).toFixed(1) + "x the healthy ceiling of " + S.fmt(HEALTHY_GAP, 5) +
          ". Your importance ratios are computed against logprobs the trainer never produced.",
          { mean_gap: m, healthy_ceiling: HEALTHY_GAP, ratio: m / HEALTHY_GAP, length_correlation: lenCorr },
          causes,
          ["Run syntropyrl.check_logprob_parity() before training and assert verdict == 'pass'",
           "Disable chunked prefill in the rollout engine, re-measure, then re-enable and compare",
           "Use the generator's logprobs as the behaviour policy instead of recomputing them",
           "Log this gap as a permanent training metric - it is cheap and load-bearing"]);
      }
    },
    {
      code: "RLD-092", title: "Chat template / tokenization mismatch", family: "numerics",
      minSteps: 5, cooldown: 80, requires: [],
      check: function (h) {
        var HEAD = 8;
        var rs = h.recentRollouts(48).filter(function (r) {
          return r.rolloutLogprobs && r.trainerLogprobs &&
            Math.min(r.rolloutLogprobs.length, r.trainerLogprobs.length) >= 24;
        });
        if (rs.length < 6) return null;
        var headGaps = [], tailGaps = [];
        for (var i = 0; i < rs.length; i++) {
          var a = rs[i].rolloutLogprobs, b = rs[i].trainerLogprobs;
          var n = Math.min(a.length, b.length), hs = 0, ts = 0, tc = 0;
          for (var j = 0; j < n; j++) {
            var d = Math.abs(a[j] - b[j]);
            if (j < HEAD) hs += d; else { ts += d; tc++; }
          }
          headGaps.push(hs / HEAD);
          if (tc) tailGaps.push(ts / tc);
        }
        var hm = S.mean(headGaps), tm = S.mean(tailGaps);
        if (hm === null || tm === null) return null;
        if (hm <= 5 * HEALTHY_GAP) return null;
        var ratio = tm > 1e-9 ? hm / tm : 999;
        if (ratio < 8) return null;
        return dx(this, h, "critical",
          "The first " + HEAD + " tokens diverge " + ratio.toFixed(1) + "x more than the rest of the " +
          "sequence (" + S.fmt(hm, 5) + " vs " + S.fmt(tm, 5) + "). That shape is a prompt-boundary bug, " +
          "not a kernel difference.",
          { head_gap: hm, tail_gap: tm, ratio: ratio, samples: rs.length },
          ["Generation uses apply_chat_template but the trainer tokenizes raw text (or vice versa)",
           "BOS token added twice, or dropped on one side",
           "add_special_tokens defaulting differently between the two call sites",
           "A system prompt present at rollout time and absent at training time"],
          ["Print decoded token IDs from both paths for one sample and diff them character by character",
           "Tokenize once, at rollout time, and pass token IDs to the trainer instead of strings",
           "Assert trainer token count equals generator token count for every sample in batch 1"]);
      }
    },
    {
      code: "RLD-071", title: "Gradient pathology", family: "numerics",
      minSteps: 10, cooldown: 40, requires: ["grad_norm"],
      check: function (h) {
        var raw = h.series("grad_norm");
        if (raw.length < 10) return null;
        var last = h.lastStep() ? h.lastStep().metrics.grad_norm : null;
        if (last !== null && last !== undefined && !isFinite(last)) {
          return dx(this, h, "critical",
            "Gradient norm is not finite. Training is over; every checkpoint after this step is suspect.",
            { grad_norm: String(last) },
            ["fp16 overflow in the logprob computation", "Advantage outliers from a single extreme reward",
             "Empty or single-token completions producing degenerate ratios"],
            ["Switch to bf16", "Clip gradients (max_norm=1.0) and clip advantages before the loss",
             "Filter empty completions out of the batch"]);
        }
        var recent = tail(raw, 12), med = S.median(raw), peak = Math.max.apply(null, recent);
        if (med === null || med <= 0) return null;
        if (peak > 20 * med && peak > 1.0) {
          return dx(this, h, "warn",
            "Gradient norm spiked to " + S.fmt(peak, 2) + ", " + (peak / med).toFixed(0) +
            "x the running median of " + S.fmt(med, 3) + ". One spike like this can undo thousands of good steps.",
            { peak: peak, median: med, ratio: peak / med },
            ["Advantage outliers from a single very high or very low reward",
             "Unnormalized advantages combined with a large learning rate",
             "A malformed batch (empty completions, length-1 sequences)"],
            ["Clip gradients (max_norm=1.0) and clip advantages before the loss",
             "Reduce learning rate by 3x and check whether the spike disappears",
             "Log the batch that produced the spike and read it"]);
        }
        return null;
      }
    },
    {
      code: "RLD-001", title: "Advantage collapse (degenerate groups)", family: "optimization",
      minSteps: 8, cooldown: 50, requires: ["degenerate_group_frac"],
      check: function (h) {
        var dg = h.series("degenerate_group_frac");
        if (dg.length < 8) return null;
        var dead = S.mean(tail(dg, 10));
        var adv = h.has("advantage_std") ? S.mean(tail(h.series("advantage_std"), 10)) : null;
        if (dead === null) return null;
        if (dead < 0.55 && (adv === null || adv > 0.15)) return null;
        var sev = dead > 0.75 ? "critical" : "warn";
        return dx(this, h, sev,
          S.pct(dead) + " of GRPO groups have identical rewards, so their advantages are exactly zero" +
          (adv !== null ? " (advantage std " + S.fmt(adv, 3) + ")" : "") +
          ". You are paying for rollouts that carry no gradient.",
          { degenerate_group_frac: dead, advantage_std: adv },
          ["Prompts far too hard: every rollout fails, so every advantage is zero",
           "Prompts far too easy: every rollout succeeds, so every advantage is zero",
           "Binary reward with a small group size, so ties dominate",
           "Group size of 1, which makes group-relative advantage meaningless"],
          ["Filter prompts to pass rates between 0.2 and 0.8 for the current policy",
           "Increase group size (8-16) so ties are less likely",
           "Add partial credit so the reward is not purely binary",
           "Re-score prompt difficulty every N steps and resample"]);
      }
    },
    {
      code: "RLD-021", title: "KL blowup", family: "optimization",
      minSteps: 12, cooldown: 50, requires: ["kl"],
      check: function (h) {
        var kl = h.series("kl");
        if (kl.length < 12) return null;
        var early = S.mean(kl.slice(0, Math.min(8, Math.floor(kl.length / 3)))) || 0;
        var recent = S.mean(tail(kl, 6));
        if (recent === null || recent < 0.5) return null;
        var growth = early > 1e-6 ? recent / early : 999;
        if (growth < 5 && recent < 2.0) return null;
        var sev = recent > 2.0 ? "critical" : "warn";
        return dx(this, h, sev,
          "KL to the reference policy is " + S.fmt(recent, 3) + ", up " +
          (growth > 900 ? "from ~0" : growth.toFixed(1) + "x") + " from " + S.fmt(early, 3) +
          " early in the run. The policy is escaping the trust region and output quality follows.",
          { kl_recent: recent, kl_early: early, growth: growth },
          ["KL coefficient too small, or the KL penalty accidentally disabled",
           "Learning rate too high for the batch size",
           "Reward model being exploited off-distribution, pulling the policy away fast",
           "Reference model updated or reloaded mid-run"],
          ["Raise the KL coefficient, or use adaptive KL targeting a fixed value",
           "Lower the learning rate (RL post-training usually wants 1e-6, not 1e-5)",
           "Add reward clipping so one exploitable prompt cannot dominate",
           "Verify the reference model is frozen and is the model you think it is"]);
      }
    },
    {
      code: "RLD-055", title: "Clip saturation (stale off-policy data)", family: "optimization",
      minSteps: 10, cooldown: 50, requires: ["clip_frac"],
      check: function (h) {
        var cf = h.series("clip_frac");
        if (cf.length < 10) return null;
        var m = S.mean(tail(cf, 10));
        if (m === null || m < 0.3) return null;
        var sev = m > 0.5 ? "critical" : "warn";
        return dx(this, h, sev,
          S.pct(m) + " of your batch is being clipped (healthy is 2-20%). Most of the gradient you paid " +
          "to generate is being discarded before it reaches the weights.",
          { clip_frac: m, healthy_range: "0.02-0.20" },
          ["Too many optimizer epochs per rollout batch, so late updates are far off-policy",
           "Rollout batch reused across many minibatches without recomputing logprobs",
           "Learning rate too high, pushing the ratio outside the trust region immediately",
           "Asynchronous rollout workers lagging several policy versions behind"],
          ["Reduce inner epochs to 1 or 2", "Reduce minibatches per rollout batch",
           "Bound rollout staleness explicitly in async setups",
           "Lower the learning rate and re-check clip_frac before changing anything else"]);
      }
    },
    {
      code: "RLD-084", title: "Value function divergence", family: "optimization",
      minSteps: 20, cooldown: 60, requires: ["value_loss"],
      check: function (h) {
        var vl = h.series("value_loss");
        if (vl.length < 20) return null;
        var halves = S.splitHalves(vl);
        if (!halves || halves.first === null || halves.first <= 1e-9) return null;
        var ratio = halves.second / halves.first;
        var sl = S.slope(tail(vl, 30));
        if (ratio <= 1.5 || sl <= 0) return null;
        return dx(this, h, "warn",
          "Critic loss is diverging: " + S.fmt(halves.second, 3) + " in the second half of the window " +
          "versus " + S.fmt(halves.first, 3) + " in the first (" + ratio.toFixed(2) + "x, slope " +
          S.fmt(sl, 4) + "). Every advantage you compute is getting noisier.",
          { first_half: halves.first, second_half: halves.second, ratio: ratio, slope: sl },
          ["Value head learning rate too high, or a shared trunk fighting the policy objective",
           "Reward scale changed mid-run",
           "Value targets computed with a different discount or GAE lambda than intended",
           "Critic initialized from a model that never saw this reward distribution"],
          ["Normalize rewards or value targets, and keep it consistent",
           "Give the value head its own lower learning rate",
           "Warm up the critic before enabling policy updates",
           "Consider a critic-free method (GRPO) if the critic keeps fighting you"]);
      }
    },
    {
      code: "RLD-031", title: "Reward hacking (proxy/truth divergence)", family: "reward",
      minSteps: 25, cooldown: 60, requires: ["reward_mean"],
      check: function (h) {
        var rewards = h.series("reward_mean");
        if (rewards.length < 25) return null;
        // Compare the two halves of the whole window with a *relative* change,
        // exactly like the package. A last-10-vs-first-10 absolute delta is far
        // noisier and fired on healthy runs.
        var rh = S.splitHalves(rewards);
        if (!rh) return null;
        var rRel = (rh.second - rh.first) / Math.max(Math.abs(rh.first), 1e-8);
        if (rRel < 0.12) return null;
        var rc = { from: rh.first, to: rh.second };

        var truthName = null, truthRel = null, truthFrom = null, truthTo = null;
        var candidates = ["accuracy", "task_accuracy", "eval_accuracy", "pass_rate"];
        for (var i = 0; i < candidates.length; i++) {
          if (h.has(candidates[i])) {
            var th = S.splitHalves(h.series(candidates[i]));
            if (th) {
              truthName = candidates[i];
              truthFrom = th.first;
              truthTo = th.second;
              truthRel = (th.second - th.first) / Math.max(Math.abs(th.first), 1e-8);
              break;
            }
          }
        }
        // Flat *or falling* ground truth counts as divergence.
        var truthFlat = truthRel !== null && truthRel < 0.04;

        var rs = h.recentRollouts(96);
        var texts = rs.map(function (r) { return r.completion; });
        var rw = rs.map(function (r) { return r.reward; });
        // Range, not standard deviation: the threshold is a fraction of the
        // batch's reward spread, and std understates that by roughly 2.5x.
        var spread = rw.length ? (Math.max.apply(null, rw) - Math.min.apply(null, rw)) : 0;
        var best = S.rewardCorrelatedNgram(texts, rw, 4, 40, 5);
        // A large raw gap is not enough: we searched ~40 candidates, so the
        // biggest gap is nonzero by chance. Require a standardized effect.
        var phraseExploit = !!(best && spread > 0 && best.t >= 6.0 &&
          best.delta >= Math.max(0.15, 0.25 * spread));

        if (!truthFlat && !phraseExploit) return null;
        var sev = truthFlat ? "critical" : "warn";

        var parts = ["Reward climbed " + S.fmt(rc.from, 3) + " -> " + S.fmt(rc.to, 3) + "."];
        if (truthFlat) {
          parts.push("Meanwhile " + truthName + " went " + S.fmt(truthFrom, 3) + " -> " +
            S.fmt(truthTo, 3) + ", which is flat. The policy is optimizing your proxy, not the task.");
        }
        if (phraseExploit) {
          parts.push('Rollouts containing "' + best.gram + '" score ' + S.fmt(best.with, 3) +
            " versus " + S.fmt(best.without, 3) + " without it, and " + S.pct(best.prevalence) +
            " of the batch now contains it.");
        }
        var ev = {
          reward_change: rRel,
          truth_metric: truthName,
          truth_change: truthRel
        };
        if (best && phraseExploit) {
          ev.paying_ngram = best.gram;
          ev.reward_with_ngram = best.with;
          ev.reward_without_ngram = best.without;
          ev.ngram_prevalence = best.prevalence;
          ev.ngram_effect_t = best.t;
        }
        return dx(this, h, sev, parts.join(" "), ev,
          ["Verifier matching a prefix or a format marker instead of the answer",
           "Regex or string-equality grader accepting a superset of correct answers",
           "LLM judge rewarding confidence, length, or politeness",
           "Reward model overoptimized off-distribution"],
          ["Hold out a verifier the policy never trains against, and evaluate on it",
           "Read the 20 highest-reward completions by hand - this always works",
           "Add an adversarial unit test: does an empty answer score above zero?",
           "Log proxy reward and ground truth on the same chart, permanently"]);
      }
    },
    {
      code: "RLD-033", title: "Length exploit", family: "reward",
      minSteps: 15, cooldown: 60, requires: ["seq_len_mean"],
      check: function (h) {
        var lens = h.series("seq_len_mean");
        if (lens.length < 15) return null;
        var rs = h.recentRollouts(120);
        if (rs.length < 12) return null;
        var corr = S.pearson(rs.map(function (r) { return r.nTokens; }), rs.map(function (r) { return r.reward; }));
        if (corr === null || corr < 0.45) return null;
        var n = Math.min(10, Math.floor(lens.length / 3));
        var lc = S.relativeChange(lens, n, n);
        if (!lc || lc.rel === null || lc.rel < 0.25) return null;
        var sev = corr > 0.7 && lc.rel > 0.6 ? "critical" : "warn";
        return dx(this, h, sev,
          "Completion length and reward are correlated at r=" + S.fmt(corr, 2) + " while mean length grew " +
          S.pct(lc.rel) + " (" + Math.round(lc.from) + " -> " + Math.round(lc.to) + " tokens). " +
          "The policy found that talking more pays more.",
          { length_reward_correlation: corr, length_growth: lc.rel, from: lc.from, to: lc.to },
          ["Judge or reward model prefers verbose answers",
           "Reward summed over tokens instead of averaged",
           "Repetition earning partial credit from a fuzzy matcher",
           "No length penalty while the context budget still has headroom"],
          ["Normalize reward per sequence, not per token",
           "Add an explicit length penalty or token budget to the reward",
           "Score length-matched pairs to test whether your judge is length-biased"]);
      }
    },
    {
      code: "RLD-060", title: "Dead reward signal", family: "reward",
      minSteps: 10, cooldown: 80, requires: ["reward_std"],
      check: function (h) {
        var stds = h.series("reward_std");
        if (stds.length < 10) return null;
        var m = S.mean(tail(stds, 12));
        if (m === null || m > 1e-4) return null;
        var rm = h.has("reward_mean") ? S.mean(tail(h.series("reward_mean"), 12)) : null;
        return dx(this, h, "critical",
          "Reward standard deviation across the batch is " + S.fmt(m, 6) + " - every rollout is getting " +
          "the same score" + (rm !== null ? " (" + S.fmt(rm, 3) + ")" : "") +
          ". There is no signal here at all; the run is a no-op.",
          { reward_std: m, reward_mean: rm },
          ["Reward function raising an exception swallowed by a bare except",
           "Verifier receiving the wrong field (prompt instead of completion)",
           "All completions truncated before reaching the answer",
           "Reward computed on a padded tensor and averaged away to a constant"],
          ["Unit-test the reward function on three known-good and three known-bad strings",
           "Print one full (prompt, completion, reward) triple every 50 steps and read it",
           "Remove the try/except around the verifier and let it crash loudly"]);
      }
    },
    {
      code: "RLD-042", title: "Truncation bias", family: "reward",
      minSteps: 12, cooldown: 60, requires: ["truncated_frac"],
      check: function (h) {
        var tf = h.series("truncated_frac");
        if (tf.length < 12) return null;
        var frac = S.mean(tail(tf, 10));
        if (frac === null || frac < 0.12) return null;
        var rs = h.recentRollouts(120);
        var trunc = [], full = [];
        for (var i = 0; i < rs.length; i++) (rs[i].truncated ? trunc : full).push(rs[i].reward);
        if (trunc.length < 4 || full.length < 4) return null;
        var mt = S.mean(trunc), mf = S.mean(full);
        if (mt === null || mf === null || Math.abs(mf) < 1e-9) return null;
        var gap = (mf - mt) / Math.abs(mf);
        if (gap < 0.15) return null;
        var sev = frac > 0.3 ? "critical" : "warn";
        return dx(this, h, sev,
          S.pct(frac) + " of rollouts hit the generation limit, and they score " + S.fmt(mt, 3) +
          " against " + S.fmt(mf, 3) + " for completed ones (" + S.pct(gap) + " worse). " +
          "You are teaching the model that thinking gets punished.",
          { truncated_frac: frac, truncated_reward: mt, complete_reward: mf, reward_gap: gap },
          ["max_new_tokens too small for the reasoning the task requires",
           "Truncated sequences scored as incorrect rather than masked out",
           "Reasoning traces growing over training until they hit the ceiling",
           "EOS token never emitted because the template does not teach it"],
          ["Raise max_new_tokens and re-measure the truncation rate",
           "Mask truncated rollouts out of the loss instead of scoring them zero",
           "Or score them deliberately, with a documented penalty",
           "Track the truncation rate as a permanent training metric"]);
      }
    },
    {
      code: "RLD-007", title: "Entropy collapse", family: "distribution",
      minSteps: 15, cooldown: 50, requires: ["entropy"],
      check: function (h) {
        var ent = h.series("entropy");
        if (ent.length < 15) return null;
        var base = S.mean(ent.slice(0, Math.min(10, Math.floor(ent.length / 3))));
        var recent = S.mean(tail(ent, 6));
        if (base === null || recent === null) return null;
        var drop = base - recent;
        if (drop < 0.55 || recent > 0.4) return null;
        var sev = recent < 0.15 ? "critical" : "warn";
        var adv = h.has("advantage_std") ? S.mean(tail(h.series("advantage_std"), 6)) : null;
        return dx(this, h, sev,
          "Policy entropy fell from " + S.fmt(base, 3) + " to " + S.fmt(recent, 3) + " (down " +
          S.fmt(drop, 2) + " nats)" +
          (adv !== null ? ", and within-group advantage std is now " + S.fmt(adv, 3) : "") +
          ". Exploration is nearly dead, and this is close to irreversible.",
          { entropy_baseline: base, entropy_recent: recent, drop: drop, advantage_std: adv },
          ["No entropy bonus, or a coefficient that is far too small",
           "Sampling temperature below 1.0 during rollouts",
           "Learning rate high enough to sharpen the policy faster than it explores",
           "Repeatedly training on a narrow prompt set the policy has already mastered"],
          ["Add or raise the entropy bonus (start around 0.001-0.01)",
           "Sample rollouts at temperature 1.0; save low temperature for evaluation",
           "Increase group size so groups still span different behaviours",
           "Refresh the prompt distribution toward unsolved items"]);
      }
    },
    {
      code: "RLD-018", title: "Mode collapse (duplicate rollouts)", family: "distribution",
      minSteps: 10, cooldown: 60, requires: ["duplicate_frac"],
      check: function (h) {
        var df = h.series("duplicate_frac");
        if (df.length < 10) return null;
        var m = S.mean(tail(df, 8));
        if (m === null || m < 0.35) return null;
        var sev = m > 0.6 ? "critical" : "warn";
        return dx(this, h, sev,
          S.pct(m) + " of rollouts inside each group are byte-identical. GRPO advantages are zero by " +
          "construction, so you are paying for N samples and learning from one.",
          { duplicate_frac: m },
          ["do_sample=False or temperature 0 left on in the rollout config",
           "Same seed used for every sample in the group",
           "top_k=1 or an aggressive top_p flattening the distribution",
           "Severe entropy collapse (RLD-007) reaching its endpoint"],
          ["Check the sampling config first: do_sample=True, temperature=1.0, top_p >= 0.95",
           "Ensure each rollout in a group uses a different seed",
           "Confirm the inference engine is not caching and replaying identical outputs"]);
      }
    }
  ];

  // -------------------------------------------------------------- doctor
  function Doctor(opts) {
    opts = opts || {};
    this.detectors = opts.detectors || DETECTORS;
    this.minSeverity = opts.minSeverity || "info";
    this.history = new RunHistory(opts.window || 250);
    this.diagnoses = [];
    this.errors = {};
    this._lastFired = {};
    this._n = 0;
  }
  Doctor.prototype.observe = function (step) {
    this._n++;
    this.history.observe(step);
    // Cooldowns are measured in training steps, not in observations, so that a
    // caller who skips steps gets the same behaviour as the Python package.
    var stepNo = (step && step.step !== undefined && step.step !== null) ? step.step : this._n;
    var fired = [];
    for (var i = 0; i < this.detectors.length; i++) {
      var det = this.detectors[i];
      if (this.history.length() < det.minSteps) continue;
      var last = this._lastFired[det.code];
      if (last !== undefined && stepNo - last < det.cooldown) continue;
      var ok = true;
      for (var r = 0; r < det.requires.length; r++) {
        if (!this.history.has(det.requires[r])) { ok = false; break; }
      }
      if (!ok) continue;
      var d = null;
      try {
        // A broken detector must never kill a training run. Same contract as Python.
        d = det.check(this.history);
      } catch (err) {
        this.errors[det.code] = String(err && err.message ? err.message : err);
        continue;
      }
      if (!d) continue;
      if (SEVERITY[d.severity] < SEVERITY[this.minSeverity]) continue;
      this._lastFired[det.code] = stepNo;
      this.diagnoses.push(d);
      fired.push(d);
    }
    return fired;
  };
  Doctor.prototype.counts = function () {
    var c = { info: 0, warn: 0, critical: 0 };
    for (var i = 0; i < this.diagnoses.length; i++) c[this.diagnoses[i].severity]++;
    return c;
  };

  // ------------------------------------------------------------ simulator
  var FILLER = [
    "let me work through this step by step",
    "first we identify what the question is asking",
    "the key observation here is symmetry",
    "combining both halves of the expression",
    "we can factor the remaining terms",
    "substituting the value back in",
    "checking the boundary conditions carefully",
    "this simplifies to a single fraction"
  ];
  var EXPLOIT = "therefore the answer is obviously";
  var N_LP = 32;

  var SCENARIOS = {
    healthy: { label: "Healthy run", code: null, blurb: "Reward improves, entropy decays gently, nothing fires. This is the control - a diagnostic tool that cannot stay quiet on a good run is useless." },
    logprob_divergence: { label: "Trainer/generator divergence", code: "RLD-014", blurb: "Inference and training disagree about the logprobs of the very same tokens. Every dashboard looks fine." },
    template_mismatch: { label: "Chat template mismatch", code: "RLD-092", blurb: "Divergence concentrated in the first 8 tokens: a prompt-boundary bug, not a kernel difference." },
    reward_hacking: { label: "Reward hacking", code: "RLD-031", blurb: "Reward climbs beautifully. Accuracy does not move. A phrase is paying the rent." },
    length_exploit: { label: "Length exploit", code: "RLD-033", blurb: "The policy discovered that longer answers score higher, regardless of correctness." },
    entropy_collapse: { label: "Entropy collapse", code: "RLD-007", blurb: "Fast early progress, then exploration dies and the run plateaus forever." },
    advantage_collapse: { label: "Advantage collapse", code: "RLD-001", blurb: "Every rollout in a group scores the same, so every advantage is zero and nothing is learned." },
    truncation_bias: { label: "Truncation bias", code: "RLD-042", blurb: "Reasoning traces grow into the token limit and get scored as failures." },
    kl_blowup: { label: "KL blowup", code: "RLD-021", blurb: "The policy escapes the trust region; quality collapses after the reward peak." },
    stale_offpolicy: { label: "Stale off-policy data", code: "RLD-055", blurb: "Clip fraction saturates: you generate rollouts and then throw the gradient away." },
    dead_reward: { label: "Dead reward signal", code: "RLD-060", blurb: "The verifier silently returns a constant. Everything looks smooth because nothing is happening." },
    mode_collapse: { label: "Mode collapse", code: "RLD-018", blurb: "Rollouts within a group are byte-identical - usually a sampling config left on greedy." },
    gradient_spike: { label: "Gradient spike", code: "RLD-071", blurb: "A single outlier batch produces a gradient spike that undoes hours of training." },
    value_divergence: { label: "Value divergence", code: "RLD-084", blurb: "The critic diverges while the policy keeps training on its increasingly noisy advantages." }
  };

  function completion(rng, tokens, opts) {
    opts = opts || {};
    // Greedy decoding: every sample in the group is the same string. The length
    // must not vary either, or the completions stop being byte-identical and
    // RLD-018 (which compares whole completions) never sees a duplicate.
    if (opts.degenerate) return "the answer is 42, " + FILLER[0];
    var parts = [];
    if (opts.exploit) parts.push(EXPLOIT);
    var want = Math.max(1, Math.round(tokens / 6));
    for (var i = 0; i < want; i++) parts.push(rng.choice(FILLER));
    return parts.join(", ");
  }

  function logprobPair(rng, gap, headGap) {
    var a = [], b = [];
    for (var i = 0; i < N_LP; i++) {
      var base = -rng.next() * 2.2 - 0.05;
      var g = (i < 8 && headGap !== null && headGap !== undefined) ? headGap : gap;
      a.push(base);
      b.push(base + (rng.next() < 0.5 ? -1 : 1) * g * (0.6 + 0.8 * rng.next()));
    }
    return [a, b];
  }

  /* Generate one run. Returns { steps: [Step], scenario, opts }.
     `overrides` lets the playground apply the mitigations from the sidebar
     (entropy bonus, length penalty, KL penalty, bigger groups...) so people can
     watch a diagnosis disappear when they fix the cause. */
  function simulateRun(scenario, steps, seed, overrides) {
    scenario = scenario in SCENARIOS ? scenario : "healthy";
    steps = steps || 300;
    var o = overrides || {};
    var rng = new Rng(seed || 7);
    var groups = o.groups || 4;
    var groupSize = o.groupSize || 6;
    var onset = 0.3;
    var out = [];

    var entropyBonus = !!o.entropyBonus;
    var lengthPenalty = !!o.lengthPenalty;
    var klPenalty = !!o.klPenalty;
    var holdout = !!o.holdoutVerifier;
    var raiseLimit = !!o.raiseTokenLimit;
    var lowerLr = !!o.lowerLr;

    var entBase = 1.55, klRun = 0.02, valueLoss = 0.42;

    for (var s = 0; s < steps; s++) {
      var frac = steps > 1 ? s / (steps - 1) : 0;
      var ramp = frac <= onset ? 0 : (frac - onset) / (1 - onset);

      // ---- baseline healthy trajectory
      var skill = 0.28 + 0.34 * (1 - Math.exp(-3.1 * frac));
      var entropy = entBase * Math.exp(-1.05 * frac) + 0.42 + rng.gauss(0, 0.02);
      if (entropyBonus) entropy += 0.34 * frac;
      var lenSigma = 70.0;
      var lenMu = 210 + 60 * frac;
      var clipFrac = 0.05 + 0.03 * rng.next();
      var kl = 0.03 + 0.05 * frac + rng.gauss(0, 0.004);
      var gradNorm = 0.55 + rng.gauss(0, 0.09);
      var gap = 4e-5 + 3e-5 * rng.next();
      var headGap = null;
      var tokenLimit = raiseLimit ? 1400 : 900;
      var forceDegenerate = false, exploitFrac = 0, flatTruth = false;
      var deadReward = false, dupTarget = 0;

      switch (scenario) {
        case "logprob_divergence":
          gap = 4e-5 + ramp * 9e-3;
          skill = 0.28 + 0.1 * (1 - Math.exp(-3.1 * frac)) - 0.05 * ramp;
          break;
        case "template_mismatch":
          gap = 4e-5 + ramp * 6e-4;
          headGap = 4e-5 + ramp * 4.2e-2;
          break;
        case "reward_hacking":
          // Ground truth is FLAT for the whole run: the policy is not learning
          // the task, it is learning the grader.
          skill = 0.29 + 0.01 * frac;
          exploitFrac = Math.min(0.92, ramp * 1.25);
          flatTruth = true;
          break;
        case "length_exploit":
          lenSigma = 70.0 + 240.0 * ramp;
          lenMu = 210 + 60 * frac + 520 * ramp;
          break;
        case "entropy_collapse":
          entropy = entBase * Math.exp(-1.05 * frac) + 0.42 - 1.15 * ramp + rng.gauss(0, 0.015);
          if (entropyBonus) entropy += 0.62 * ramp;
          if (ramp > 0.55 && !entropyBonus) forceDegenerate = true;
          break;
        case "advantage_collapse":
          if (ramp > 0.15) forceDegenerate = true;
          break;
        case "truncation_bias":
          lenMu = 210 + 60 * frac + 700 * ramp;
          lenSigma = 70.0 + 150.0 * ramp;
          break;
        case "kl_blowup":
          klRun += ramp * ramp * 0.12;
          kl = 0.03 + klRun * (klPenalty ? 0.18 : 1) * 22 + rng.gauss(0, 0.01);
          if (!klPenalty && ramp > 0.6) skill = Math.max(0.05, skill - 0.5 * (ramp - 0.6));
          break;
        case "stale_offpolicy":
          clipFrac = 0.05 + ramp * (lowerLr ? 0.14 : 0.48) + rng.gauss(0, 0.01);
          break;
        case "dead_reward":
          deadReward = ramp > 0.05;
          break;
        case "mode_collapse":
          dupTarget = Math.min(0.85, ramp * 1.3);
          entropy = entBase * Math.exp(-1.05 * frac) + 0.42 - 0.5 * ramp;
          break;
        case "gradient_spike":
          if (frac > 0.62 && frac < 0.645) gradNorm = 34.0 + 8 * rng.next();
          break;
        case "value_divergence":
          valueLoss = 0.42 + Math.pow(ramp, 1.6) * 3.4;
          break;
      }
      if (lengthPenalty) { lenMu = Math.min(lenMu, 340); lenSigma = Math.min(lenSigma, 90); }
      if (lowerLr) { entropy += 0.12 * frac; gradNorm *= 0.6; }

      // ---- rollouts
      var rollouts = [];
      for (var g = 0; g < groups; g++) {
        // Two distinct failures that are easy to conflate: identical *rewards*
        // (advantage collapse) and identical *text* (mode collapse). Only the
        // second one makes rollouts byte-identical.
        var groupDuplicate = dupTarget > 0 && rng.next() < dupTarget;
        var groupDegenerate = forceDegenerate || groupDuplicate;
        for (var k = 0; k < groupSize; k++) {
          var nTokens = Math.max(24, Math.round(rng.gauss(lenMu, lenSigma)));
          var truncated = nTokens >= tokenLimit;
          if (truncated) nTokens = tokenLimit;
          var isExploit = exploitFrac > 0 && rng.next() < exploitFrac;
          var correct = rng.next() < skill;

          var reward;
          if (deadReward) {
            reward = 0.5;
            correct = null;
          } else if (scenario === "length_exploit") {
            reward = 0.4 * (correct ? 1 : 0) + 0.6 * Math.min(1, nTokens / 900);
          } else if (isExploit) {
            // The exploit reliably tops the batch: that is what makes it an exploit.
            reward = 1.0 + 0.02 * rng.next();
          } else {
            // Match the package exactly: binary correctness plus a tiny format
            // credit, and no length term. A length-proportional bonus here is a
            // real length/reward coupling, and because longer completions carry
            // more filler n-grams, RLD-031 was right to flag it on healthy runs.
            reward = (correct ? 1 : 0) + 0.02 * rng.next();
          }
          if (truncated && scenario === "truncation_bias") reward *= 0.25;
          if (lengthPenalty && nTokens > 340) reward -= 0.12;
          if (groupDegenerate) reward = 0.5;
          if (holdout && isExploit) reward = 0.31 + 0.05 * rng.next();

          var text = completion(rng, nTokens, {
            exploit: isExploit,
            degenerate: groupDuplicate
          });
          var lp = logprobPair(rng, gap, headGap);
          rollouts.push(new Rollout({
            reward: reward, completion: text, nTokens: nTokens, truncated: truncated,
            correct: correct, groupId: "g" + g,
            rolloutLogprobs: lp[0], trainerLogprobs: lp[1]
          }));
        }
      }

      var metrics = {
        entropy: Math.max(0.02, entropy),
        clip_frac: Math.max(0, Math.min(1, clipFrac)),
        kl: Math.max(0, kl),
        grad_norm: Math.max(0.01, gradNorm),
        value_loss: Math.max(0.01, valueLoss + rng.gauss(0, 0.03)),
        learning_rate: lowerLr ? 3e-7 : 1e-6
      };
      if (flatTruth || scenario === "reward_hacking") {
        metrics.accuracy = Math.max(0, Math.min(1, skill + rng.gauss(0, 0.012)));
      }
      out.push(new Step(s, rollouts, metrics));
    }
    return { steps: out, scenario: scenario };
  }

  /* Run the doctor over a simulated run and return everything the UI needs. */
  function diagnoseRun(scenario, steps, seed, overrides) {
    var run = simulateRun(scenario, steps, seed, overrides);
    var doctor = new Doctor({ window: 250 });
    var series = {
      reward_mean: [], accuracy: [], entropy: [], logprob_gap: [],
      clip_frac: [], kl: [], truncated_frac: [], seq_len_mean: [],
      advantage_std: [], degenerate_group_frac: [], duplicate_frac: [],
      grad_norm: [], value_loss: [], reward_std: []
    };
    var events = [];
    for (var i = 0; i < run.steps.length; i++) {
      var st = run.steps[i];
      var fired = doctor.observe(st);
      var m = st.merged();
      for (var key in series) {
        if (series.hasOwnProperty(key)) {
          series[key].push(m[key] === undefined || m[key] === null ? null : m[key]);
        }
      }
      for (var f = 0; f < fired.length; f++) {
        fired[f].step = st.step;
        events.push(fired[f]);
      }
    }
    return {
      scenario: scenario,
      steps: run.steps.length,
      series: series,
      events: events,
      counts: doctor.counts(),
      errors: doctor.errors,
      lastStep: run.steps[run.steps.length - 1],
      expected: SCENARIOS[scenario] ? SCENARIOS[scenario].code : null
    };
  }

  global.SyntropyRL = {
    stats: S,
    Rng: Rng,
    Rollout: Rollout,
    Step: Step,
    RunHistory: RunHistory,
    Doctor: Doctor,
    DETECTORS: DETECTORS,
    SCENARIOS: SCENARIOS,
    EXPLOIT: EXPLOIT,
    simulateRun: simulateRun,
    diagnoseRun: diagnoseRun,
    HEALTHY_GAP: HEALTHY_GAP
  };
})(typeof window !== "undefined" ? window : this);
