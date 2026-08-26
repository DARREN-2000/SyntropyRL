/* syntropyrl playground UI
 * Vanilla JS, no libraries, canvas charts drawn by hand.
 * All diagnosis work happens in engine.js, which is a port of the Python package.
 */
(function () {
  "use strict";
  var E = window.SyntropyRL;
  var S = E.stats;

  var state = {
    scenario: "reward_hacking",
    steps: 300,
    seed: 7,
    result: null
  };

  var el = {
    list: document.getElementById("scenario-list"),
    blurb: document.getElementById("scenario-blurb"),
    steps: document.getElementById("steps"),
    stepsOut: document.getElementById("steps-out"),
    seed: document.getElementById("seed"),
    seedOut: document.getElementById("seed-out"),
    run: document.getElementById("run-btn"),
    reset: document.getElementById("reset-btn"),
    verdict: document.getElementById("verdict"),
    charts: document.getElementById("charts"),
    timeline: document.getElementById("timeline"),
    report: document.getElementById("report"),
    reportTitle: document.getElementById("report-title"),
    rollouts: document.getElementById("rollouts"),
    fidelity: document.getElementById("fidelity")
  };

  var TOGGLES = [
    ["opt-entropy", "entropyBonus"],
    ["opt-length", "lengthPenalty"],
    ["opt-kl", "klPenalty"],
    ["opt-holdout", "holdoutVerifier"],
    ["opt-limit", "raiseTokenLimit"],
    ["opt-lr", "lowerLr"],
    ["opt-groups", "bigGroups"]
  ];

  function options() {
    var o = {};
    for (var i = 0; i < TOGGLES.length; i++) {
      var node = document.getElementById(TOGGLES[i][0]);
      o[TOGGLES[i][1]] = !!(node && node.checked);
    }
    if (o.bigGroups) { o.groupSize = 12; }
    return o;
  }

  // ------------------------------------------------------------------ charts
  var CHARTS = [
    {
      key: "reward_mean", name: "Reward", color: "#5E9FE8", digits: 3,
      overlay: "accuracy", overlayColor: "#72BC8F",
      foot: "Blue: proxy reward. Green: ground-truth accuracy when logged."
    },
    {
      key: "entropy", name: "Policy entropy", color: "#BF8EDA", digits: 3,
      threshold: 0.4, thresholdLabel: "RLD-007 floor",
      foot: "Below 0.4 with a 0.55 nat drop fires RLD-007."
    },
    {
      key: "logprob_gap", name: "Logprob gap (trainer vs generator)", color: "#E97366", digits: 5,
      threshold: 0.003, thresholdLabel: "RLD-014 gate",
      foot: "Healthy is below 1e-3. Three times that fires RLD-014."
    },
    {
      key: "degenerate_group_frac", name: "Degenerate GRPO groups", color: "#DE9255", digits: 3,
      threshold: 0.55, thresholdLabel: "RLD-001 gate",
      foot: "Fraction of groups where every rollout scored identically."
    },
    {
      key: "clip_frac", name: "Clip fraction", color: "#EAC26B", digits: 3,
      threshold: 0.3, thresholdLabel: "RLD-055 gate",
      foot: "Healthy 0.02-0.20. Above 0.30 you are discarding the batch."
    },
    {
      key: "seq_len_mean", name: "Completion length", color: "#4FB9C9", digits: 0,
      overlay: "truncated_frac", overlayColor: "#E97366", overlayScale: true,
      foot: "Teal: mean tokens. Red: truncated fraction, rescaled."
    }
  ];

  function buildCharts() {
    el.charts.innerHTML = "";
    for (var i = 0; i < CHARTS.length; i++) {
      var c = CHARTS[i];
      var card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML =
        '<div class="chart-head"><span class="chart-name">' + c.name + "</span>" +
        '<span class="chart-value" id="cv-' + c.key + '">-</span></div>' +
        '<canvas id="cc-' + c.key + '" height="96"></canvas>' +
        '<p class="chart-foot">' + c.foot + "</p>";
      el.charts.appendChild(card);
    }
  }

  function drawChart(chart, series, events) {
    var canvas = document.getElementById("cc-" + chart.key);
    if (!canvas) return;
    var data = (series[chart.key] || []).slice();
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || 320;
    var h = 96;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var clean = S.finite(data);
    var valueOut = document.getElementById("cv-" + chart.key);
    if (!clean.length) {
      ctx.fillStyle = "#9B9892";
      ctx.font = "12px -apple-system, system-ui, sans-serif";
      ctx.fillText("not logged in this scenario", 8, h / 2);
      if (valueOut) valueOut.textContent = "n/a";
      return;
    }

    var pad = 6;
    var lo = Math.min.apply(null, clean);
    var hi = Math.max.apply(null, clean);
    if (chart.threshold !== undefined) {
      lo = Math.min(lo, chart.threshold);
      hi = Math.max(hi, chart.threshold * 1.08);
    }
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var span = hi - lo;
    lo -= span * 0.08;
    hi += span * 0.08;

    var xAt = function (i) { return pad + (i / Math.max(1, data.length - 1)) * (w - 2 * pad); };
    var yAt = function (v) { return h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad); };

    // baseline grid
    ctx.strokeStyle = "#F0EFED";
    ctx.lineWidth = 1;
    for (var g = 0; g <= 2; g++) {
      var gy = pad + (g / 2) * (h - 2 * pad);
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w - pad, gy); ctx.stroke();
    }

    // threshold
    if (chart.threshold !== undefined) {
      ctx.save();
      ctx.strokeStyle = "#D6D4D1";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad, yAt(chart.threshold));
      ctx.lineTo(w - pad, yAt(chart.threshold));
      ctx.stroke();
      ctx.restore();
    }

    // event ticks
    if (events && events.length) {
      ctx.save();
      for (var e = 0; e < events.length; e++) {
        var idx = events[e].step;
        if (idx == null) continue;
        ctx.strokeStyle = events[e].severity === "critical" ? "rgba(229,100,88,0.55)" : "rgba(213,128,59,0.45)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(xAt(idx), pad);
        ctx.lineTo(xAt(idx), h - pad);
        ctx.stroke();
      }
      ctx.restore();
    }

    // overlay series
    if (chart.overlay && series[chart.overlay]) {
      var ov = series[chart.overlay];
      var ovClean = S.finite(ov);
      if (ovClean.length) {
        var oLo = Math.min.apply(null, ovClean), oHi = Math.max.apply(null, ovClean);
        if (oHi - oLo < 1e-9) oHi = oLo + 1;
        ctx.strokeStyle = chart.overlayColor;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        var started = false;
        for (var oi = 0; oi < ov.length; oi++) {
          var ovv = ov[oi];
          if (ovv === null || !isFinite(ovv)) continue;
          var oy = chart.overlayScale
            ? h - pad - ((ovv - oLo) / (oHi - oLo)) * (h - 2 * pad)
            : yAt(ovv);
          if (!started) { ctx.moveTo(xAt(oi), oy); started = true; }
          else ctx.lineTo(xAt(oi), oy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // main series
    ctx.strokeStyle = chart.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    var begun = false;
    for (var i = 0; i < data.length; i++) {
      var v = data[i];
      if (v === null || !isFinite(v)) continue;
      if (!begun) { ctx.moveTo(xAt(i), yAt(v)); begun = true; }
      else ctx.lineTo(xAt(i), yAt(v));
    }
    ctx.stroke();

    if (valueOut) {
      var last = clean[clean.length - 1];
      valueOut.textContent = chart.digits === 0 ? Math.round(last) + " tok" : S.fmt(last, chart.digits);
    }
  }

  // ---------------------------------------------------------------- verdict
  function renderVerdict(res) {
    var counts = res.counts;
    var cls = "verdict", icon = "&#10003;", title, sub;
    var unique = {};
    for (var i = 0; i < res.events.length; i++) unique[res.events[i].code] = 1;
    var codes = Object.keys(unique);

    if (counts.critical > 0) {
      cls += " is-crit"; icon = "&#9888;";
      title = counts.critical + " critical" + (counts.warn ? ", " + counts.warn + " warning" : "");
      sub = codes.join(", ") + " fired across " + res.steps + " steps.";
    } else if (counts.warn > 0) {
      cls += " is-warn"; icon = "&#9888;";
      title = counts.warn + " warning" + (counts.warn > 1 ? "s" : "");
      sub = codes.join(", ") + " fired across " + res.steps + " steps.";
    } else {
      cls += " is-clean";
      title = "No failures detected";
      sub = "13 detectors ran over " + res.steps + " steps and stayed silent. That means no known " +
        "failure mode fired, not that the run is good.";
    }
    el.verdict.className = cls;
    el.verdict.innerHTML =
      '<div class="verdict-icon" aria-hidden="true">' + icon + "</div>" +
      '<div class="verdict-text"><strong>' + title + "</strong><span>" + sub + "</span></div>";
  }

  // --------------------------------------------------------------- timeline
  function renderTimeline(res) {
    var firstByCode = {};
    var order = [];
    for (var i = 0; i < res.events.length; i++) {
      var ev = res.events[i];
      if (!firstByCode[ev.code]) { firstByCode[ev.code] = ev; order.push(ev.code); }
    }
    if (!order.length) {
      el.timeline.innerHTML = '<p class="tl-empty">Nothing fired. For a healthy run that is the ' +
        "correct answer - a detector that cannot stay quiet is useless.</p>";
      return;
    }
    var html = "";
    for (var k = 0; k < order.length; k++) {
      var d = firstByCode[order[k]];
      html +=
        '<div class="tl-row sev-' + d.severity + '">' +
        '<span class="tl-step">step ' + d.step + "</span>" +
        '<span><span class="code-chip">' + d.code + "</span></span>" +
        '<div class="tl-body"><strong>' + escapeHtml(d.title) + "</strong>" +
        "<p>" + escapeHtml(d.summary) + "</p>" +
        (d.fixes && d.fixes.length
          ? '<p class="tl-fix"><b>Try first:</b> ' + escapeHtml(d.fixes[0]) + "</p>"
          : "") +
        '<p class="tl-fix"><a href="' + d.url + '">Atlas entry for ' + d.code + " &rarr;</a></p>" +
        "</div></div>";
    }
    el.timeline.innerHTML = html;
  }

  // ----------------------------------------------------------------- report
  function renderReport(res) {
    var lines = [];
    var counts = res.counts;
    var head = [];
    if (counts.critical) head.push(counts.critical + " critical");
    if (counts.warn) head.push(counts.warn + " warning" + (counts.warn > 1 ? "s" : ""));
    if (counts.info) head.push(counts.info + " info");

    lines.push(["$ syntropyrl demo " + res.scenario + " --steps " + res.steps + " --seed " + state.seed, "t-dim"]);
    lines.push(["", ""]);
    lines.push(["  \u{1FA7A} syntropyrl  \u2014  step " + res.steps + "  \u2014  " +
      (head.length ? head.join(", ") : "clean"), "t-bold"]);
    lines.push(["  " + repeat("\u2500", 58), "t-dim"]);
    lines.push(["", ""]);

    var seen = {};
    var any = false;
    for (var i = 0; i < res.events.length; i++) {
      var d = res.events[i];
      if (seen[d.code]) continue;
      seen[d.code] = 1;
      any = true;
      var label = d.severity === "critical" ? "[CRITICAL]" : d.severity === "warn" ? "[WARN]    " : "[INFO]    ";
      var cls = d.severity === "critical" ? "t-crit" : d.severity === "warn" ? "t-orange" : "t-blue";
      lines.push(["  " + label + " " + d.code + "  " + d.title, cls]);
      var wrapped = wrap(d.summary, 66);
      for (var w = 0; w < wrapped.length; w++) lines.push(["    " + wrapped[w], ""]);
      for (var f = 0; f < Math.min(2, d.fixes.length); f++) {
        var fw = wrap(d.fixes[f], 62);
        lines.push(["      fix  " + fw[0], "t-green"]);
        for (var x = 1; x < fw.length; x++) lines.push(["           " + fw[x], "t-green"]);
      }
      lines.push(["      see  syntropyrl.dev/atlas.html#" + d.code, "t-blue"]);
      lines.push(["", ""]);
    }
    if (!any) {
      lines.push(["  No failures detected across " + res.steps + " steps.", "t-green"]);
      lines.push(["", ""]);
      var caveat = wrap("A clean report means no known failure mode fired, not that the run is " +
        "good. 13 detectors ran; the ones needing metrics you did not log were skipped.", 66);
      for (var c = 0; c < caveat.length; c++) lines.push(["  " + caveat[c], "t-dim"]);
      lines.push(["", ""]);
    }
    var errCount = Object.keys(res.errors || {}).length;
    lines.push(["  " + Object.keys(seen).length + " unique diagnoses across " + res.steps +
      " steps. 13 detectors ran, " + errCount + " errored.", "t-dim"]);

    el.report.textContent = "";
    for (var L = 0; L < lines.length; L++) {
      var span = document.createElement("span");
      if (lines[L][1]) span.className = lines[L][1];
      span.textContent = lines[L][0] + "\n";
      el.report.appendChild(span);
    }
    el.reportTitle.textContent = "syntropyrl demo " + res.scenario;
  }

  // --------------------------------------------------------------- rollouts
  function renderRollouts(res) {
    var step = res.lastStep;
    if (!step) { el.rollouts.innerHTML = ""; return; }
    var top = step.topRollouts(6);

    // Which phrase is paying? Ask the same helper the detector uses.
    var all = step.rollouts;
    var best = S.rewardCorrelatedNgram(
      all.map(function (r) { return r.completion; }),
      all.map(function (r) { return r.reward; }), 4, 40, 3
    );
    var flagged = best && best.t >= 6.0 ? best.gram : null;

    var html = "";
    for (var i = 0; i < top.length; i++) {
      var r = top[i];
      var text = r.completion.length > 190 ? r.completion.slice(0, 190) + "..." : r.completion;
      var isFlagged = flagged && text.indexOf(flagged) !== -1;
      var safe = escapeHtml(text);
      if (flagged) {
        safe = safe.split(escapeHtml(flagged)).join("<mark>" + escapeHtml(flagged) + "</mark>");
      }
      html += '<div class="rollout' + (isFlagged ? " is-flagged" : "") + '">' +
        '<span class="r-reward">' + S.fmt(r.reward, 3) + "</span>" +
        '<span class="r-text">' + safe +
        '<span style="display:block;color:var(--text-3);margin-top:4px">' +
        r.nTokens + " tokens" + (r.truncated ? " &middot; truncated" : "") +
        (r.correct === null ? "" : r.correct ? " &middot; correct" : " &middot; wrong") +
        "</span></span></div>";
    }
    el.rollouts.innerHTML = html;
  }

  function renderFidelity(res) {
    var expected = res.expected;
    var fired = {};
    for (var i = 0; i < res.events.length; i++) fired[res.events[i].code] = 1;
    var codes = Object.keys(fired);
    var msg;
    if (!expected) {
      msg = codes.length
        ? "This is the healthy control and " + codes.join(", ") + " fired - on the real package " +
          "that is a bug, and the test suite would fail."
        : "Healthy control: nothing fired, which is the assertion the Python test suite makes on " +
          "every seed.";
    } else if (fired[expected]) {
      msg = "Expected " + expected + ", fired " + codes.join(", ") + ". Cascades are normal and " +
        "intentional: one broken thing breaks others, so the tests assert the expected code is " +
        "present rather than alone.";
    } else {
      msg = "Expected " + expected + " but it did not fire" +
        (codes.length ? " (got " + codes.join(", ") + ")" : "") +
        ". Try more steps, or turn the fixes back off - some mitigations legitimately silence it.";
    }
    el.fidelity.textContent = msg + " Same thresholds as the Python package; run " +
      "`syntropyrl selftest` locally to check all 13 at once.";
  }

  // ------------------------------------------------------------------- util
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function repeat(ch, n) {
    var out = "";
    for (var i = 0; i < n; i++) out += ch;
    return out;
  }
  function wrap(text, width) {
    var words = String(text).split(/\s+/), lines = [], cur = "";
    for (var i = 0; i < words.length; i++) {
      if (!cur.length) { cur = words[i]; continue; }
      if ((cur + " " + words[i]).length <= width) cur += " " + words[i];
      else { lines.push(cur); cur = words[i]; }
    }
    if (cur.length) lines.push(cur);
    return lines.length ? lines : [""];
  }

  // ------------------------------------------------------------------- run
  function run() {
    el.run.disabled = true;
    el.run.textContent = "Running...";
    // Yield once so the button state paints before we block the thread.
    setTimeout(function () {
      var res;
      try {
        res = E.diagnoseRun(state.scenario, state.steps, state.seed, options());
      } catch (err) {
        el.report.textContent = "engine error: " + (err && err.message ? err.message : err);
        el.run.disabled = false;
        el.run.textContent = "Run " + state.steps + " steps";
        return;
      }
      state.result = res;
      renderVerdict(res);
      for (var i = 0; i < CHARTS.length; i++) drawChart(CHARTS[i], res.series, res.events);
      renderTimeline(res);
      renderReport(res);
      renderRollouts(res);
      renderFidelity(res);
      el.run.disabled = false;
      el.run.textContent = "Run " + state.steps + " steps";
    }, 16);
  }

  function buildScenarioList() {
    var keys = Object.keys(E.SCENARIOS);
    el.list.innerHTML = "";
    for (var i = 0; i < keys.length; i++) {
      var key = keys[i];
      var sc = E.SCENARIOS[key];
      var b = document.createElement("button");
      b.type = "button";
      b.className = "scenario-btn" + (key === state.scenario ? " is-active" : "");
      b.setAttribute("data-scenario", key);
      b.setAttribute("aria-pressed", key === state.scenario ? "true" : "false");
      b.innerHTML = escapeHtml(sc.label) +
        '<span class="sc-code">' + (sc.code ? sc.code : "control - nothing should fire") + "</span>";
      el.list.appendChild(b);
    }
    el.blurb.textContent = E.SCENARIOS[state.scenario].blurb;
  }

  el.list.addEventListener("click", function (ev) {
    var btn = ev.target.closest ? ev.target.closest(".scenario-btn") : null;
    if (!btn) return;
    state.scenario = btn.getAttribute("data-scenario");
    var all = el.list.querySelectorAll(".scenario-btn");
    for (var i = 0; i < all.length; i++) {
      var on = all[i] === btn;
      all[i].classList.toggle("is-active", on);
      all[i].setAttribute("aria-pressed", on ? "true" : "false");
    }
    el.blurb.textContent = E.SCENARIOS[state.scenario].blurb;
    run();
  });

  el.steps.addEventListener("input", function () {
    state.steps = parseInt(el.steps.value, 10);
    el.stepsOut.textContent = state.steps;
    el.run.textContent = "Run " + state.steps + " steps";
  });
  el.steps.addEventListener("change", run);
  el.seed.addEventListener("input", function () {
    state.seed = parseInt(el.seed.value, 10);
    el.seedOut.textContent = state.seed;
  });
  el.seed.addEventListener("change", run);
  el.run.addEventListener("click", run);
  el.reset.addEventListener("click", function () {
    for (var i = 0; i < TOGGLES.length; i++) {
      var node = document.getElementById(TOGGLES[i][0]);
      if (node) node.checked = false;
    }
    run();
  });
  for (var t = 0; t < TOGGLES.length; t++) {
    var node = document.getElementById(TOGGLES[t][0]);
    if (node) node.addEventListener("change", run);
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    if (!state.result) return;
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      for (var i = 0; i < CHARTS.length; i++) drawChart(CHARTS[i], state.result.series, state.result.events);
    }, 120);
  });

  buildScenarioList();
  buildCharts();
  run();
})();
