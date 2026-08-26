#!/usr/bin/env node
/* Parity check: docs/assets/engine.js must agree with the Python package.
 *
 * The playground is a port, and a port drifts. This runs the browser engine
 * headlessly over every scenario and asserts the same thing the Python
 * `syntropyrl selftest` asserts:
 *
 *   - each broken scenario trips its own detector
 *   - the healthy scenario trips nothing
 *
 * Run it with: node scripts/selftest_js.js
 */
"use strict";

var path = require("path");
var loaded = require(path.join(__dirname, "..", "docs", "assets", "engine.js"));
var E = loaded.SyntropyRL || global.SyntropyRL;

if (!E) {
  console.error("could not load SyntropyRL from docs/assets/engine.js");
  process.exit(2);
}

// Must stay identical to EXPECTED in src/syntropyrl/cli.py
var EXPECTED = {
  logprob_divergence: "RLD-014",
  template_mismatch: "RLD-092",
  reward_hacking: "RLD-031",
  length_exploit: "RLD-033",
  entropy_collapse: "RLD-007",
  advantage_collapse: "RLD-001",
  truncation_bias: "RLD-042",
  kl_blowup: "RLD-021",
  stale_offpolicy: "RLD-055",
  dead_reward: "RLD-060",
  mode_collapse: "RLD-018",
  gradient_spike: "RLD-071",
  value_divergence: "RLD-084"
};

var SEEDS = [7, 21, 99];
var STEPS = 300;
var failures = [];
var pad = function (s, n) {
  s = String(s);
  while (s.length < n) s += " ";
  return s;
};

function uniqueCodes(res) {
  var seen = {}, out = [];
  for (var i = 0; i < res.events.length; i++) {
    if (!seen[res.events[i].code]) { seen[res.events[i].code] = 1; out.push(res.events[i].code); }
  }
  return out;
}

SEEDS.forEach(function (seed) {
  console.log("\nseed " + seed);
  var t0 = Date.now();

  var healthy = E.diagnoseRun("healthy", STEPS, seed, {});
  var healthyCodes = uniqueCodes(healthy);
  console.log("  " + pad("healthy", 22) +
    (healthyCodes.length ? "FALSE POSITIVES: " + healthyCodes.join(", ") : "ok"));
  if (healthyCodes.length) {
    failures.push("seed " + seed + ": healthy produced " + healthyCodes.join(", "));
  }

  Object.keys(EXPECTED).forEach(function (scenario) {
    var res = E.diagnoseRun(scenario, STEPS, seed, {});
    var codes = uniqueCodes(res);
    var hit = codes.indexOf(EXPECTED[scenario]) !== -1;
    console.log("  " + pad(scenario, 22) + (hit ? "ok   " : "MISS ") +
      "expected " + EXPECTED[scenario] + " | fired " + (codes.join(", ") || "-"));
    if (!hit) {
      failures.push("seed " + seed + ": " + scenario + " did not fire " + EXPECTED[scenario]);
    }
    var errs = Object.keys(res.errors || {});
    if (errs.length) {
      failures.push("seed " + seed + ": " + scenario + " detector errors " + JSON.stringify(res.errors));
    }
  });

  console.log("  (" + (Date.now() - t0) + " ms for " + (Object.keys(EXPECTED).length + 1) +
    " runs of " + STEPS + " steps)");
});

console.log("");
if (failures.length) {
  failures.forEach(function (f) { console.log("  FAIL: " + f); });
  console.log("\n  The playground has drifted from the package. Fix engine.js.");
  process.exit(1);
}
console.log("  browser engine matches the package on " + SEEDS.length + " seeds");
