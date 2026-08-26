"""The RL Failure Atlas: the knowledge, as data.

Detectors answer "is this happening?". This module answers "what is it, why does
it happen, and what do I do?". Keeping it as structured data means one source of
feeds the CLI, ATLAS.md, and the website, and it means a contributor can add an
entry without touching detector code.

Every detector code must have an entry here. There is a test that enforces it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

ATLAS_URL = "https://syntropyrl.dev/atlas.html"

# Each entry:
#   symptom  - what you see in your dashboard, in the user's words
#   measure  - what syntropyrl actually computes, so the claim is falsifiable
#   causes   - ranked most-likely-first, from real reports
#   fixes    - concrete, in order of "try this first"
#   cost     - why you care; the reason this entry exists
#   repro    - a command that reproduces the failure locally
ENTRIES: Dict[str, dict] = {
    "RLD-014": {
        "title": "Trainer/generator logprob divergence",
        "family": "numerics",
        "aka": ["training-inference mismatch", "vLLM logprob mismatch"],
        "symptom": (
            "Training is stable, losses look fine, reward improves slowly or not at all. "
            "Sometimes it works at small scale and fails when you turn on tensor parallelism "
            "or a longer context."
        ),
        "measure": (
            "Mean absolute per-token difference between the logprobs your inference engine "
            "returned and the logprobs your trainer recomputes for the same tokens. Healthy "
            "is below 1e-3. syntropyrl also correlates the gap with sequence length, which "
            "separates kernel differences from boundary bugs."
        ),
        "causes": [
            "Chunked prefill or paged attention producing different numerics than the training forward pass",
            "bf16/fp16 generation compared against fp32 recomputation",
            "Sampling parameters (temperature, top_p, repetition penalty) applied in one path but not the other",
            "Tensor/sequence parallel reductions changing summation order",
            "The trainer scoring a different token sequence than the one that was generated (off-by-one on the prompt boundary)",
        ],
        "fixes": [
            "Run syntropyrl.check_logprob_parity() before training and assert verdict == 'pass'",
            "Disable chunked prefill in the rollout engine, re-measure, then re-enable and compare",
            "Use the generator's logprobs as the behaviour policy instead of recomputing them",
            "Match dtype between generation and training for the logprob computation specifically",
            "Log the gap as a first-class training metric, permanently. It is cheap and it is load-bearing",
        ],
        "cost": (
            "This is the single most expensive bug in modern RL post-training. Importance "
            "ratios are computed against logprobs the trainer never produced, so the policy "
            "gradient is systematically wrong while every dashboard looks healthy."
        ),
        "repro": "syntropyrl demo logprob_divergence",
    },
    "RLD-092": {
        "title": "Chat template / tokenization mismatch",
        "family": "numerics",
        "aka": ["template drift", "special token mismatch"],
        "symptom": (
            "Divergence between generation and training that is large on the first few tokens "
            "and negligible afterwards. Often accompanied by a model that suddenly forgets "
            "its instruction format."
        ),
        "measure": (
            "Ratio of mean |logprob gap| over the first 8 tokens to the gap over the rest of "
            "the sequence. A ratio above 8 with an absolute head gap above 5e-3 is a template "
            "problem, not a numerics problem."
        ),
        "causes": [
            "Generation uses apply_chat_template but the trainer tokenizes raw text (or vice versa)",
            "BOS token added twice, or dropped on one side",
            "add_special_tokens defaulting differently between the two call sites",
            "A system prompt present at rollout time and absent at training time",
        ],
        "fixes": [
            "Print the decoded token IDs from both paths for one sample and diff them character by character",
            "Tokenize once, at rollout time, and pass token IDs to the trainer instead of strings",
            "Assert that trainer token count equals generator token count for every sample in the first batch",
        ],
        "cost": (
            "Cheap to fix, embarrassing to discover late, and it silently corrupts the ratio "
            "for the highest-leverage tokens in the sequence: the first ones."
        ),
        "repro": "syntropyrl demo template_mismatch",
    },
    "RLD-071": {
        "title": "Gradient pathology",
        "family": "numerics",
        "aka": ["gradient spike", "NaN loss"],
        "symptom": "Loss goes to NaN, or reward falls off a cliff and never recovers.",
        "measure": (
            "Non-finite gradient norm, or a peak in the last 12 steps that is more than 20x "
            "the running median and above 1.0."
        ),
        "causes": [
            "Advantage outliers from a single very high or very low reward",
            "Unnormalized advantages combined with a large learning rate",
            "Sequences of length 1 or empty completions producing degenerate ratios",
            "fp16 overflow in the logprob computation",
        ],
        "fixes": [
            "Clip gradients (max_norm=1.0) and clip advantages before the loss",
            "Reduce learning rate by 3x and check whether the spike disappears",
            "Filter out empty or single-token completions before the loss",
            "Switch to bf16 if you are on fp16",
        ],
        "cost": "One spike can undo thousands of good steps, and checkpoints saved after it are worthless.",
        "repro": "syntropyrl demo gradient_spike",
    },
    "RLD-001": {
        "title": "Advantage collapse (degenerate groups)",
        "family": "optimization",
        "aka": ["zero advantage", "GRPO group collapse"],
        "symptom": (
            "Reward is flat. Gradients are tiny but finite. Nothing is obviously broken and "
            "nothing is happening."
        ),
        "measure": (
            "Fraction of GRPO groups where every rollout received an identical reward, plus "
            "the standard deviation of within-group advantages."
        ),
        "causes": [
            "Prompts far too hard: every rollout fails, so every advantage is zero",
            "Prompts far too easy: every rollout succeeds, so every advantage is zero",
            "Binary reward with a small group size, so ties dominate",
            "Group size of 1, which makes group-relative advantage meaningless",
        ],
        "fixes": [
            "Filter the prompt set to items with pass rate between 0.2 and 0.8 for the current policy",
            "Increase group size (8 to 16) so ties are less likely",
            "Add partial credit to the reward so it is not purely binary",
            "Curriculum: re-score prompt difficulty every N steps and resample",
        ],
        "cost": (
            "You pay full price for rollouts that carry no learning signal. This is the most "
            "common reason a GRPO run 'just does nothing'."
        ),
        "repro": "syntropyrl demo advantage_collapse",
    },
    "RLD-021": {
        "title": "KL blowup",
        "family": "optimization",
        "aka": ["reference drift", "policy escape"],
        "symptom": (
            "Reward climbs for a while, then collapses. Generations become strange, repetitive, "
            "or switch language mid-sentence."
        ),
        "measure": "KL to the reference policy growing 5x over the window and exceeding 0.5.",
        "causes": [
            "KL coefficient too small, or KL penalty accidentally disabled",
            "Learning rate too high for the batch size",
            "Reward model being exploited off-distribution, which pulls the policy away fast",
            "Reference model updated or reloaded mid-run",
        ],
        "fixes": [
            "Raise the KL coefficient, or switch to adaptive KL targeting a fixed value",
            "Lower the learning rate; RL post-training usually wants 1e-6 territory, not 1e-5",
            "Add reward clipping so a single exploitable prompt cannot dominate",
            "Verify the reference model is frozen and is the model you think it is",
        ],
        "cost": "By the time output quality visibly degrades, you have usually wasted hours of compute.",
        "repro": "syntropyrl demo kl_blowup",
    },
    "RLD-055": {
        "title": "Clip saturation (stale off-policy data)",
        "family": "optimization",
        "aka": ["clipfrac saturation", "too many inner epochs"],
        "symptom": "Training is slow to improve and oddly insensitive to the learning rate.",
        "measure": (
            "Mean PPO clip fraction over 10 steps. Healthy is 0.02 to 0.20; above 0.30 most of "
            "your batch is being discarded by the clip."
        ),
        "causes": [
            "Too many optimizer epochs per rollout batch, so late updates are far off-policy",
            "Rollout batch reused across many minibatches without recomputing logprobs",
            "Learning rate too high, pushing the ratio outside the trust region immediately",
            "Asynchronous rollout workers lagging several policy versions behind",
        ],
        "fixes": [
            "Reduce inner epochs to 1 or 2",
            "Reduce the number of minibatches per rollout batch",
            "Bound rollout staleness explicitly in async setups (max policy-version lag)",
            "Lower the learning rate and re-check clip_frac before anything else",
        ],
        "cost": "You are paying for rollouts and then throwing most of the gradient away.",
        "repro": "syntropyrl demo stale_offpolicy",
    },
    "RLD-084": {
        "title": "Value function divergence",
        "family": "optimization",
        "aka": ["critic divergence"],
        "symptom": "Critic loss rises steadily while the policy keeps training. Advantages get noisier over time.",
        "measure": "Second half of the value-loss window more than 1.5x the first half, with a positive slope.",
        "causes": [
            "Value head learning rate too high, or shared trunk fighting the policy objective",
            "Reward scale changed mid-run (new reward term, different normalization)",
            "Value targets computed with a different discount or GAE lambda than intended",
            "Critic initialized from a model that never saw this reward distribution",
        ],
        "fixes": [
            "Normalize rewards, or normalize value targets, and keep it consistent",
            "Give the value head its own lower learning rate",
            "Warm up the critic for a few hundred steps before enabling policy updates",
            "Consider a critic-free method (GRPO) if the critic keeps fighting you",
        ],
        "cost": "A diverging critic injects noise into every advantage, so the policy learns from garbage.",
        "repro": "syntropyrl demo value_divergence",
    },
    "RLD-031": {
        "title": "Reward hacking (proxy/truth divergence)",
        "family": "reward",
        "aka": ["verifier gaming", "format exploit"],
        "symptom": (
            "The best-looking run you have ever had. Reward climbs smoothly. Then you read the "
            "outputs and they are nonsense that happens to satisfy the grader."
        ),
        "measure": (
            "Reward rising while a logged ground-truth metric stays flat, and/or an n-gram whose "
            "presence predicts materially higher reward at a large standardized effect size "
            "(t >= 6, searched across candidate phrases so chance findings do not fire)."
        ),
        "causes": [
            "Verifier matching a prefix or a format marker instead of the answer",
            "Regex or string-equality grader accepting a superset of correct answers",
            "LLM judge rewarding confidence, length, or politeness",
            "Reward model overoptimized off-distribution",
        ],
        "fixes": [
            "Hold out a verifier the policy never trains against, and evaluate on it",
            "Read the 20 highest-reward completions by hand. This always works",
            "Add an adversarial unit test: does an empty or nonsense answer score above zero?",
            "Log proxy reward and ground truth on the same chart, permanently",
        ],
        "cost": (
            "The failure mode most likely to get shipped, because every metric says success. "
            "Detection requires at least one metric the policy cannot optimize."
        ),
        "repro": "syntropyrl demo reward_hacking",
    },
    "RLD-033": {
        "title": "Length exploit",
        "family": "reward",
        "aka": ["verbosity bias", "length hacking"],
        "symptom": "Completions get longer every hour. Reward tracks length. Answers get no better.",
        "measure": (
            "Pearson correlation between completion length and reward across recent rollouts "
            "(>= 0.45) combined with mean length growth of at least 25% across the window."
        ),
        "causes": [
            "Judge or reward model prefers verbose answers",
            "Reward summed over tokens instead of averaged, or applied per token by accident",
            "Repetition earning partial credit from a fuzzy matcher",
            "No length penalty while the context budget still has headroom",
        ],
        "fixes": [
            "Normalize reward per sequence, not per token, unless you truly mean per token",
            "Add an explicit length penalty or a token budget to the reward",
            "Score length-matched pairs to test whether your judge is length-biased",
        ],
        "cost": "Doubles your rollout cost per step while quality stays flat, and it compounds with truncation bias.",
        "repro": "syntropyrl demo length_exploit",
    },
    "RLD-042": {
        "title": "Truncation bias",
        "family": "reward",
        "aka": ["max_new_tokens bias", "clipped rollouts"],
        "symptom": (
            "Reward plateaus exactly when average completion length approaches your generation "
            "limit. The model starts answering suspiciously briefly."
        ),
        "measure": (
            "Fraction of rollouts hitting the length limit (>= 0.12) together with a reward gap "
            "of at least 15% between truncated and complete rollouts."
        ),
        "causes": [
            "max_new_tokens too small for the reasoning the task requires",
            "Truncated sequences scored as incorrect rather than masked out",
            "Reasoning traces growing over training until they hit the ceiling",
            "EOS token never emitted because the template does not teach it",
        ],
        "fixes": [
            "Raise max_new_tokens and re-measure the truncation rate",
            "Mask truncated rollouts out of the loss instead of scoring them zero",
            "Or score them explicitly and deliberately, with a documented penalty",
            "Track the truncation rate as a permanent training metric",
        ],
        "cost": (
            "You are teaching the model that thinking is punished. It learns to answer shorter, "
            "which looks like the model getting worse at reasoning, because it is."
        ),
        "repro": "syntropyrl demo truncation_bias",
    },
    "RLD-060": {
        "title": "Dead reward signal",
        "family": "reward",
        "aka": ["silent verifier failure", "all-zero reward"],
        "symptom": "Nothing moves. Loss is smooth, gradients are finite, reward is a flat line.",
        "measure": "Reward standard deviation across the batch at or below 1e-4 for 12 consecutive steps.",
        "causes": [
            "Reward function raising an exception swallowed by a bare except",
            "Verifier receiving the wrong field (prompt instead of completion)",
            "All completions truncated before reaching the answer",
            "Reward computed on a padded tensor and averaged away to a constant",
        ],
        "fixes": [
            "Unit-test the reward function on three known-good and three known-bad strings",
            "Print one full (prompt, completion, reward) triple every 50 steps and read it",
            "Remove the try/except around the verifier and let it crash loudly",
        ],
        "cost": "The cheapest bug to detect and the most humiliating to find after a weekend of training.",
        "repro": "syntropyrl demo dead_reward",
    },
    "RLD-007": {
        "title": "Entropy collapse",
        "family": "distribution",
        "aka": ["premature convergence", "exploration death"],
        "symptom": (
            "Fast early progress, then a hard plateau. Rollouts within a group become nearly "
            "identical and the model stops discovering anything new."
        ),
        "measure": (
            "Policy entropy down at least 0.55 nats from its early-run baseline and currently "
            "below 0.4, with within-group advantage standard deviation shrinking alongside it."
        ),
        "causes": [
            "No entropy bonus, or a coefficient that is far too small",
            "Sampling temperature below 1.0 during rollouts",
            "Learning rate high enough to sharpen the policy faster than it can explore",
            "Repeatedly training on a narrow prompt set the policy has already mastered",
        ],
        "fixes": [
            "Add or raise the entropy bonus (start around 0.001 to 0.01)",
            "Sample rollouts at temperature 1.0; save low temperature for evaluation",
            "Increase group size so the group still spans different behaviours",
            "Refresh the prompt distribution toward items the policy has not solved",
        ],
        "cost": (
            "Entropy collapse is close to irreversible. Once exploration dies, extra compute "
            "buys nothing, and no amount of later tuning brings the diversity back."
        ),
        "repro": "syntropyrl demo entropy_collapse",
    },
    "RLD-018": {
        "title": "Mode collapse (duplicate rollouts)",
        "family": "distribution",
        "aka": ["identical generations", "greedy rollouts"],
        "symptom": "Rollouts within a group are byte-identical. GRPO advantages are all zero as a result.",
        "measure": "Fraction of duplicate completions within groups averaged over 8 steps, at or above 0.35.",
        "causes": [
            "do_sample=False, or temperature 0, left on in the rollout config",
            "Same seed used for every sample in the group",
            "top_k=1 or an aggressive top_p flattening the distribution",
            "Severe entropy collapse (see RLD-007) reaching its endpoint",
        ],
        "fixes": [
            "Check the sampling config first: do_sample=True, temperature=1.0, top_p >= 0.95",
            "Ensure each rollout in a group uses a different seed",
            "Confirm the inference engine is not caching and replaying identical outputs",
        ],
        "cost": "You are paying for N rollouts and getting one, with zero learning signal from the group.",
        "repro": "syntropyrl demo mode_collapse",
    },
}

FAMILY_ORDER = ["numerics", "optimization", "reward", "distribution"]
FAMILY_BLURB = {
    "numerics": "The math is wrong. Nothing crashes; the gradient is just not the gradient you think it is.",
    "optimization": "The math is right and the update is broken: no signal, too much drift, or a discarded batch.",
    "reward": "The policy is learning exactly what you asked for, which turns out not to be what you wanted.",
    "distribution": "The policy stopped exploring, so there is nothing left to learn from.",
}


def entry(code: str) -> Optional[dict]:
    """Look up one Atlas entry by code, e.g. 'RLD-014'."""
    e = ENTRIES.get(code.upper().strip())
    if e is None:
        return None
    out = dict(e)
    out["code"] = code.upper().strip()
    out["url"] = "{}#{}".format(ATLAS_URL, out["code"])
    return out


def entries_by_family() -> List[dict]:
    """All entries, grouped and ordered for rendering."""
    groups = []
    for fam in FAMILY_ORDER:
        items = [
            dict(e, code=code, url="{}#{}".format(ATLAS_URL, code))
            for code, e in ENTRIES.items()
            if e["family"] == fam
        ]
        if items:
            groups.append({"family": fam, "blurb": FAMILY_BLURB.get(fam, ""), "entries": items})
    return groups


def render_text(code: str, width: int = 88) -> str:
    """Terminal rendering of a single entry, used by `syntropyrl atlas <code>`."""
    import textwrap

    e = entry(code)
    if e is None:
        known = ", ".join(sorted(ENTRIES))
        return "Unknown code {!r}.\nKnown codes: {}".format(code, known)

    def block(label: str, text: str) -> List[str]:
        return ["  {}".format(label)] + [
            "    " + ln for ln in textwrap.wrap(text, width - 4)
        ]

    def bullets(label: str, items: List[str]) -> List[str]:
        out = ["  {}".format(label)]
        for it in items:
            wrapped = textwrap.wrap(it, width - 8)
            out.append("    - " + (wrapped[0] if wrapped else ""))
            out.extend("      " + ln for ln in wrapped[1:])
        return out

    lines = ["", "{}  {}".format(e["code"], e["title"]), "-" * min(width, 88)]
    if e.get("aka"):
        lines.append("  also called: {}".format(", ".join(e["aka"])))
    lines += block("Symptom", e["symptom"])
    lines += block("What syntropyrl measures", e["measure"])
    lines += bullets("Likely causes", e["causes"])
    lines += bullets("Fixes, in order", e["fixes"])
    lines += block("Why it matters", e["cost"])
    lines += ["  Reproduce", "    $ " + e["repro"], "  " + e["url"], ""]
    return "\n".join(lines)
