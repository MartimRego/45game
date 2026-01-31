"""Optimal strategy solver for the 45-point dice-slot game.

This solves the finite-horizon MDP exactly using backward induction with exact
rational arithmetic (fractions.Fraction).

Model:
- Pre-roll state:  (mask, score)
- Post-roll state: (mask, score, roll)
- Terminal reward: +1 if final score >= 45 else -1

The optimal policy maximizes expected terminal reward, which is equivalent to
maximizing the win probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable


TARGET_SCORE = 45


@dataclass(frozen=True)
class Slot:
    name: str

    def points(self, roll: int) -> int:
        raise NotImplementedError


class X3(Slot):
    def points(self, roll: int) -> int:
        return 3 * roll


class Flip(Slot):
    def points(self, roll: int) -> int:
        return 7 - roll


class Plus4Only12(Slot):
    def points(self, roll: int) -> int:
        return roll + 4 if roll in (1, 2) else 0


class X2EvenOnly(Slot):
    def points(self, roll: int) -> int:
        return 2 * roll if roll % 2 == 0 else 0


class Plus4OddOnly(Slot):
    def points(self, roll: int) -> int:
        return roll + 4 if roll % 2 == 1 else 0


class Bin(Slot):
    def points(self, roll: int) -> int:
        return 0


SLOTS: list[Slot] = [
    X3("x3"),
    Flip("flip"),
    Plus4Only12("+4only12"),
    X2EvenOnly("x2even"),
    Plus4OddOnly("+4odd"),
    Bin("bin"),
]

FULL_MASK = (1 << len(SLOTS)) - 1

# Precompute immediate points for speed and consistency.
POINTS: list[list[int]] = [[0] * 7 for _ in SLOTS]  # POINTS[slot_index][roll]
for i, slot in enumerate(SLOTS):
    for r in range(1, 7):
        POINTS[i][r] = slot.points(r)


def terminal_reward(score: int) -> int:
    return 1 if score >= TARGET_SCORE else -1


def iter_actions(mask: int) -> Iterable[int]:
    # Yields slot indices that are still available.
    for i in range(len(SLOTS)):
        if mask & (1 << i):
            yield i


def popcount(mask: int) -> int:
    return mask.bit_count()


def mask_to_names(mask: int) -> list[str]:
    return [SLOTS[i].name for i in range(len(SLOTS)) if mask & (1 << i)]


def mask_to_bits_slot_order(mask: int) -> str:
    """Render a mask as bits in *slot order* (left-to-right): x3..bin.

    Internally, bit i corresponds to SLOTS[i] (LSB is x3). This function prints
    a human-friendly string that aligns with the slot list order.
    """

    return "".join("1" if mask & (1 << i) else "0" for i in range(len(SLOTS)))


def parse_mask(mask_text: str) -> int:
    """Parse a mask from CLI input.

    Accepts:
    - Bare binary like "101010": interpreted in *slot order* (x3..bin)
    - "0b101010": interpreted as a normal base-2 integer (standard binary)
    - Decimal integers like "42" (base-10)
    """

    text = mask_text.strip().lower().replace("_", "")
    if text.startswith("0b"):
        return int(text, 2)
    if text and all(ch in "01" for ch in text):
        if len(text) == len(SLOTS):
            # Slot-order bits: leftmost is SLOTS[0] (x3), rightmost is SLOTS[5] (bin).
            mask = 0
            for i, ch in enumerate(text):
                if ch == "1":
                    mask |= 1 << i
            return mask
        # If it's not 6 bits, fall back to standard base-2 parsing.
        return int(text, 2)
    return int(text, 10)


@lru_cache(maxsize=None)
def v_pre(mask: int, score: int) -> Fraction:
    """Optimal expected terminal reward from pre-roll state (mask, score)."""

    if mask == 0:
        return Fraction(terminal_reward(score), 1)

    total = Fraction(0, 1)
    for roll in range(1, 7):
        total += v_post(mask, score, roll)

    return total / 6


@lru_cache(maxsize=None)
def v_post(mask: int, score: int, roll: int) -> Fraction:
    """Optimal expected terminal reward from post-roll state (mask, score, roll)."""

    best_value: Fraction | None = None
    for action in iter_actions(mask):
        next_mask = mask & ~(1 << action)
        next_score = score + POINTS[action][roll]
        value = v_pre(next_mask, next_score)
        if best_value is None or value > best_value:
            best_value = value

    if best_value is None:
        # Should only happen if mask == 0, which is handled by v_pre.
        return Fraction(terminal_reward(score), 1)

    return best_value


def best_action(mask: int, score: int, roll: int) -> int:
    """Argmax action for a post-roll state.

    Tie-breaking (deterministic):
    1) higher v_pre(next_mask, next_score)
    2) higher immediate points
    3) earlier slot index
    """

    best_idx: int | None = None
    best_value: Fraction | None = None
    best_points: int | None = None

    for action in iter_actions(mask):
        next_mask = mask & ~(1 << action)
        pts = POINTS[action][roll]
        value = v_pre(next_mask, score + pts)

        if best_value is None:
            best_idx, best_value, best_points = action, value, pts
            continue

        if value > best_value:
            best_idx, best_value, best_points = action, value, pts
            continue

        if value == best_value:
            if pts > (best_points or 0):
                best_idx, best_value, best_points = action, value, pts
                continue
            if pts == (best_points or 0) and action < (best_idx or 0):
                best_idx, best_value, best_points = action, value, pts

    assert best_idx is not None
    return best_idx


def win_probability_from_start() -> Fraction:
    start_value = v_pre(FULL_MASK, 0)
    return (start_value + 1) / 2


def expected_terminal_reward_from_start() -> Fraction:
    return v_pre(FULL_MASK, 0)


def reachable_pre_states_under_optimal_policy() -> set[tuple[int, int]]:
    """All (mask,score) pre-roll states reachable if we always act optimally."""

    seen: set[tuple[int, int]] = set()
    frontier: list[tuple[int, int]] = [(FULL_MASK, 0)]

    while frontier:
        mask, score = frontier.pop()
        if (mask, score) in seen:
            continue
        seen.add((mask, score))

        if mask == 0:
            continue

        for roll in range(1, 7):
            action = best_action(mask, score, roll)
            next_mask = mask & ~(1 << action)
            next_score = score + POINTS[action][roll]
            frontier.append((next_mask, next_score))

    return seen


def _compress_score_actions(pairs: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Compress (score, action_name) into contiguous ranges with identical action."""

    if not pairs:
        return []

    pairs.sort(key=lambda t: t[0])
    out: list[tuple[int, int, str]] = []

    start_s, start_a = pairs[0]
    prev_s = start_s

    for s, a in pairs[1:]:
        if a == start_a and s == prev_s + 1:
            prev_s = s
            continue

        out.append((start_s, prev_s, start_a))
        start_s, start_a = s, a
        prev_s = s

    out.append((start_s, prev_s, start_a))
    return out


def render_results_markdown() -> str:
    p_win = win_probability_from_start()
    v0 = expected_terminal_reward_from_start()

    lines: list[str] = []
    lines.append("# Optimal strategy results")
    lines.append("")
    lines.append("## Start-state win probability")
    lines.append("")
    lines.append(f"- Optimal expected terminal reward: `{v0}` ≈ `{float(v0):.6f}`")
    lines.append(f"- Optimal win probability: `{p_win}` ≈ `{float(p_win):.6f}`")
    lines.append("")

    lines.append("## First move (from the start state)")
    lines.append("")
    lines.append("From `(R = all slots, score = 0)`, after observing the roll:")
    lines.append("")
    lines.append("| Roll X | Best slot | Points gained |")
    lines.append("|---:|---|---:|")
    for roll in range(1, 7):
        a = best_action(FULL_MASK, 0, roll)
        lines.append(f"| {roll} | {SLOTS[a].name} | {POINTS[a][roll]} |")
    lines.append("")

    lines.append("## How to query the policy")
    lines.append("")
    lines.append("Run:")
    lines.append("")
    lines.append("- `python solver.py --query --mask <mask> --score <s> --roll <X>`")
    lines.append("")
    lines.append("Where:")
    lines.append("- `mask` is a 6-bit bitmask over the slots in this fixed order: `x3, flip, +4only12, x2even, +4odd, bin`.")
    lines.append("- If you pass 6-bit binary like `101010`, it’s interpreted in this *slot order* (left-to-right is `x3..bin`).")
    lines.append("- If you pass `0b101010`, it’s interpreted as standard binary. You can also pass decimal like `42`.")
    lines.append("- A `1` bit means the slot is still available.")
    lines.append("- The query output shows `Win% before roll` (pre-roll state) and `Win% after seeing roll` (post-roll state, acting optimally).")
    lines.append("")

    lines.append("## Simulate a full game")
    lines.append("")
    lines.append("You can simulate a full 6-roll game and see if you would win when following the optimal policy:")
    lines.append("")
    lines.append("- `python solver.py --simulate 132563`")
    lines.append("- Add `--trace` to see the chosen slot each turn.")
    lines.append("")

    lines.append("## Evaluate skill and luck")
    lines.append("")
    lines.append("You can score a played game by providing both the rolls and the chosen moves:")
    lines.append("")
    lines.append("- `python solver.py --evaluate --rolls 132563 --moves 213645`")
    lines.append("- Add `--trace` to show per-turn skill/luck numbers in the chosen trace.")
    lines.append("")
    lines.append("Move digits are slot numbers in this fixed order:")
    lines.append("- `1=x3, 2=flip, 3=+4only12, 4=x2even, 5=+4odd, 6=bin`")
    lines.append("")

    lines.append("## Policy summary over reachable states")
    lines.append("")
    lines.append("This section lists, for each reachable remaining-slot set `R`, what the optimal choice is for each roll `X` as a function of current score. (It only includes **pre-roll states reachable under the optimal policy**.)")
    lines.append("")

    reachable = reachable_pre_states_under_optimal_policy()

    # Group by mask.
    scores_by_mask: dict[int, set[int]] = {}
    for mask, score in reachable:
        scores_by_mask.setdefault(mask, set()).add(score)

    # Sort masks by number of remaining slots (descending), then mask value.
    def mask_sort_key(m: int) -> tuple[int, int]:
        return (-popcount(m), m)

    for mask in sorted(scores_by_mask.keys(), key=mask_sort_key):
        if mask == 0:
            continue

        remaining_names = ", ".join(mask_to_names(mask))
        lines.append(f"### Remaining slots: `{remaining_names}` (mask `{mask}`, {popcount(mask)} slots)")
        lines.append("")

        scores = sorted(scores_by_mask[mask])
        lines.append(f"Reachable current scores: `{scores[0]}` … `{scores[-1]}` (count {len(scores)})")
        lines.append("")

        for roll in range(1, 7):
            pairs: list[tuple[int, str]] = []
            for s in scores:
                a = best_action(mask, s, roll)
                pairs.append((s, SLOTS[a].name))

            ranges = _compress_score_actions(pairs)
            formatted = ", ".join(
                [
                    (f"{lo}→{hi}: {a}" if lo != hi else f"{lo}: {a}")
                    for (lo, hi, a) in ranges
                ]
            )
            lines.append(f"- Roll `{roll}`: {formatted}")

        lines.append("")

    return "\n".join(lines)


def query(mask: int, score: int, roll: int) -> str:
    p_before = (v_pre(mask, score) + 1) / 2
    p_after_observe = (v_post(mask, score, roll) + 1) / 2

    def best_next_rolls_from(pre_mask: int, pre_score: int) -> tuple[list[int], float] | None:
        """Return (rolls, max_win_prob) for the next roll, from a pre-roll state.

        If pre_mask == 0, the game is over and there is no next roll.

        For a particular roll r, conditional win probability is:
        (v_post(pre_mask, pre_score, r) + 1) / 2
        because after observing r we then act optimally.
        """

        if pre_mask == 0:
            return None

        best_p: float | None = None
        best_rolls: list[int] = []
        for r in range(1, 7):
            p_r = float((v_post(pre_mask, pre_score, r) + 1) / 2)
            if best_p is None or p_r > best_p + 1e-15:
                best_p = p_r
                best_rolls = [r]
            elif abs(p_r - best_p) <= 1e-15:
                best_rolls.append(r)

        assert best_p is not None
        return best_rolls, best_p

    a = best_action(mask, score, roll)
    pts = POINTS[a][roll]
    next_mask = mask & ~(1 << a)
    next_score = score + pts
    p_after_action = (v_pre(next_mask, next_score) + 1) / 2
    best_next_for_opt = best_next_rolls_from(next_mask, next_score)

    # Compute outcomes for all available actions.
    outcomes: list[tuple[float, int, int, int]] = []
    # (win_prob_float, action_index, next_mask, next_score)
    for action in iter_actions(mask):
        pts_any = POINTS[action][roll]
        nm = mask & ~(1 << action)
        ns = score + pts_any
        p = (v_pre(nm, ns) + 1) / 2
        outcomes.append((float(p), action, nm, ns))

    # Sort by win% (desc), then points (desc), then name.
    outcomes.sort(
        key=lambda t: (
            -t[0],
            -POINTS[t[1]][roll],
            SLOTS[t[1]].name,
        )
    )

    remaining = ", ".join(mask_to_names(mask))
    remaining_bits = mask_to_bits_slot_order(mask)
    next_remaining_bits = mask_to_bits_slot_order(next_mask)

    lines: list[str] = []
    lines.append(f"Remaining slots: [{remaining}] (mask={remaining_bits})")
    lines.append(f"Score so far: {score}")
    lines.append(f"Win% before roll: {float(p_before) * 100:.3f}%")
    lines.append(f"Roll: {roll}")
    lines.append(f"Win% after seeing roll (optimal): {float(p_after_observe) * 100:.3f}%")
    lines.append("")
    lines.append(f"Best slot: {SLOTS[a].name} (+{pts} points)")
    lines.append(f"Next state: mask={next_remaining_bits}, score={next_score}")
    lines.append(f"Win% after taking best action: {float(p_after_action) * 100:.3f}%")
    if best_next_for_opt is None:
        lines.append("Next roll to root for: (game already ended)")
    else:
        rolls, p_best = best_next_for_opt
        roll_text = ",".join(str(r) for r in rolls)
        lines.append(f"Next roll to root for: {roll_text} (Win% if rolled: {p_best * 100:.3f}%)")
    lines.append("")
    lines.append("All options (if you choose differently):")
    lines.append("| Choice | Points | Next mask | Next score | Win% | Root-for roll(s) | Win% if rolled |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for p_float, action, nm, ns in outcomes:
        marker = "*" if action == a else ""
        name = f"{SLOTS[action].name}{marker}"
        best_next = best_next_rolls_from(nm, ns)
        if best_next is None:
            root_for = "(ended)"
            root_for_p = "-"
        else:
            rolls, p_best = best_next
            root_for = ",".join(str(r) for r in rolls)
            root_for_p = f"{p_best * 100:.3f}%"
        lines.append(
            f"| {name} | {POINTS[action][roll]} | {mask_to_bits_slot_order(nm)} | {ns} | {p_float * 100:.3f}% | {root_for} | {root_for_p} |"
        )

    lines.append("")
    lines.append("`*` marks the optimal choice.")

    return "\n".join(lines)


def _parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Write results to results.md in the current directory.",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query the optimal action for a given (mask,score,roll).",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        metavar="ROLLS",
        help=(
            "Simulate playing the optimal policy for a given roll sequence, e.g. 132563. "
            "Outputs final score and win/lose."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Evaluate a game given --rolls and --moves, producing skill/luck scores and traces. "
            "Example: --evaluate --rolls 132563 --moves 213645"
        ),
    )
    parser.add_argument(
        "--rolls",
        type=str,
        default=None,
        help="With --evaluate: roll sequence as digits 1-6, e.g. 132563",
    )
    parser.add_argument(
        "--moves",
        type=str,
        default=None,
        help=(
            "With --evaluate: chosen slot numbers as digits 1-6, e.g. 213645. "
            "Slot order is 1=x3, 2=flip, 3=+4only12, 4=x2even, 5=+4odd, 6=bin."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="With --simulate, print a step-by-step trace of chosen slots and scores.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help=(
            "Remaining-slots bitmask. You can pass: "
            "(a) 6-bit binary in slot order (e.g. 101010 means x3..bin), "
            "(b) standard binary with 0b prefix (e.g. 0b101010), or "
            "(c) decimal (e.g. 42). "
            "Slot order is: x3, flip, +4only12, x2even, +4odd, bin."
        ),
    )
    parser.add_argument("--score", type=int, default=None)
    parser.add_argument("--roll", type=int, default=None)

    return parser.parse_args()


def _parse_roll_sequence(text: str) -> list[int]:
    cleaned = text.strip().replace(" ", "").replace("_", "")
    if not cleaned:
        raise ValueError("empty roll sequence")
    if any(ch not in "123456" for ch in cleaned):
        raise ValueError("roll sequence must contain only digits 1-6")
    return [int(ch) for ch in cleaned]


def _parse_move_sequence(text: str) -> list[int]:
    """Parse chosen slot numbers as digits 1..6.

    Slot numbering is 1-based in the fixed slot order:
    1=x3, 2=flip, 3=+4only12, 4=x2even, 5=+4odd, 6=bin.
    Returns a list of 0-based slot indices.
    """

    cleaned = text.strip().replace(" ", "").replace("_", "")
    if not cleaned:
        raise ValueError("empty move sequence")
    if any(ch not in "123456" for ch in cleaned):
        raise ValueError("move sequence must contain only digits 1-6")
    return [int(ch) - 1 for ch in cleaned]


def simulate_rolls(rolls: list[int], *, trace: bool = False) -> tuple[int, bool]:
    """Simulate the game under the optimal policy given an observed roll sequence.

    Returns (final_score, did_win).
    """

    mask = FULL_MASK
    score = 0

    if len(rolls) > len(SLOTS):
        raise ValueError(f"roll sequence too long: got {len(rolls)}, expected at most {len(SLOTS)}")

    for turn, roll in enumerate(rolls, start=1):
        if mask == 0:
            break

        action = best_action(mask, score, roll)
        pts = POINTS[action][roll]
        next_mask = mask & ~(1 << action)
        next_score = score + pts

        if trace:
            print(
                f"Turn {turn}: roll={roll} | choose={SLOTS[action].name} (+{pts}) | "
                f"score {score}->{next_score} | mask {mask_to_bits_slot_order(mask)}->{mask_to_bits_slot_order(next_mask)}"
            )

        mask, score = next_mask, next_score

    # If the sequence is shorter than 6, this is the score after consuming the provided rolls.
    did_win = score >= TARGET_SCORE and mask == 0
    return score, did_win


def _prob_pre(mask: int, score: int) -> float:
    return float((v_pre(mask, score) + 1) / 2)


def _prob_post_opt(mask: int, score: int, roll: int) -> float:
    return float((v_post(mask, score, roll) + 1) / 2)


def _format_slot_number(action_index: int) -> str:
    return str(action_index + 1)


def evaluate_game(rolls: list[int], moves: list[int], *, trace: bool = False) -> str:
    """Evaluate a played game (rolls + chosen moves) into skill and luck.

    Skill is based on how close each chosen action is to the best action for the observed roll,
    normalized using min/max over available actions.

    Luck is based on how good each observed roll is compared to the best/worst possible roll,
    normalized using min/max over the six die results.
    """

    if len(rolls) != len(moves):
        raise ValueError(f"rolls and moves must have same length (got {len(rolls)} vs {len(moves)})")
    if len(rolls) != len(SLOTS):
        raise ValueError(f"provide exactly {len(SLOTS)} rolls/moves to evaluate a full game")

    # Track two paths using the same rolls: optimal decisions vs chosen decisions.
    opt_mask, opt_score = FULL_MASK, 0
    chosen_mask, chosen_score = FULL_MASK, 0

    # Per-turn metrics.
    skill_terms: list[tuple[float, float]] = []  # (weight, normalized_skill)
    luck_terms: list[tuple[float, float]] = []  # (weight, normalized_luck)

    # Critical points.
    critical_decision: dict[str, object] | None = None
    critical_roll: dict[str, object] | None = None

    opt_trace_lines: list[str] = []
    chosen_trace_lines: list[str] = []

    for turn, roll in enumerate(rolls, start=1):
        # ===== Luck (based on pre-roll state, before observing the roll) =====
        pre_mask, pre_score = chosen_mask, chosen_score
        q_values = [_prob_post_opt(pre_mask, pre_score, r) for r in range(1, 7)]
        q_best = max(q_values)
        q_worst = min(q_values)
        q_obs = q_values[roll - 1]
        roll_swing = q_best - q_worst
        if roll_swing <= 0:
            # The roll cannot affect the win probability from this state.
            # Luck is undefined here; we treat it as neutral for aggregation (weight=0)
            # and display it as N/A in traces.
            luck_i: float | None = None
        else:
            luck_val = (q_obs - q_worst) / roll_swing
            luck_val = min(1.0, max(0.0, luck_val))
            luck_i = luck_val
        luck_terms.append((roll_swing, 0.5 if luck_i is None else luck_i))

        # Track critical roll (largest swing).
        if critical_roll is None or roll_swing > float(critical_roll["swing"]):
            best_rolls = [i + 1 for i, q in enumerate(q_values) if abs(q - q_best) <= 1e-15]
            worst_rolls = [i + 1 for i, q in enumerate(q_values) if abs(q - q_worst) <= 1e-15]
            critical_roll = {
                "turn": turn,
                "mask": pre_mask,
                "mask_bits": mask_to_bits_slot_order(pre_mask),
                "score": pre_score,
                "observed": roll,
                "best_rolls": best_rolls,
                "worst_rolls": worst_rolls,
                "q_obs": q_obs,
                "q_best": q_best,
                "q_worst": q_worst,
                "swing": roll_swing,
            }

        # ===== Optimal path step =====
        opt_action = best_action(opt_mask, opt_score, roll)
        opt_pts = POINTS[opt_action][roll]
        opt_next_mask = opt_mask & ~(1 << opt_action)
        opt_next_score = opt_score + opt_pts
        opt_trace_lines.append(
            f"Turn {turn}: roll={roll} | choose={_format_slot_number(opt_action)}:{SLOTS[opt_action].name} (+{opt_pts}) | "
            f"score {opt_score}->{opt_next_score} | mask {mask_to_bits_slot_order(opt_mask)}->{mask_to_bits_slot_order(opt_next_mask)}"
        )
        opt_mask, opt_score = opt_next_mask, opt_next_score

        # ===== Chosen path step + skill scoring =====
        chosen_action = moves[turn - 1]
        if not (0 <= chosen_action < len(SLOTS)):
            raise ValueError(f"invalid move at turn {turn}: {chosen_action + 1}")
        if not (chosen_mask & (1 << chosen_action)):
            raise ValueError(
                f"illegal move at turn {turn}: slot {_format_slot_number(chosen_action)}:{SLOTS[chosen_action].name} already used"
            )

        # Compute p(action) for all available actions given the observed roll.
        action_ps: list[tuple[int, float]] = []
        for action in iter_actions(chosen_mask):
            nm = chosen_mask & ~(1 << action)
            ns = chosen_score + POINTS[action][roll]
            action_ps.append((action, _prob_pre(nm, ns)))

        p_best = max(p for _, p in action_ps)
        p_worst = min(p for _, p in action_ps)
        decision_swing = p_best - p_worst

        chosen_next_mask = chosen_mask & ~(1 << chosen_action)
        chosen_pts = POINTS[chosen_action][roll]
        chosen_next_score = chosen_score + chosen_pts
        p_chosen = _prob_pre(chosen_next_mask, chosen_next_score)

        if decision_swing <= 0:
            # All available actions are equally good/bad by win probability.
            skill_i: float | None = None
        else:
            skill_val = (p_chosen - p_worst) / decision_swing
            skill_val = min(1.0, max(0.0, skill_val))
            skill_i = skill_val

        skill_terms.append((decision_swing, 1.0 if skill_i is None else skill_i))

        # Determine best/worst actions for reporting.
        best_actions = [a for a, p in action_ps if abs(p - p_best) <= 1e-15]
        worst_actions = [a for a, p in action_ps if abs(p - p_worst) <= 1e-15]

        if critical_decision is None or decision_swing > float(critical_decision["swing"]):
            critical_decision = {
                "turn": turn,
                "mask": chosen_mask,
                "mask_bits": mask_to_bits_slot_order(chosen_mask),
                "score": chosen_score,
                "roll": roll,
                "chosen_action": chosen_action,
                "chosen_points": chosen_pts,
                "p_chosen": p_chosen,
                "best_actions": best_actions,
                "worst_actions": worst_actions,
                "p_best": p_best,
                "p_worst": p_worst,
                "swing": decision_swing,
            }

        chosen_trace_lines.append(
            f"Turn {turn}: roll={roll} | choose={_format_slot_number(chosen_action)}:{SLOTS[chosen_action].name} (+{chosen_pts}) | "
            f"score {chosen_score}->{chosen_next_score} | mask {mask_to_bits_slot_order(chosen_mask)}->{mask_to_bits_slot_order(chosen_next_mask)}"
            + (
                (
                    " | "
                    + f"skill={(skill_i * 100):.1f}%" if skill_i is not None else " | skill=N/A"
                )
                + (
                    f" luck={(luck_i * 100):.1f}%" if luck_i is not None else " luck=N/A"
                )
                if trace
                else ""
            )
        )

        chosen_mask, chosen_score = chosen_next_mask, chosen_next_score

    # Final aggregation.
    sum_skill_w = sum(w for w, _ in skill_terms)
    sum_luck_w = sum(w for w, _ in luck_terms)
    skill_score = 100.0 if sum_skill_w <= 0 else 100.0 * sum(w * s for w, s in skill_terms) / sum_skill_w
    luck_score = 100.0 if sum_luck_w <= 0 else 100.0 * sum(w * l for w, l in luck_terms) / sum_luck_w

    # Final game outcome for chosen path.
    did_win = chosen_score >= TARGET_SCORE

    # Render report in the requested format.
    out: list[str] = []
    out.append("Here is the review of today's game:")
    out.append("")
    out.append(f"Skill {skill_score:.1f}/100 (How good the decisions were)")
    out.append(f"Luck {luck_score:.1f}/100 (How lucky the rolls were)")
    out.append("")
    out.append(f"Final score: {chosen_score} ({'WIN' if did_win else 'LOSE'}; target={TARGET_SCORE})")
    out.append("")
    out.append("Here's what the optimal path would've been:")
    out.append("")
    out.extend(opt_trace_lines)
    out.append("")
    out.append("Here's the path that was chosen:")
    out.append("")
    out.extend(chosen_trace_lines)
    out.append("")
    out.append("Critical decision:")
    out.append("")
    if critical_decision is None:
        out.append("(none)")
    else:
        cd = critical_decision
        best_str = ", ".join(
            f"{_format_slot_number(a)}:{SLOTS[a].name}" for a in cd["best_actions"]  # type: ignore[index]
        )
        worst_str = ", ".join(
            f"{_format_slot_number(a)}:{SLOTS[a].name}" for a in cd["worst_actions"]  # type: ignore[index]
        )
        out.append(
            f"Turn {cd['turn']} | state mask={cd['mask_bits']} score={cd['score']} | roll={cd['roll']}"
        )
        out.append(
            f"Chosen: {_format_slot_number(cd['chosen_action'])}:{SLOTS[cd['chosen_action']].name} (+{cd['chosen_points']})"
        )
        out.append(
            f"Win% chosen: {float(cd['p_chosen']) * 100:.3f}% | Win% best: {float(cd['p_best']) * 100:.3f}% | Win% worst: {float(cd['p_worst']) * 100:.3f}%"
        )
        out.append(f"Decision swing: {float(cd['swing']) * 100:.3f}%")
        out.append(f"Best option(s): {best_str}")
        out.append(f"Worst option(s): {worst_str}")

    out.append("")
    out.append("Critical roll:")
    out.append("")
    if critical_roll is None:
        out.append("(none)")
    else:
        cr = critical_roll
        out.append(
            f"Turn {cr['turn']} | state mask={cr['mask_bits']} score={cr['score']} | observed roll={cr['observed']}"
        )
        out.append(
            f"Win% if observed roll: {float(cr['q_obs']) * 100:.3f}% | best roll Win%: {float(cr['q_best']) * 100:.3f}% | worst roll Win%: {float(cr['q_worst']) * 100:.3f}%"
        )
        out.append(f"Roll swing: {float(cr['swing']) * 100:.3f}%")
        out.append(f"Best roll(s): {', '.join(str(r) for r in cr['best_rolls'])}")
        out.append(f"Worst roll(s): {', '.join(str(r) for r in cr['worst_rolls'])}")

    return "\n".join(out)


def main() -> int:
    args = _parse_args()

    if args.query:
        if args.mask is None or args.score is None or args.roll is None:
            raise SystemExit("--query requires --mask, --score, and --roll")
        mask = parse_mask(args.mask)
        print(query(mask, args.score, args.roll))
        return 0

    if args.simulate is not None:
        try:
            rolls = _parse_roll_sequence(args.simulate)
        except ValueError as e:
            raise SystemExit(f"Invalid --simulate rolls: {e}")

        try:
            final_score, did_win = simulate_rolls(rolls, trace=args.trace)
        except ValueError as e:
            raise SystemExit(f"Simulation error: {e}")

        if len(rolls) != len(SLOTS):
            print(
                f"Simulated {len(rolls)}/{len(SLOTS)} turns under optimal policy. "
                f"Current score: {final_score}."
            )
            print("Provide 6 rolls to determine win/lose.")
            return 0

        # Full game simulated.
        outcome = "WIN" if did_win else "LOSE"
        print(f"Final score: {final_score}")
        print(f"Outcome: {outcome} (target={TARGET_SCORE})")
        return 0

    if args.evaluate:
        if args.rolls is None or args.moves is None:
            raise SystemExit("--evaluate requires --rolls and --moves")
        try:
            rolls = _parse_roll_sequence(args.rolls)
            moves = _parse_move_sequence(args.moves)
        except ValueError as e:
            raise SystemExit(f"Invalid input: {e}")

        try:
            report = evaluate_game(rolls, moves, trace=args.trace)
        except ValueError as e:
            raise SystemExit(f"Evaluation error: {e}")

        print(report)
        return 0

    p_win = win_probability_from_start()
    v0 = expected_terminal_reward_from_start()

    print(f"Optimal expected terminal reward: {v0} ≈ {float(v0):.6f}")
    print(f"Optimal win probability:         {p_win} ≈ {float(p_win):.6f}")

    if args.write_results:
        md = render_results_markdown()
        with open("results.md", "w", encoding="utf-8") as f:
            f.write(md)
        print("Wrote results.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
