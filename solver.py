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
