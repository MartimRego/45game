# 45game CLI usage

This workspace contains a single Python program, `solver.py`, which:
- Computes the *optimal* policy for the 6-turn dice/slot game.
- Lets you query decisions for a specific game state.
- Simulates optimal play on a roll sequence.
- Evaluates a played game (skill vs luck) from rolls + chosen moves.

## Quick start

If you already have a virtualenv in `.venv`:

```bash
source .venv/bin/activate
python solver.py --write-results
```

Or run via the venv Python directly:

```bash
.venv/bin/python solver.py --write-results
```

## Slot order (important)

All masks and move numbers use this fixed slot order:

| Slot # | Name | Description |
|---:|---|---|
| 1 | `x3` | scores `3×roll` |
| 2 | `flip` | scores `7-roll` |
| 3 | `+4only12` | scores `roll+4` only if roll is 1 or 2, else 0 |
| 4 | `x2even` | scores `2×roll` only if roll is even, else 0 |
| 5 | `+4odd` | scores `roll+4` only if roll is odd, else 0 |
| 6 | `bin` | always scores 0 |

## 1) Write a results report

Generates `results.md` (start-state win probability, first-move table, and a policy summary over reachable states).

```bash
python solver.py --write-results
```

## 2) Query the optimal decision for a specific state

```bash
python solver.py --query --mask <mask> --score <score> --roll <roll>
```

### Mask formats

`--mask` describes which slots are still available.

- **6-bit binary without `0b`** (e.g. `101010`) is interpreted in **slot order** (left-to-right = `x3..bin`).
- **Binary with `0b`** (e.g. `0b101010`) is interpreted as standard binary.
- **Decimal** (e.g. `42`) is accepted.

A `1` bit means “slot still available”.

### Example

From the start state (all slots available = `111111`), score 0, after rolling a 4:

```bash
python solver.py --query --mask 111111 --score 0 --roll 4
```

The output includes:
- Win% before roll (pre-roll)
- Win% after seeing the roll (optimal play)
- Best slot and next state
- A table of all available choices with their Win%
- “Next roll to root for” from the resulting state

## 3) Simulate optimal play for a full game

Provide a roll sequence (digits 1–6). This plays the *optimal* policy for those rolls and prints the final score and WIN/LOSE.

```bash
python solver.py --simulate 132563
```

Add `--trace` for a per-turn trace:

```bash
python solver.py --simulate 132563 --trace
```

Roll sequences may include spaces/underscores (e.g. `1 3 2_5 6 3`).

## 4) Evaluate a played game (skill and luck)

Provide:
- `--rolls`: the roll sequence (digits 1–6)
- `--moves`: the chosen slot numbers (digits 1–6 in the slot order table above)

```bash
python solver.py --evaluate --rolls 132563 --moves 253146
```

Add `--trace` to include per-turn skill/luck values in the chosen-path trace:

```bash
python solver.py --evaluate --rolls 132563 --moves 253146 --trace
```

### What the evaluation prints

- **Skill (0–100)**: how close each decision was to optimal (weighted by how much the decision could change Win%).
- **Luck (0–100)**: how favorable each observed roll was compared to other possible rolls *from the same state* (weighted by how much the roll could change Win%).
- A trace for the chosen path, and (when mistakes were made) the optimal path for comparison.
- A Win% timeline.
- A “Top mistakes” list (biggest Win% regret).

### “N/A” in per-turn skill/luck

Sometimes the game outcome is already forced (guaranteed WIN or guaranteed LOSE) no matter what you do or roll. In those turns, per-turn skill/luck is shown as `N/A`.

### Perfect-play special case

If your moves match the optimal policy for every roll (no mistakes), the evaluation:
- Omits the “optimal path would’ve been” section.
- Prints a short “Perfect play!” message.
- Shows “No mistakes were made!” under Top mistakes.
