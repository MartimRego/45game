# Optimal strategy results

## Start-state win probability

- Optimal expected terminal reward: `-9511/23328` ≈ `-0.407707`
- Optimal win probability: `13817/46656` ≈ `0.296146`

## First move (from the start state)

From `(R = all slots, score = 0)`, after observing the roll:

| Roll X | Best slot | Points gained |
|---:|---|---:|
| 1 | flip | 6 |
| 2 | +4only12 | 6 |
| 3 | +4odd | 7 |
| 4 | x2even | 8 |
| 5 | +4odd | 9 |
| 6 | x2even | 12 |

## How to query the policy

Run:

- `python solver.py --query --mask <mask> --score <s> --roll <X>`

Where:
- `mask` is a 6-bit bitmask over the slots in this fixed order: `x3, flip, +4only12, x2even, +4odd, bin`.
- If you pass 6-bit binary like `101010`, it’s interpreted in this *slot order* (left-to-right is `x3..bin`).
- If you pass `0b101010`, it’s interpreted as standard binary. You can also pass decimal like `42`.
- A `1` bit means the slot is still available.
- The query output shows `Win% before roll` (pre-roll state) and `Win% after seeing roll` (post-roll state, acting optimally).

## Simulate a full game

You can simulate a full 6-roll game and see if you would win when following the optimal policy:

- `python solver.py --simulate 132563`
- Add `--trace` to see the chosen slot each turn.

## Evaluate skill and luck

You can score a played game by providing both the rolls and the chosen moves:

- `python solver.py --evaluate --rolls 132563 --moves 213645`
- Add `--trace` to show per-turn skill/luck numbers in the chosen trace.

Move digits are slot numbers in this fixed order:
- `1=x3, 2=flip, 3=+4only12, 4=x2even, 5=+4odd, 6=bin`

## Policy summary over reachable states

This section lists, for each reachable remaining-slot set `R`, what the optimal choice is for each roll `X` as a function of current score. (It only includes **pre-roll states reachable under the optimal policy**.)

### Remaining slots: `x3, flip, +4only12, x2even, +4odd, bin` (mask `63`, 6 slots)

Reachable current scores: `0` … `0` (count 1)

- Roll `1`: 0: flip
- Roll `2`: 0: +4only12
- Roll `3`: 0: +4odd
- Roll `4`: 0: x2even
- Roll `5`: 0: +4odd
- Roll `6`: 0: x2even

### Remaining slots: `x3, flip, +4only12, x2even, bin` (mask `47`, 5 slots)

Reachable current scores: `7` … `9` (count 2)

- Roll `1`: 7: flip, 9: +4only12
- Roll `2`: 7: +4only12, 9: +4only12
- Roll `3`: 7: bin, 9: flip
- Roll `4`: 7: bin, 9: x2even
- Roll `5`: 7: x3, 9: x3
- Roll `6`: 7: x2even, 9: x2even

### Remaining slots: `x3, flip, +4only12, +4odd, bin` (mask `55`, 5 slots)

Reachable current scores: `8` … `12` (count 2)

- Roll `1`: 8: flip, 12: +4only12
- Roll `2`: 8: +4only12, 12: +4only12
- Roll `3`: 8: +4odd, 12: +4odd
- Roll `4`: 8: bin, 12: bin
- Roll `5`: 8: +4odd, 12: +4odd
- Roll `6`: 8: x3, 12: x3

### Remaining slots: `x3, flip, x2even, +4odd, bin` (mask `59`, 5 slots)

Reachable current scores: `6` … `6` (count 1)

- Roll `1`: 6: flip
- Roll `2`: 6: flip
- Roll `3`: 6: +4odd
- Roll `4`: 6: x2even
- Roll `5`: 6: +4odd
- Roll `6`: 6: x2even

### Remaining slots: `x3, +4only12, x2even, +4odd, bin` (mask `61`, 5 slots)

Reachable current scores: `6` … `6` (count 1)

- Roll `1`: 6: +4only12
- Roll `2`: 6: +4only12
- Roll `3`: 6: +4odd
- Roll `4`: 6: x2even
- Roll `5`: 6: +4odd
- Roll `6`: 6: x2even

### Remaining slots: `x3, flip, +4only12, x2even` (mask `15`, 4 slots)

Reachable current scores: `7` … `7` (count 1)

- Roll `1`: 7: flip
- Roll `2`: 7: +4only12
- Roll `3`: 7: flip
- Roll `4`: 7: x2even
- Roll `5`: 7: x3
- Roll `6`: 7: x2even

### Remaining slots: `x3, flip, +4only12, +4odd` (mask `23`, 4 slots)

Reachable current scores: `8` … `12` (count 2)

- Roll `1`: 8: flip, 12: +4only12
- Roll `2`: 8: +4only12, 12: +4only12
- Roll `3`: 8: +4odd, 12: +4odd
- Roll `4`: 8: x3, 12: flip
- Roll `5`: 8: +4odd, 12: +4odd
- Roll `6`: 8: x3, 12: x3

### Remaining slots: `x3, flip, +4only12, bin` (mask `39`, 4 slots)

Reachable current scores: `15` … `21` (count 4)

- Roll `1`: 15: flip, 17: flip, 19: flip, 21: +4only12
- Roll `2`: 15: +4only12, 17: +4only12, 19: +4only12, 21: +4only12
- Roll `3`: 15: bin, 17: flip, 19: flip, 21: flip
- Roll `4`: 15: bin, 17: bin, 19: flip, 21: bin
- Roll `5`: 15: bin, 17: bin, 19: x3, 21: x3
- Roll `6`: 15: x3, 17: x3, 19: x3, 21: x3

### Remaining slots: `x3, flip, x2even, bin` (mask `43`, 4 slots)

Reachable current scores: `13` … `15` (count 3)

- Roll `1`: 13→15: flip
- Roll `2`: 13→15: flip
- Roll `3`: 13: bin, 14→15: flip
- Roll `4`: 13→15: x2even
- Roll `5`: 13→15: x3
- Roll `6`: 13→15: x2even

### Remaining slots: `x3, +4only12, x2even, bin` (mask `45`, 4 slots)

Reachable current scores: `13` … `15` (count 2)

- Roll `1`: 13: +4only12, 15: +4only12
- Roll `2`: 13: +4only12, 15: +4only12
- Roll `3`: 13: bin, 15: bin
- Roll `4`: 13: x2even, 15: x2even
- Roll `5`: 13: x3, 15: x3
- Roll `6`: 13: x2even, 15: x2even

### Remaining slots: `flip, +4only12, x2even, bin` (mask `46`, 4 slots)

Reachable current scores: `22` … `24` (count 2)

- Roll `1`: 22: flip, 24: +4only12
- Roll `2`: 22: +4only12, 24: +4only12
- Roll `3`: 22: bin, 24: flip
- Roll `4`: 22: bin, 24: flip
- Roll `5`: 22: bin, 24: bin
- Roll `6`: 22: x2even, 24: x2even

### Remaining slots: `x3, flip, +4odd, bin` (mask `51`, 4 slots)

Reachable current scores: `14` … `18` (count 3)

- Roll `1`: 14: flip, 17→18: flip
- Roll `2`: 14: flip, 17→18: flip
- Roll `3`: 14: +4odd, 17→18: +4odd
- Roll `4`: 14: bin, 17→18: bin
- Roll `5`: 14: +4odd, 17→18: +4odd
- Roll `6`: 14: x3, 17→18: x3

### Remaining slots: `x3, +4only12, +4odd, bin` (mask `53`, 4 slots)

Reachable current scores: `14` … `18` (count 2)

- Roll `1`: 14: +4only12, 18: +4only12
- Roll `2`: 14: +4only12, 18: +4only12
- Roll `3`: 14: +4odd, 18: +4odd
- Roll `4`: 14: bin, 18: bin
- Roll `5`: 14: +4odd, 18: +4odd
- Roll `6`: 14: x3, 18: x3

### Remaining slots: `flip, +4only12, +4odd, bin` (mask `54`, 4 slots)

Reachable current scores: `26` … `30` (count 2)

- Roll `1`: 26: flip, 30: +4only12
- Roll `2`: 26: +4only12, 30: +4only12
- Roll `3`: 26: +4odd, 30: +4odd
- Roll `4`: 26: bin, 30: flip
- Roll `5`: 26: +4odd, 30: +4odd
- Roll `6`: 26: bin, 30: bin

### Remaining slots: `x3, x2even, +4odd, bin` (mask `57`, 4 slots)

Reachable current scores: `11` … `12` (count 2)

- Roll `1`: 11→12: bin
- Roll `2`: 11→12: bin
- Roll `3`: 11→12: +4odd
- Roll `4`: 11→12: x2even
- Roll `5`: 11: x3, 12: +4odd
- Roll `6`: 11: x2even, 12: x3

### Remaining slots: `x3, flip, +4only12` (mask `7`, 3 slots)

Reachable current scores: `15` … `21` (count 4)

- Roll `1`: 15: flip, 17: flip, 19: +4only12, 21: +4only12
- Roll `2`: 15: +4only12, 17: +4only12, 19: +4only12, 21: +4only12
- Roll `3`: 15: x3, 17: flip, 19: flip, 21: flip
- Roll `4`: 15: x3, 17: x3, 19: flip, 21: flip
- Roll `5`: 15: x3, 17: x3, 19: x3, 21: x3
- Roll `6`: 15: x3, 17: x3, 19: x3, 21: x3

### Remaining slots: `x3, flip, x2even` (mask `11`, 3 slots)

Reachable current scores: `13` … `13` (count 1)

- Roll `1`: 13: flip
- Roll `2`: 13: flip
- Roll `3`: 13: flip
- Roll `4`: 13: x2even
- Roll `5`: 13: x3
- Roll `6`: 13: x2even

### Remaining slots: `x3, +4only12, x2even` (mask `13`, 3 slots)

Reachable current scores: `11` … `15` (count 3)

- Roll `1`: 11: +4only12, 13: +4only12, 15: +4only12
- Roll `2`: 11: +4only12, 13: +4only12, 15: +4only12
- Roll `3`: 11: x3, 13: x3, 15: +4only12
- Roll `4`: 11: x3, 13: x2even, 15: x2even
- Roll `5`: 11: x3, 13: x3, 15: x3
- Roll `6`: 11: x3, 13: x2even, 15: x2even

### Remaining slots: `flip, +4only12, x2even` (mask `14`, 3 slots)

Reachable current scores: `22` … `24` (count 2)

- Roll `1`: 22: flip, 24: +4only12
- Roll `2`: 22: +4only12, 24: +4only12
- Roll `3`: 22: flip, 24: flip
- Roll `4`: 22: x2even, 24: flip
- Roll `5`: 22: flip, 24: flip
- Roll `6`: 22: x2even, 24: x2even

### Remaining slots: `x3, flip, +4odd` (mask `19`, 3 slots)

Reachable current scores: `14` … `18` (count 3)

- Roll `1`: 14: flip, 17→18: flip
- Roll `2`: 14: flip, 17→18: flip
- Roll `3`: 14: +4odd, 17→18: +4odd
- Roll `4`: 14: x3, 17→18: flip
- Roll `5`: 14: +4odd, 17→18: +4odd
- Roll `6`: 14: x3, 17→18: x3

### Remaining slots: `x3, +4only12, +4odd` (mask `21`, 3 slots)

Reachable current scores: `14` … `18` (count 3)

- Roll `1`: 14→15: +4only12, 18: +4only12
- Roll `2`: 14→15: +4only12, 18: +4only12
- Roll `3`: 14→15: +4odd, 18: +4odd
- Roll `4`: 14→15: x3, 18: x3
- Roll `5`: 14→15: +4odd, 18: +4odd
- Roll `6`: 14→15: x3, 18: x3

### Remaining slots: `flip, +4only12, +4odd` (mask `22`, 3 slots)

Reachable current scores: `20` … `30` (count 3)

- Roll `1`: 20: flip, 26: flip, 30: +4only12
- Roll `2`: 20: +4only12, 26: +4only12, 30: +4only12
- Roll `3`: 20: +4odd, 26: +4odd, 30: +4odd
- Roll `4`: 20: flip, 26: flip, 30: flip
- Roll `5`: 20: +4odd, 26: +4odd, 30: +4odd
- Roll `6`: 20: flip, 26: flip, 30: flip

### Remaining slots: `x3, x2even, +4odd` (mask `25`, 3 slots)

Reachable current scores: `11` … `12` (count 2)

- Roll `1`: 11→12: +4odd
- Roll `2`: 11→12: x3
- Roll `3`: 11→12: +4odd
- Roll `4`: 11→12: x2even
- Roll `5`: 11: x3, 12: +4odd
- Roll `6`: 11→12: x3

### Remaining slots: `x3, flip, bin` (mask `35`, 3 slots)

Reachable current scores: `21` … `27` (count 7)

- Roll `1`: 21→27: flip
- Roll `2`: 21: bin, 22→27: flip
- Roll `3`: 21→22: bin, 23→24: flip, 25: bin, 26→27: flip
- Roll `4`: 21→23: bin, 24: flip, 25→26: bin, 27: flip
- Roll `5`: 21→23: bin, 24→27: x3
- Roll `6`: 21→27: x3

### Remaining slots: `x3, +4only12, bin` (mask `37`, 3 slots)

Reachable current scores: `21` … `27` (count 5)

- Roll `1`: 21: bin, 22→23: +4only12, 25: +4only12, 27: +4only12
- Roll `2`: 21→23: +4only12, 25: +4only12, 27: +4only12
- Roll `3`: 21→23: bin, 25: bin, 27: bin
- Roll `4`: 21→23: bin, 25: bin, 27: bin
- Roll `5`: 21→23: bin, 25: x3, 27: x3
- Roll `6`: 21→23: x3, 25: x3, 27: x3

### Remaining slots: `flip, +4only12, bin` (mask `38`, 3 slots)

Reachable current scores: `33` … `39` (count 6)

- Roll `1`: 33→35: flip, 36→37: +4only12, 39: flip
- Roll `2`: 33→37: +4only12, 39: +4only12
- Roll `3`: 33→34: bin, 35→37: flip, 39: flip
- Roll `4`: 33→35: bin, 36→37: flip, 39: flip
- Roll `5`: 33→37: bin, 39: flip
- Roll `6`: 33→37: bin, 39: flip

### Remaining slots: `x3, x2even, bin` (mask `41`, 3 slots)

Reachable current scores: `18` … `21` (count 4)

- Roll `1`: 18→21: bin
- Roll `2`: 18→21: bin
- Roll `3`: 18→21: bin
- Roll `4`: 18: bin, 19→20: x2even, 21: x3
- Roll `5`: 18→21: x3
- Roll `6`: 18: x2even, 19→20: x3, 21: x2even

### Remaining slots: `flip, x2even, bin` (mask `42`, 3 slots)

Reachable current scores: `28` … `30` (count 3)

- Roll `1`: 28→30: flip
- Roll `2`: 28→30: flip
- Roll `3`: 28: bin, 29→30: flip
- Roll `4`: 28→29: bin, 30: flip
- Roll `5`: 28→30: bin
- Roll `6`: 28→30: x2even

### Remaining slots: `+4only12, x2even, bin` (mask `44`, 3 slots)

Reachable current scores: `27` … `30` (count 3)

- Roll `1`: 27: bin, 28: +4only12, 30: +4only12
- Roll `2`: 27→28: +4only12, 30: +4only12
- Roll `3`: 27→28: bin, 30: bin
- Roll `4`: 27→28: bin, 30: bin
- Roll `5`: 27→28: bin, 30: bin
- Roll `6`: 27→28: x2even, 30: x2even

### Remaining slots: `x3, +4odd, bin` (mask `49`, 3 slots)

Reachable current scores: `19` … `24` (count 5)

- Roll `1`: 19→20: bin, 22→24: +4odd
- Roll `2`: 19→20: bin, 22→24: bin
- Roll `3`: 19: bin, 20: +4odd, 22→24: +4odd
- Roll `4`: 19→20: bin, 22→23: bin, 24: x3
- Roll `5`: 19→20: +4odd, 22: +4odd, 23: x3, 24: +4odd
- Roll `6`: 19→20: x3, 22→24: x3

### Remaining slots: `flip, +4odd, bin` (mask `50`, 3 slots)

Reachable current scores: `32` … `36` (count 3)

- Roll `1`: 32: flip, 35→36: flip
- Roll `2`: 32: flip, 35→36: flip
- Roll `3`: 32: +4odd, 35→36: +4odd
- Roll `4`: 32: bin, 35→36: flip
- Roll `5`: 32: +4odd, 35→36: +4odd
- Roll `6`: 32: bin, 35→36: bin

### Remaining slots: `+4only12, +4odd, bin` (mask `52`, 3 slots)

Reachable current scores: `32` … `36` (count 3)

- Roll `1`: 32→33: +4only12, 36: +4only12
- Roll `2`: 32→33: +4only12, 36: +4only12
- Roll `3`: 32→33: +4odd, 36: +4odd
- Roll `4`: 32→33: bin, 36: bin
- Roll `5`: 32→33: +4odd, 36: +4odd
- Roll `6`: 32→33: bin, 36: bin

### Remaining slots: `x2even, +4odd, bin` (mask `56`, 3 slots)

Reachable current scores: `26` … `30` (count 2)

- Roll `1`: 26: bin, 30: +4odd
- Roll `2`: 26: bin, 30: bin
- Roll `3`: 26: +4odd, 30: +4odd
- Roll `4`: 26: bin, 30: x2even
- Roll `5`: 26: +4odd, 30: +4odd
- Roll `6`: 26: x2even, 30: x2even

### Remaining slots: `x3, flip` (mask `3`, 2 slots)

Reachable current scores: `21` … `27` (count 7)

- Roll `1`: 21→27: flip
- Roll `2`: 21: x3, 22→27: flip
- Roll `3`: 21→22: x3, 23→27: flip
- Roll `4`: 21→23: x3, 24→27: flip
- Roll `5`: 21→27: x3
- Roll `6`: 21→27: x3

### Remaining slots: `x3, +4only12` (mask `5`, 2 slots)

Reachable current scores: `21` … `27` (count 6)

- Roll `1`: 21→25: +4only12, 27: +4only12
- Roll `2`: 21→25: +4only12, 27: +4only12
- Roll `3`: 21→25: x3, 27: +4only12
- Roll `4`: 21→25: x3, 27: x3
- Roll `5`: 21→25: x3, 27: x3
- Roll `6`: 21→25: x3, 27: x3

### Remaining slots: `flip, +4only12` (mask `6`, 2 slots)

Reachable current scores: `24` … `39` (count 11)

- Roll `1`: 24: flip, 27: flip, 29→30: flip, 32→35: flip, 36→37: +4only12, 39: flip
- Roll `2`: 24: +4only12, 27: +4only12, 29→30: +4only12, 32→37: +4only12, 39: +4only12
- Roll `3`: 24: flip, 27: flip, 29→30: flip, 32→37: flip, 39: flip
- Roll `4`: 24: flip, 27: flip, 29→30: flip, 32→37: flip, 39: flip
- Roll `5`: 24: flip, 27: flip, 29→30: flip, 32→37: flip, 39: flip
- Roll `6`: 24: flip, 27: flip, 29→30: flip, 32→37: flip, 39: flip

### Remaining slots: `x3, x2even` (mask `9`, 2 slots)

Reachable current scores: `15` … `21` (count 7)

- Roll `1`: 15→21: x3
- Roll `2`: 15→21: x3
- Roll `3`: 15→21: x3
- Roll `4`: 15→18: x3, 19→20: x2even, 21: x3
- Roll `5`: 15→21: x3
- Roll `6`: 15→17: x3, 18: x2even, 19→20: x3, 21: x2even

### Remaining slots: `flip, x2even` (mask `10`, 2 slots)

Reachable current scores: `28` … `30` (count 3)

- Roll `1`: 28→30: flip
- Roll `2`: 28→30: flip
- Roll `3`: 28→30: flip
- Roll `4`: 28→29: x2even, 30: flip
- Roll `5`: 28→30: flip
- Roll `6`: 28→30: x2even

### Remaining slots: `+4only12, x2even` (mask `12`, 2 slots)

Reachable current scores: `20` … `30` (count 9)

- Roll `1`: 20: +4only12, 22→24: +4only12, 26→30: +4only12
- Roll `2`: 20: +4only12, 22→24: +4only12, 26→30: +4only12
- Roll `3`: 20: +4only12, 22→24: +4only12, 26→30: +4only12
- Roll `4`: 20: x2even, 22→24: x2even, 26→30: x2even
- Roll `5`: 20: +4only12, 22→24: +4only12, 26→30: +4only12
- Roll `6`: 20: x2even, 22→24: x2even, 26→30: x2even

### Remaining slots: `x3, +4odd` (mask `17`, 2 slots)

Reachable current scores: `19` … `24` (count 6)

- Roll `1`: 19→24: +4odd
- Roll `2`: 19→24: x3
- Roll `3`: 19: x3, 20→24: +4odd
- Roll `4`: 19→24: x3
- Roll `5`: 19→22: +4odd, 23: x3, 24: +4odd
- Roll `6`: 19→24: x3

### Remaining slots: `flip, +4odd` (mask `18`, 2 slots)

Reachable current scores: `26` … `36` (count 4)

- Roll `1`: 26: flip, 32: flip, 35→36: flip
- Roll `2`: 26: flip, 32: flip, 35→36: flip
- Roll `3`: 26: +4odd, 32: +4odd, 35→36: +4odd
- Roll `4`: 26: flip, 32: flip, 35→36: flip
- Roll `5`: 26: +4odd, 32: +4odd, 35→36: +4odd
- Roll `6`: 26: flip, 32: flip, 35→36: flip

### Remaining slots: `+4only12, +4odd` (mask `20`, 2 slots)

Reachable current scores: `21` … `36` (count 10)

- Roll `1`: 21: +4only12, 23: +4only12, 26→27: +4only12, 29→33: +4only12, 36: +4only12
- Roll `2`: 21: +4only12, 23: +4only12, 26→27: +4only12, 29→33: +4only12, 36: +4only12
- Roll `3`: 21: +4odd, 23: +4odd, 26→27: +4odd, 29→33: +4odd, 36: +4odd
- Roll `4`: 21: +4only12, 23: +4only12, 26→27: +4only12, 29→33: +4only12, 36: +4only12
- Roll `5`: 21: +4odd, 23: +4odd, 26→27: +4odd, 29→33: +4odd, 36: +4odd
- Roll `6`: 21: +4only12, 23: +4only12, 26→27: +4only12, 29→33: +4only12, 36: +4only12

### Remaining slots: `x2even, +4odd` (mask `24`, 2 slots)

Reachable current scores: `17` … `30` (count 5)

- Roll `1`: 17→18: +4odd, 26: +4odd, 29→30: +4odd
- Roll `2`: 17→18: x2even, 26: x2even, 29→30: x2even
- Roll `3`: 17→18: +4odd, 26: +4odd, 29→30: +4odd
- Roll `4`: 17→18: x2even, 26: x2even, 29→30: x2even
- Roll `5`: 17→18: +4odd, 26: +4odd, 29→30: +4odd
- Roll `6`: 17→18: x2even, 26: x2even, 29→30: x2even

### Remaining slots: `x3, bin` (mask `33`, 2 slots)

Reachable current scores: `27` … `33` (count 7)

- Roll `1`: 27→33: bin
- Roll `2`: 27→33: bin
- Roll `3`: 27→33: bin
- Roll `4`: 27→32: bin, 33: x3
- Roll `5`: 27→29: bin, 30→33: x3
- Roll `6`: 27→33: x3

### Remaining slots: `flip, bin` (mask `34`, 2 slots)

Reachable current scores: `39` … `45` (count 7)

- Roll `1`: 39→45: flip
- Roll `2`: 39: bin, 40→45: flip
- Roll `3`: 39→40: bin, 41→45: flip
- Roll `4`: 39→41: bin, 42→45: flip
- Roll `5`: 39→42: bin, 43→45: flip
- Roll `6`: 39→43: bin, 44→45: flip

### Remaining slots: `+4only12, bin` (mask `36`, 2 slots)

Reachable current scores: `39` … `45` (count 6)

- Roll `1`: 39: bin, 40→43: +4only12, 45: +4only12
- Roll `2`: 39→43: +4only12, 45: +4only12
- Roll `3`: 39→43: bin, 45: +4only12
- Roll `4`: 39→43: bin, 45: +4only12
- Roll `5`: 39→43: bin, 45: +4only12
- Roll `6`: 39→43: bin, 45: +4only12

### Remaining slots: `x2even, bin` (mask `40`, 2 slots)

Reachable current scores: `33` … `39` (count 7)

- Roll `1`: 33→39: bin
- Roll `2`: 33→39: bin
- Roll `3`: 33→39: bin
- Roll `4`: 33→36: bin, 37→39: x2even
- Roll `5`: 33→39: bin
- Roll `6`: 33→39: x2even

### Remaining slots: `+4odd, bin` (mask `48`, 2 slots)

Reachable current scores: `36` … `42` (count 7)

- Roll `1`: 36→39: bin, 40→42: +4odd
- Roll `2`: 36→42: bin
- Roll `3`: 36→37: bin, 38→42: +4odd
- Roll `4`: 36→42: bin
- Roll `5`: 36→42: +4odd
- Roll `6`: 36→42: bin

### Remaining slots: `x3` (mask `1`, 1 slots)

Reachable current scores: `24` … `33` (count 10)

- Roll `1`: 24→33: x3
- Roll `2`: 24→33: x3
- Roll `3`: 24→33: x3
- Roll `4`: 24→33: x3
- Roll `5`: 24→33: x3
- Roll `6`: 24→33: x3

### Remaining slots: `flip` (mask `2`, 1 slots)

Reachable current scores: `27` … `45` (count 16)

- Roll `1`: 27: flip, 30→31: flip, 33→45: flip
- Roll `2`: 27: flip, 30→31: flip, 33→45: flip
- Roll `3`: 27: flip, 30→31: flip, 33→45: flip
- Roll `4`: 27: flip, 30→31: flip, 33→45: flip
- Roll `5`: 27: flip, 30→31: flip, 33→45: flip
- Roll `6`: 27: flip, 30→31: flip, 33→45: flip

### Remaining slots: `+4only12` (mask `4`, 1 slots)

Reachable current scores: `25` … `45` (count 20)

- Roll `1`: 25→43: +4only12, 45: +4only12
- Roll `2`: 25→43: +4only12, 45: +4only12
- Roll `3`: 25→43: +4only12, 45: +4only12
- Roll `4`: 25→43: +4only12, 45: +4only12
- Roll `5`: 25→43: +4only12, 45: +4only12
- Roll `6`: 25→43: +4only12, 45: +4only12

### Remaining slots: `x2even` (mask `8`, 1 slots)

Reachable current scores: `18` … `39` (count 22)

- Roll `1`: 18→39: x2even
- Roll `2`: 18→39: x2even
- Roll `3`: 18→39: x2even
- Roll `4`: 18→39: x2even
- Roll `5`: 18→39: x2even
- Roll `6`: 18→39: x2even

### Remaining slots: `+4odd` (mask `16`, 1 slots)

Reachable current scores: `21` … `42` (count 21)

- Roll `1`: 21→23: +4odd, 25→42: +4odd
- Roll `2`: 21→23: +4odd, 25→42: +4odd
- Roll `3`: 21→23: +4odd, 25→42: +4odd
- Roll `4`: 21→23: +4odd, 25→42: +4odd
- Roll `5`: 21→23: +4odd, 25→42: +4odd
- Roll `6`: 21→23: +4odd, 25→42: +4odd

### Remaining slots: `bin` (mask `32`, 1 slots)

Reachable current scores: `45` … `51` (count 7)

- Roll `1`: 45→51: bin
- Roll `2`: 45→51: bin
- Roll `3`: 45→51: bin
- Roll `4`: 45→51: bin
- Roll `5`: 45→51: bin
- Roll `6`: 45→51: bin
