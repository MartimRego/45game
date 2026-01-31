# 45-Point Dice Slot Game — Markov Chain / MDP Breakdown

## 1) Restating the game

- We play **6 turns**.
- On each turn we:
  1. **Roll one fair six-sided die** and observe the outcome $X \in \{1,2,3,4,5,6\}$.
  2. **Choose one empty slot** (from a fixed set of 6 slots) and place that die there.
  3. Earn **immediate points** determined by the chosen slot and the observed roll.
- Each slot can be used **at most once**.
- After 6 turns all slots are filled and we compute the **total score**.
- We **win** iff total score $\ge 45$.

The 6 slots and their scoring rules (given an observed roll $X$):

1. **x3**: score $3X$
2. **flip**: score $7-X$ (i.e., 1→6, 2→5, 3→4, 4→3, 5→2, 6→1)
3. **+4 (only for 1 or 2)**: score $X+4$ if $X\in\{1,2\}$ else 0
4. **x2 (even only)**: score $2X$ if $X\in\{2,4,6\}$ else 0
5. **+4odd**: score $X+4$ if $X\in\{1,3,5\}$ else 0
6. **Bin**: score 0 always (a “discard”)

## 2) What we want to compute

We want an **optimal strategy**: for every possible situation during play and every observed roll $X$, decide which remaining slot to use to **maximize the probability of eventually winning** (reaching total $\ge 45$ after all 6 slots are used).

Outputs we ultimately want (later steps):

- The **optimal policy**: “If the remaining slots are … and current total is … and you roll $X$, pick slot …”.
- The **probability of winning under the optimal policy** from the start state.
- Optionally: intuition tables/plots like “when to spend x3 vs keep it for later”, and how sensitive win-prob is to each decision.

## 3) Why this fits Markov chains (MDP framing)

The game is not purely a Markov chain by itself because we control decisions. The correct formal model is a **finite-horizon Markov Decision Process (MDP)**.

- The **Markov property** holds if we define the state to contain everything needed for future decisions:
  - which slots remain unused
  - the current accumulated score (or enough info to determine it)
  - (optionally) the current observed roll, if we want a purely “action depends on state” formulation

Given that state, the next roll is independent of history, and the effect of choosing a slot depends only on the current roll and current remaining set.

Once we compute an optimal policy, the game induced by following that fixed policy *does* become an ordinary **Markov chain** over states.

## 4) State representation

A clean state definition that makes the “roll then choose” timing explicit is to use two kinds of states:

- **Pre-roll state**: $(R, s)$
- **Post-roll state**: $(R, s, X)$

Where:

- $R$ is the set of **remaining (unused) slots**. Initially $R$ is all 6 slots.
- $s$ is the **current total score** accumulated so far.
- $X \in \{1,2,3,4,5,6\}$ is the current observed die roll.

We do **not** need a separate turn counter $t$ because it’s derivable from $R$:

$$
t = 6 - |R|.
$$

Notes:

- Because there are only 6 slots, $R$ can be represented as a **6-bit mask** (0–63).
- The score $s$ is bounded (max is when you do very well). A loose upper bound is $3\cdot 6 + 3\cdot 6 + \dots$ etc, but in practice it’s small enough to enumerate exactly.

## 5) Action space

At each nonterminal **post-roll** state we choose an action:

- Action $a \in R$ = pick one of the **currently available slots**.

Important: The action is chosen **after** seeing $X$ (this is crucial).

## 6) Transition model

From a **pre-roll** state $(R, s)$:

1. Nature draws a roll $X$ uniformly from $\{1,2,3,4,5,6\}$.
2. We transition to the corresponding **post-roll** state $(R, s, X)$.

From a **post-roll** state $(R, s, X)$:

1. We choose an available slot $a \in R$.
2. We receive immediate points $g(a, X)$.
3. The next state is the next **pre-roll** state:

$$
(R', s') = (R \setminus \{a\},\; s + g(a,X)).
$$

This is a finite process: after 6 actions we reach $R=\emptyset$.

## 7) Reward / scoring function

Define $g(\text{slot},X)$ exactly:

- $g(\text{x3},X)=3X$
- $g(\text{flip},X)=7-X$
- $g(\text{+4only12},X)=\begin{cases}X+4 & X\in\{1,2\}\\0&\text{otherwise}\end{cases}$
- $g(\text{x2even},X)=\begin{cases}2X & X\in\{2,4,6\}\\0&\text{otherwise}\end{cases}$
- $g(\text{+4odd},X)=\begin{cases}X+4 & X\in\{1,3,5\}\\0&\text{otherwise}\end{cases}$
- $g(\text{bin},X)=0$

(We can rename these slots in code later, but the above is the intended math.)

## 8) Terminal condition and objective

Terminal states: $R=\emptyset$.

We will use a terminal reward that matches the goal (win vs lose), rather than “maximize total points”. Define:

$$
r(s)=\begin{cases}+1 & s\ge 45\\-1& s<45\end{cases}
$$

All nonterminal rewards are 0; only the terminal reward is nonzero.

Objective: choose a policy $\pi$ maximizing expected terminal reward:

$$
\mathbb{E}_{\pi}[r(\text{final score})].
$$

This is equivalent to maximizing the win probability, because:

$$
\mathbb{E}[r] = (+1)\Pr(\text{win}) + (-1)\Pr(\text{lose}) = 2\Pr(\text{win}) - 1.
$$

Since $2p-1$ is strictly increasing in $p$, any policy that maximizes $\mathbb{E}[r]$ also maximizes $\Pr(\text{win})$.

## 9) Dynamic programming / “Markov chain” computation idea

Because this is small and has a fixed horizon, we can solve it exactly by **backward induction** (a standard MDP solve).

Define value functions for the two state types:

- $V_{\text{pre}}(R,s)$ = optimal expected terminal reward starting from the pre-roll state $(R,s)$.
- $V_{\text{post}}(R,s,X)$ = optimal expected terminal reward starting from the post-roll state $(R,s,X)$.

Base case (no slots remaining):

$$
V_{\text{pre}}(\emptyset, s)=r(s).
$$

Recurrence (explicit post-roll states):

- From a pre-roll state, we average over the possible rolls:

$$
V_{\text{pre}}(R,s)=\frac{1}{6}\sum_{X=1}^{6} V_{\text{post}}(R,s,X).
$$

- From a post-roll state, we choose the best available slot:

$$
V_{\text{post}}(R,s,X)=\max_{a\in R} V_{\text{pre}}(R\setminus\{a\},\; s+g(a,X)).
$$

This produces:

- the optimal expected terminal reward from the start state: $V_{\text{pre}}(R_0,0)$ where $R_0$ is all slots.
- the optimal win probability via $\Pr(\text{win}) = \frac{V_{\text{pre}}(R_0,0) + 1}{2}$.
- an optimal decision rule by recording which $a$ attains the max for each $(R,s,X)$.

## 10) What “optimal strategy” means here

An optimal strategy is a mapping:

$$
\pi(R,s,X) \in R
$$

i.e. given remaining slots $R$, current score $s$, and observed roll $X$, choose an available slot.

There can be ties (multiple optimal actions); we can pick any tie-breaking rule.

## 11) Practical considerations (implementation notes)

Later, when implementing, we’ll need to decide:

- Score range to track: easiest is to track all reachable $s$ exactly.
- Memoization: compute $V(R,s)$ via recursion + caching, or iterative DP by increasing number of used slots.
- Policy extraction: store argmax choices for each state and roll.

The state space is small:

- $|R|$ has at most $2^6=64$ possibilities.
- $s$ has a manageable number of reachable values.
- For each $(R,s)$ we consider 6 roll outcomes and up to 6 actions.

So this is very feasible to compute exactly.

## 12) Sanity checks / clarifications to confirm

The above model assumes:

- Each of the 6 turns is an independent fair d6 roll.
- We roll exactly **one** die per turn and place it in exactly **one** unused slot.
- The “+4 (only for 1 or 2)” slot gives **0** for 3–6 (no partial credit).
- The “x2 (even only)” slot gives **0** for odd results.
- We win strictly by threshold: score $\ge 45$.

If all of the above matches your intent, then the DP/MDP formulation here is the exact Markov-style solution approach.

---

If you want, next I can implement the solver (compute $V(R_0,0)$ and emit a readable strategy table) and save results (policy + win probability) into the repo.
