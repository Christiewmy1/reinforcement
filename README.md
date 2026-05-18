# Project 3: Reinforcement Learning

UC Berkeley CS 188 Pacman AI project focused on reinforcement learning. Students implement value iteration and Q-learning agents, then apply them to Gridworld, a simulated robot crawler, and Pacman.

**Final score: 25/25**

## Student Files to Edit

- `valueIterationAgents.py` - Value iteration agent for Gridworld
- `qlearningAgents.py` - Q-learning agents for Gridworld, Crawler, and Pacman
- `analysis.py` - Answers to analysis questions (parameter tuning)

## Running

### Gridworld (manual control)

```bash
python gridworld.py -m
```

### Autograder

```bash
python autograder.py
```

Grade a specific question:

```bash
python autograder.py -q q1
```

### Crawler

```bash
python crawler.py
```

### Pacman with Q-learning

```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

---

## Implementation

### Q1 — Value Iteration (`ValueIterationAgent`)

Implements batch value iteration over a known MDP. Each iteration computes a completely new value table (stored in a fresh `Counter`) before overwriting `self.values`, ensuring all states are updated using values from the *previous* iteration (Bellman backup):

```
V_{k+1}(s) = max_a  sum_{s'} T(s,a,s') [ R(s,a,s') + gamma * V_k(s') ]
```

Terminal states are assigned value 0 and skipped. The agent derives its policy and Q-values directly from the final value table.

### Q2 — Bridge Crossing Analysis (`question2`)

To get the agent to cross the bridge (prefer the +10 exit over the safe +2), noise is set to `0.0`. With no randomness, the agent is never at risk of falling off the narrow bridge, so the high-reward distant exit dominates.

```python
answerDiscount = 0.9
answerNoise    = 0.0
```

### Q3 — Discount Grid Analysis (`question3a`–`question3e`)

The discount grid has a close exit (+1), a far exit (+10), and a cliff (-10) along the bottom. Each sub-question requires a specific combination of discount, noise, and living reward to produce the target policy.

| Question | Goal | Discount | Noise | Living Reward | Key reasoning |
|----------|------|----------|-------|---------------|---------------|
| **3a** | Close exit, risk cliff | 0.1 | 0.0 | -1.0 | Low discount devalues far rewards; no noise means cliff is safe to walk near; negative living reward pushes agent to exit quickly via shortest path |
| **3b** | Close exit, avoid cliff | 0.2 | 0.2 | -1.0 | Noise makes cliff risky (20% chance of slipping south); agent detours north to reach +1 safely |
| **3c** | Far exit, risk cliff | 0.9 | 0.0 | -1.0 | High discount makes +10 worth pursuing; no noise means walking along row 1 (adjacent to cliff) is deterministically safe; negative living reward keeps agent moving |
| **3d** | Far exit, avoid cliff | 0.9 | 0.5 | 0.0 | High discount still makes +10 worthwhile; high noise (50%) makes the cliff-adjacent path very dangerous, so agent routes north to reach +10 safely |
| **3e** | Avoid all exits forever | 0.9 | 0.0 | 10.0 | Large positive living reward (+10 per step) makes the geometric sum of staying alive exceed any terminal reward; agent cycles indefinitely |

### Q4 — Asynchronous Value Iteration (`AsynchronousValueIterationAgent`)

Cyclic single-state updates: on iteration `i`, only `states[i % len(states)]` is updated, directly in-place in `self.values`. This uses the *most recent* values for all other states (unlike batch VI). Terminal states are skipped without update.

### Q5 — Prioritized Sweeping Value Iteration (`PrioritizedSweepingValueIterationAgent`)

Focuses updates on states whose values are most out of date:

1. **Predecessors**: precompute for each state the set of non-terminal states that can reach it with nonzero probability.
2. **Initialize**: push every non-terminal state onto a `util.PriorityQueue` with priority `-|V(s) - max_a Q(s,a)|` (highest error = lowest priority value = processed first).
3. **Iterate**: pop the highest-error state, update its value, then check each predecessor — if its error exceeds `theta`, push/update it in the queue.

This focuses computation where values are changing most, converging faster than cyclic updates.

### Q6 — Q-Learning Agent (`QLearningAgent`)

Tabular Q-learning from raw experience. Key implementation details:

- Q-values stored in a `util.Counter` keyed by `(state, action)` — defaults to 0.0 for unseen pairs.
- **Update rule**: `Q(s,a) += alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]`
- **Tie-breaking**: `computeActionFromQValues` collects all actions matching the max Q-value and picks one with `random.choice`, preventing systematic bias.
- **Exploration**: `getAction` uses `util.flipCoin(epsilon)` to take a random action with probability epsilon, otherwise follows the greedy policy.

### Q7 — Epsilon-Greedy (part of Q6)

The `getAction` method in `QLearningAgent` implements epsilon-greedy exploration. With probability `self.epsilon` a random legal action is chosen (exploration); otherwise the greedy best action from `computeActionFromQValues` is taken (exploitation). Returns `None` at terminal states.

### Q8 — Q-Learning and Pacman Analysis (`question8`)

Returns `'NOT POSSIBLE'`. No fixed epsilon/learning-rate combination can guarantee Pacman learns the optimal policy on the small grid within 50 training episodes — the state space is too large and exploration too limited for reliable convergence in that budget.

### Q9 — Pacman Q-Learning (`PacmanQAgent`)

Uses `QLearningAgent` with Pacman-specific defaults (`epsilon=0.05, gamma=0.8, alpha=0.2`). After 2000 training games on `smallGrid`, the agent wins 100% of test games. No code changes required — the `PacmanQAgent` wrapper calls `QLearningAgent.getAction` and notifies the parent of the chosen action via `doAction`.

### Q10 — Approximate Q-Learning (`ApproximateQAgent`)

Replaces the table with a linear function over features:

```
Q(s, a) = w · f(s, a) = sum_i  w_i * f_i(s, a)
```

- **`getQValue`**: computes the dot product of the weight vector and the feature vector returned by `self.featExtractor.getFeatures(state, action)`.
- **`update`**: computes the TD error `diff = (r + gamma * V(s')) - Q(s,a)` and adjusts each weight: `w_i += alpha * diff * f_i(s,a)`. This generalizes Q-learning to unseen states via shared features.

---

## Key Concepts

- **Value Iteration** (offline planning): computes optimal policy from a known MDP
- **Q-Learning** (online learning): learns action values from experience without a model
- **Epsilon-Greedy**: exploration strategy balancing random exploration with learned policy
- **Approximate Q-Learning**: uses feature extractors to generalize across states

## Project Structure

| File | Description |
|------|-------------|
| `mdp.py` | MDP interface definition |
| `environment.py` | Environment interface |
| `learningAgents.py` | Base classes for learning agents |
| `featureExtractors.py` | Feature extractors for approximate Q-learning |
| `gridworld.py` | Gridworld MDP implementation and main driver |
| `crawler.py` | Crawler robot simulation |
| `pacman.py` | Pacman game engine |
| `autograder.py` | Project autograder |
| `test_cases/` | Test cases for each question |

## Attribution

The Pacman AI projects were developed at UC Berkeley by John DeNero and Dan Klein. Student-side autograding by Brad Miller, Nick Hay, and Pieter Abbeel. More info at http://ai.berkeley.edu.
