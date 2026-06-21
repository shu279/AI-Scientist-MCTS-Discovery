# Title: Budget-Efficient MCTS Variants for Synthetic Planning Tasks

## Keywords
Monte Carlo Tree Search, UCT, tree search, budgeted planning, rollout policies, backup rules, selection rules, simulation allocation, synthetic planning tasks

## TL;DR
How can Monte Carlo Tree Search be modified to make better planning decisions when only a small fixed per-decision simulation budget is available?

## Abstract
Monte Carlo Tree Search is a flexible planning algorithm, but vanilla UCT can perform poorly when the simulation budget is small. In low-budget settings, it may waste simulations on uninformative branches, rely on weak random rollouts, or back up noisy value estimates too aggressively. This topic invites research ideas on budget-efficient MCTS variants for small synthetic planning tasks. The seed code provides a benchmark suite with three deterministic environments: grid reward collection, graph orienteering, and deadline-based task scheduling. Baselines include random planning, greedy planning, beam search, and unchanged vanilla UCT/MCTS, all evaluated under fixed per-decision simulation budgets. Proposed methods should introduce a named MCTS variant that modifies at least one core component of the algorithm, such as selection, expansion, rollout, backup, or simulation allocation. The method should include a clear rule, formula, or pseudocode, and should be evaluated across the provided environments using metrics such as average return, regret to the best observed solution, optimality gap where available, and performance across multiple fixed budgets. The goal is not to tune benchmark settings or only adjust the UCT exploration constant. Strong ideas should preserve the vanilla UCT/MCTS baseline, compare fairly under the same budgets, and explain why the proposed mechanism improves low-budget planning.
