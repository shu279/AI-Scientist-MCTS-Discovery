"""
Seed code for discovering budget-efficient MCTS variants.

This file is a runnable benchmark suite, not a proposed method. It provides:
- a small SearchProblem interface
- three deterministic synthetic planning environments
- random, greedy, beam-search, and vanilla UCT/MCTS baselines
- separate proposed MCTS component slots that do not overwrite vanilla UCT/MCTS
- fixed simulation-budget evaluation
- exact optimality gaps for grid and graph tasks
- result saving for plots and write-up

AI Scientist should modify MCTS components such as selection, rollout, backup,
expansion, or simulation allocation. Invalid contributions only tune environment
constants, random seeds, evaluation budgets, or only the vanilla UCT constant.

Do not modify the vanilla baseline functions: uct_select_child, rollout_random,
backup_value, mcts_choose_action, or plan_with_mcts. Implement proposed methods
through proposed_select_child, proposed_rollout, proposed_backup,
proposed_mcts_choose_action, or helper functions used only by proposed_mcts.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency.
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency.
    plt = None


WORKING_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(WORKING_DIR, exist_ok=True)

State = Any
Action = Any
Plan = List[Action]

# ---------------------------------------------------------------------
# Search problem interface and benchmark environments
# ---------------------------------------------------------------------

class SearchProblem:
    """Minimal interface for finite-horizon deterministic planning problems."""

    name: str
    instance_id: str

    def initial_state(self) -> State:
        raise NotImplementedError

    def actions(self, state: State) -> List[Action]:
        raise NotImplementedError

    def step(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError

    def evaluate_state(self, state: State) -> float:
        raise NotImplementedError

    def optimal_score(self) -> Optional[float]:
        return None

    # Optional greedy action for rollouts and baselines; not used by MCTS unless implemented by the AI Scientist.
    def greedy_action(self, state: State) -> Optional[Action]:
        actions = self.actions(state)
        if not actions:
            return None
        return max(actions, key=lambda action: self.evaluate_state(self.step(state, action)))


@dataclass
class GridRewardProblem(SearchProblem):
    """Collect rewards on a small grid within a step budget."""

    rewards: Dict[Tuple[int, int], float]
    blocked: frozenset[Tuple[int, int]]
    start: Tuple[int, int]
    size: int
    max_steps: int
    instance_id: str
    name: str = "grid_reward_collection"

    def __post_init__(self) -> None:
        self.reward_positions = sorted(self.rewards)
        self.reward_to_bit = {pos: i for i, pos in enumerate(self.reward_positions)}
        self.full_mask = (1 << len(self.reward_positions)) - 1

    # State = (row, col, mask, steps, score)
    def initial_state(self) -> State:
        return (self.start[0], self.start[1], self.full_mask, 0, 0.0)

    def actions(self, state: State) -> List[Action]:
        row, col, _, steps, _ = state
        if steps >= self.max_steps:
            return []
        moves = []
        for name, dr, dc in [("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1)]: #check if each move is valid
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.blocked:
                moves.append(name)
        return moves

    def step(self, state: State, action: Action) -> State:
        row, col, mask, steps, score = state
        delta = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[action]
        nr, nc = row + delta[0], col + delta[1]
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.blocked:
            nr, nc = row, col
        new_mask = mask
        new_score = score - 0.03 # step penalty to encourage shorter plans
        bit = self.reward_to_bit.get((nr, nc))
        if bit is not None and (mask & (1 << bit)):
            new_mask &= ~(1 << bit)
            new_score += float(self.rewards[(nr, nc)])
        return (nr, nc, new_mask, steps + 1, new_score)

    def is_terminal(self, state: State) -> bool:
        _, _, mask, steps, _ = state
        return steps >= self.max_steps or mask == 0 #no more rewards or out of steps

    def evaluate_state(self, state: State) -> float:
        return float(state[4])

    def optimal_score(self) -> Optional[float]:
        @lru_cache(maxsize=None) # create memo table
        def best(row: int, col: int, mask: int, steps: int) -> float:
            if steps >= self.max_steps or mask == 0:
                return 0.0
            state = (row, col, mask, steps, 0.0)
            actions = self.actions(state)
            if not actions:
                return 0.0
            best_value = -float("inf")
            for action in actions:
                nr, nc, new_mask, _, delta_score = self.step(state, action)
                value = float(delta_score) + best(int(nr), int(nc), int(new_mask), steps + 1)
                best_value = max(best_value, value)
            return best_value

        start_state = self.initial_state()
        return best(int(start_state[0]), int(start_state[1]), int(start_state[2]), int(start_state[3]))

    def greedy_action(self, state: State) -> Optional[Action]:
        actions = self.actions(state)
        if not actions:
            return None
        row, col, mask, _, _ = state

        def score(action: Action) -> float:
            next_state = self.step(state, action)
            nr, nc = next_state[0], next_state[1]
            immediate = self.evaluate_state(next_state) - self.evaluate_state(state)
            future = 0.0
            for pos, reward in self.rewards.items():
                bit = self.reward_to_bit[pos]
                if mask & (1 << bit):
                    dist = abs(nr - pos[0]) + abs(nc - pos[1]) # manhattan distance to next reward
                    future = max(future, reward / (1.0 + dist)) # heuristic future value based on reward and distance
            return immediate + 0.15 * future # weight for future reward encourages going towards clusters of rewards rather than just the immediate next one

        return max(actions, key=score)


@dataclass
class GraphOrienteeringProblem(SearchProblem):
    """Visit valuable graph nodes under a travel-time budget."""

    travel_time: np.ndarray
    rewards: np.ndarray
    start: int
    max_time: int
    instance_id: str
    name: str = "graph_orienteering"

    def __post_init__(self) -> None:
        self.n_nodes = int(len(self.rewards))
        self.full_mask = (1 << self.n_nodes) - 1
        self.start_mask = self.full_mask & ~(1 << self.start)

    # State = (node, mask, time_used, score)
    def initial_state(self) -> State:
        return (self.start, self.start_mask, 0, 0.0)

    def actions(self, state: State) -> List[Action]:
        node, mask, time_used, _ = state
        actions = []
        for nxt in range(self.n_nodes):
            if mask & (1 << nxt):
                travel = int(self.travel_time[node, nxt])
                if time_used + travel <= self.max_time:
                    actions.append(nxt)  # within time budget and lead to unvisited nodes
        return actions

    def step(self, state: State, action: Action) -> State:
        node, mask, time_used, score = state
        travel = int(self.travel_time[node, action])
        new_time = time_used + travel
        new_mask = mask & ~(1 << int(action))
        new_score = score + float(self.rewards[action]) - 0.02 * travel
        return (int(action), new_mask, new_time, new_score)

    def is_terminal(self, state: State) -> bool:
        return len(self.actions(state)) == 0

    def evaluate_state(self, state: State) -> float:
        return float(state[3])

    def optimal_score(self) -> Optional[float]:
        @lru_cache(maxsize=None)  # create memo table
        def best(node: int, mask: int, time_used: int) -> float:
            state = (node, mask, time_used, 0.0)
            actions = self.actions(state)
            if not actions:
                return 0.0
            best_value = -float("inf")
            for action in actions:
                next_state = self.step(state, action)
                delta_score = float(next_state[3])
                value = delta_score + best(int(next_state[0]), int(next_state[1]), int(next_state[2]))
                best_value = max(best_value, value)
            return best_value

        start_state = self.initial_state()
        return best(int(start_state[0]), int(start_state[1]), int(start_state[2]))

    def greedy_action(self, state: State) -> Optional[Action]:
        actions = self.actions(state)
        if not actions:
            return None
        node, _, _, _ = state
        return max(actions, key=lambda nxt: self.rewards[nxt] / max(self.travel_time[node, nxt], 1.0))


@dataclass
class TaskSchedulingProblem(SearchProblem):
    """Choose a task order for deadline-weighted single-list scheduling."""

    durations: np.ndarray
    deadlines: np.ndarray
    priorities: np.ndarray
    machine_count: int
    instance_id: str
    precedence: Optional[Dict[int, Tuple[int, ...]]] = None
    name: str = "deadline_task_scheduling"

    def __post_init__(self) -> None:
        self.n_tasks = int(len(self.durations))
        self.full_mask = (1 << self.n_tasks) - 1
        self.precedence = self.precedence or {}
        self.predecessor_masks = {
            task: sum(1 << pred for pred in preds)
            for task, preds in self.precedence.items()
        }

    # State = (loads, mask, score)
    # loads = until when each machine has a task
    def initial_state(self) -> State:
        loads = tuple(0.0 for _ in range(self.machine_count))
        return (loads, self.full_mask, 0.0)

    def actions(self, state: State) -> List[Action]:
        _, mask, _ = state
        actions = []
        for task in range(self.n_tasks):
            if not (mask & (1 << task)):
                continue
            pred_mask = self.predecessor_masks.get(task, 0)
            if pred_mask & mask:
                continue
            actions.append(task)
        return actions

    def step(self, state: State, action: Action) -> State:
        loads, mask, score = state
        machine = int(np.argmin(loads))
        new_loads = list(loads)
        completion = new_loads[machine] + float(self.durations[action])
        new_loads[machine] = completion
        tardiness = max(0.0, completion - float(self.deadlines[action]))
        reward = 1.5 * float(self.priorities[action]) - 0.18 * float(self.priorities[action]) * tardiness
        return (tuple(new_loads), mask & ~(1 << int(action)), score + reward)

    def is_terminal(self, state: State) -> bool:
        return state[1] == 0

    def evaluate_state(self, state: State) -> float:
        return float(state[2])

    def greedy_action(self, state: State) -> Optional[Action]:
        actions = self.actions(state)
        if not actions:
            return None
        loads, _, _ = state
        earliest_machine_load = min(loads)

        # Find the most important, quickest, and most urgent task.
        def priority_density(task: int) -> float:
            slack = max(float(self.deadlines[task]) - earliest_machine_load, 1.0)
            return float(self.priorities[task]) / (float(self.durations[task]) * slack)

        return max(actions, key=priority_density)

# ---------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------

@dataclass
class MCTSNode:
    state: State
    parent: Optional["MCTSNode"]
    action: Optional[Action]
    untried_actions: List[Action]
    children: Dict[Action, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(self.visits, 1)

# ---------------------------------------------------------------------
# Vanilla MCTS baseline for comparison
# ---------------------------------------------------------------------
# AI Scientist should NOT modify these functions.

def uct_select_child(node: MCTSNode, exploration_constant: float = 1.4) -> MCTSNode:
    """Vanilla UCT child selection."""
    log_parent = math.log(max(node.visits, 1))

    def uct_score(child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        exploration = exploration_constant * math.sqrt(log_parent / child.visits)
        return child.mean_value + exploration

    return max(node.children.values(), key=uct_score)


def rollout_random(problem: SearchProblem, state: State, rng: np.random.Generator) -> float:
    """Vanilla random rollout to a terminal state."""
    current = state
    while not problem.is_terminal(current):
        actions = problem.actions(current)
        if not actions:
            break
        action = actions[int(rng.integers(0, len(actions)))]
        current = problem.step(current, action)
    return problem.evaluate_state(current)


def backup_value(node: MCTSNode, value: float) -> None:
    """Vanilla Monte Carlo backup of a terminal rollout value."""
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visits += 1
        current.value_sum += float(value)
        current = current.parent


def mcts_choose_action(
    problem: SearchProblem,
    root_state: State,
    per_decision_budget: int,
    rng: np.random.Generator,
    exploration_constant: float = 1.4,
) -> Optional[Action]:
    """Choose one action with vanilla UCT/MCTS under a fixed per-decision budget."""
    root = MCTSNode(
        state=root_state,
        parent=None,
        action=None,
        untried_actions=list(problem.actions(root_state)),
    )
    if not root.untried_actions:
        return None

    for _ in range(per_decision_budget):
        node = root
        while not problem.is_terminal(node.state) and not node.untried_actions and node.children:
            node = uct_select_child(node, exploration_constant=exploration_constant)

        if not problem.is_terminal(node.state) and node.untried_actions:
            idx = int(rng.integers(0, len(node.untried_actions)))
            action = node.untried_actions.pop(idx)
            next_state = problem.step(node.state, action)
            child = MCTSNode(
                state=next_state,
                parent=node,
                action=action,
                untried_actions=list(problem.actions(next_state)),
            )
            node.children[action] = child
            node = child

        value = rollout_random(problem, node.state, rng)
        backup_value(node, value)

    if not root.children:
        return root.untried_actions[0] if root.untried_actions else None
    return max(root.children.values(), key=lambda child: (child.visits, child.mean_value)).action


def plan_with_mcts(problem: SearchProblem, per_decision_budget: int, rng: np.random.Generator) -> Tuple[Plan, float]:
    """Receding-horizon vanilla MCTS planner."""
    state = problem.initial_state()
    plan: Plan = []
    while not problem.is_terminal(state):
        action = mcts_choose_action(problem, state, per_decision_budget=per_decision_budget, rng=rng)
        if action is None:
            break
        plan.append(action)
        state = problem.step(state, action)
    return plan, problem.evaluate_state(state)

# ---------------------------------------------------------------------
# Proposed MCTS extension points
# ---------------------------------------------------------------------
# AI Scientist should implement new MCTS variants here.

def proposed_select_child(node: MCTSNode, context: Dict[str, Any]) -> MCTSNode:
    """AI Scientist should replace this with a named selection mechanism.

    This fallback preserves vanilla UCT. A valid contribution should introduce a
    clear rule or formula, not only tune the UCT exploration constant.
    Do not modify uct_select_child; keep it as the unchanged baseline.
    """
    return uct_select_child(
        node,
        exploration_constant=float(context.get("exploration_constant", 1.4)),
    )


def proposed_rollout(
    problem: SearchProblem,
    state: State,
    rng: np.random.Generator,
    context: Dict[str, Any],
) -> float:
    """AI Scientist should replace this with a named rollout mechanism.

    Do not modify rollout_random; keep it as the unchanged baseline.
    """
    _ = context
    return rollout_random(problem, state, rng)


def proposed_backup(node: MCTSNode, value: float, context: Dict[str, Any]) -> None:
    """AI Scientist should replace this with a named backup mechanism.

    Do not modify backup_value; keep it as the unchanged baseline.
    """
    _ = context
    backup_value(node, value)


def proposed_mcts_choose_action(
    problem: SearchProblem,
    root_state: State,
    per_decision_budget: int,
    rng: np.random.Generator,
) -> Optional[Action]:
    """MCTS action selection using proposed components.

    AI Scientist may modify selection, expansion, rollout, backup, or simulation
    allocation here while leaving vanilla UCT/MCTS unchanged for comparison.
    Strong methods should have a name, a formula or pseudocode, and ablations.
    """
    root = MCTSNode(
        state=root_state,
        parent=None,
        action=None,
        untried_actions=list(problem.actions(root_state)),
    )
    if not root.untried_actions:
        return None

    context: Dict[str, Any] = {
        "problem_name": problem.name,
        "per_decision_budget": per_decision_budget,
        "exploration_constant": 1.4,
    }

    for _ in range(per_decision_budget):
        node = root
        while not problem.is_terminal(node.state) and not node.untried_actions and node.children:
            node = proposed_select_child(node, context)

        if not problem.is_terminal(node.state) and node.untried_actions:
            idx = int(rng.integers(0, len(node.untried_actions)))
            action = node.untried_actions.pop(idx)
            next_state = problem.step(node.state, action)
            child = MCTSNode(
                state=next_state,
                parent=node,
                action=action,
                untried_actions=list(problem.actions(next_state)),
            )
            node.children[action] = child
            node = child

        value = proposed_rollout(problem, node.state, rng, context)
        proposed_backup(node, value, context)

    if not root.children:
        return root.untried_actions[0] if root.untried_actions else None
    return max(root.children.values(), key=lambda child: (child.visits, child.mean_value)).action

def plan_with_proposed_mcts(
    problem: SearchProblem,
    per_decision_budget: int,
    rng: np.random.Generator,
) -> Tuple[Plan, float]:
    """Receding-horizon planner using proposed MCTS components."""
    state = problem.initial_state()
    plan: Plan = []
    while not problem.is_terminal(state):
        action = proposed_mcts_choose_action(
            problem,
            state,
            per_decision_budget=per_decision_budget,
            rng=rng,
        )
        if action is None:
            break
        plan.append(action)
        state = problem.step(state, action)
    return plan, problem.evaluate_state(state)

# ---------------------------------------------------------------------
# Non-MCTS baseline planners
# ---------------------------------------------------------------------

def plan_random(problem: SearchProblem, rng: np.random.Generator) -> Tuple[Plan, float]:
    state = problem.initial_state()
    plan: Plan = []
    while not problem.is_terminal(state):
        actions = problem.actions(state)
        if not actions:
            break
        action = actions[int(rng.integers(0, len(actions)))]
        plan.append(action)
        state = problem.step(state, action)
    return plan, problem.evaluate_state(state)


def plan_greedy(problem: SearchProblem) -> Tuple[Plan, float]:
    state = problem.initial_state()
    plan: Plan = []
    while not problem.is_terminal(state):
        action = problem.greedy_action(state)
        if action is None:
            break
        plan.append(action)
        state = problem.step(state, action)
    return plan, problem.evaluate_state(state)


def plan_beam_search(problem: SearchProblem, width: int = 4) -> Tuple[Plan, float]:
    """Simple fixed-width beam-search baseline."""
    beams: List[Tuple[State, Plan]] = [(problem.initial_state(), [])]
    finished: List[Tuple[State, Plan]] = []
    while beams:
        candidates: List[Tuple[State, Plan]] = []
        for state, plan in beams:
            if problem.is_terminal(state):
                finished.append((state, plan))
                continue
            for action in problem.actions(state):
                next_state = problem.step(state, action)
                candidates.append((next_state, plan + [action]))
        if not candidates:
            break
        candidates.sort(key=lambda item: problem.evaluate_state(item[0]), reverse=True)
        beams = candidates[:width]
        if all(problem.is_terminal(state) for state, _ in beams):
            finished.extend(beams)
            break
    all_plans = finished + beams
    best_state, best_plan = max(all_plans, key=lambda item: problem.evaluate_state(item[0]))
    return best_plan, problem.evaluate_state(best_state)

# ---------------------------------------------------------------------
# Synthetic problem generation
# ---------------------------------------------------------------------

def generate_problem_suite(seed: int, instances_per_env: int = 6) -> List[SearchProblem]:
    rng = np.random.default_rng(seed)
    problems: List[SearchProblem] = []
    for idx in range(instances_per_env):
        grid_seed = int(rng.integers(0, 1_000_000))
        problems.append(make_grid_problem(grid_seed, f"grid_{idx}"))
        graph_seed = int(rng.integers(0, 1_000_000))
        problems.append(make_graph_problem(graph_seed, f"graph_{idx}"))
        sched_seed = int(rng.integers(0, 1_000_000))
        problems.append(make_scheduling_problem(sched_seed, f"scheduling_{idx}"))
    return problems


def make_grid_problem(seed: int, instance_id: str) -> GridRewardProblem:
    rng = np.random.default_rng(seed)
    size = 7
    start = (0, 0)
    blocked = set()
    while len(blocked) < 7:
        cell = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if cell != start:
            blocked.add(cell)
    rewards: Dict[Tuple[int, int], float] = {}
    while len(rewards) < 8:
        cell = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if cell != start and cell not in blocked:
            rewards[cell] = float(rng.integers(2, 9))
    return GridRewardProblem(
        rewards=rewards,
        blocked=frozenset(blocked),
        start=start,
        size=size,
        max_steps=18,
        instance_id=instance_id,
    )


def make_graph_problem(seed: int, instance_id: str) -> GraphOrienteeringProblem:
    rng = np.random.default_rng(seed)
    n_nodes = 12
    coords = rng.uniform(0, 1, size=(n_nodes, 2))
    travel = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                travel[i, j] = 0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                travel[i, j] = int(max(1, round(2 + 12 * dist)))
    rewards = rng.integers(2, 12, size=n_nodes).astype(float)
    rewards[0] = 0.0
    return GraphOrienteeringProblem(
        travel_time=travel,
        rewards=rewards,
        start=0,
        max_time=28,
        instance_id=instance_id,
    )


def make_scheduling_problem(seed: int, instance_id: str) -> TaskSchedulingProblem:
    rng = np.random.default_rng(seed)
    n_tasks = 22
    durations = rng.integers(2, 10, size=n_tasks).astype(float)
    priorities = rng.choice(
        [1.0, 1.5, 2.0, 3.0, 4.0],
        size=n_tasks,
        p=[0.25, 0.25, 0.22, 0.18, 0.10],
    )

    # Harder low-budget scheduling instance: several long, high-priority jobs
    # must be started early, while short urgent jobs create tempting local choices.
    long_urgent_jobs = np.array([0, 1, 2, 3])
    durations[long_urgent_jobs] = rng.integers(9, 13, size=len(long_urgent_jobs))
    priorities[long_urgent_jobs] = np.array([5.0, 4.5, 4.5, 4.0])

    short_decoys = np.array([4, 5, 6, 7])
    durations[short_decoys] = rng.integers(1, 4, size=len(short_decoys))
    priorities[short_decoys] = rng.choice([2.0, 2.5, 3.0], size=len(short_decoys))

    deadlines = rng.uniform(8, 28, size=n_tasks) + 0.45 * durations
    deadlines[long_urgent_jobs] = durations[long_urgent_jobs] + rng.uniform(
        1.0, 4.0, size=len(long_urgent_jobs)
    )
    deadlines[short_decoys] = durations[short_decoys] + rng.uniform(
        2.0, 6.0, size=len(short_decoys)
    )

    precedence = {
        8: (0,),
        9: (0,),
        10: (1,),
        11: (2,),
        12: (4,),
        13: (5,),
    }
    for task, preds in precedence.items():
        pred_finish = max(durations[pred] for pred in preds)
        deadlines[task] = pred_finish + durations[task] + rng.uniform(2.0, 6.0)

    return TaskSchedulingProblem(
        durations=durations,
        deadlines=deadlines,
        priorities=priorities,
        machine_count=3,
        instance_id=instance_id,
        precedence=precedence,
    )

# ---------------------------------------------------------------------
# Experiment execution and metrics
# ---------------------------------------------------------------------

def parse_budgets() -> List[int]:
    raw = os.environ.get("MCTS_BUDGETS", "16,64,128")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_experiment(seed: int = 0, instances_per_env: int = 6) -> Dict[str, object]:
    budgets = parse_budgets()
    problems = generate_problem_suite(seed=seed, instances_per_env=instances_per_env)
    rows: List[Dict[str, object]] = []

    for problem_idx, problem in enumerate(problems):
        optimal_score = problem.optimal_score()
        random_plan, random_score = plan_random(problem, np.random.default_rng(seed + 10_000 + problem_idx))
        greedy_plan, greedy_score = plan_greedy(problem)
        beam_plan, beam_score = plan_beam_search(problem, width=4)
        rows.extend(
            [
                {
                    "environment": problem.name,
                    "instance_id": problem.instance_id,
                    "planner": "random",
                    "per_decision_budget": 0,
                    "score": random_score,
                    "optimal_score": optimal_score,
                    "optimality_gap": None if optimal_score is None else float(optimal_score - random_score),
                    "plan_length": len(random_plan),
                },
                {
                    "environment": problem.name,
                    "instance_id": problem.instance_id,
                    "planner": "greedy",
                    "per_decision_budget": 0,
                    "score": greedy_score,
                    "optimal_score": optimal_score,
                    "optimality_gap": None if optimal_score is None else float(optimal_score - greedy_score),
                    "plan_length": len(greedy_plan),
                },
                {
                    "environment": problem.name,
                    "instance_id": problem.instance_id,
                    "planner": "beam_search_width_4",
                    "per_decision_budget": 0,
                    "score": beam_score,
                    "optimal_score": optimal_score,
                    "optimality_gap": None if optimal_score is None else float(optimal_score - beam_score),
                    "plan_length": len(beam_plan),
                },
            ]
        )

        for per_decision_budget in budgets:
            mcts_seed = seed + 20_000 + problem_idx * 100 + per_decision_budget
            mcts_plan, mcts_score = plan_with_mcts(
                problem,
                per_decision_budget=per_decision_budget,
                rng=np.random.default_rng(mcts_seed),
            )
            rows.append(
                {
                    "environment": problem.name,
                    "instance_id": problem.instance_id,
                    "planner": "vanilla_uct_mcts",
                    "per_decision_budget": per_decision_budget,
                    "score": mcts_score,
                    "optimal_score": optimal_score,
                    "optimality_gap": None if optimal_score is None else float(optimal_score - mcts_score),
                    "plan_length": len(mcts_plan),
                }
            )
            proposed_plan, proposed_score = plan_with_proposed_mcts(
                problem,
                per_decision_budget=per_decision_budget,
                rng=np.random.default_rng(mcts_seed),
            )
            rows.append(
                {
                    "environment": problem.name,
                    "instance_id": problem.instance_id,
                    "planner": "proposed_mcts",
                    "per_decision_budget": per_decision_budget,
                    "score": proposed_score,
                    "optimal_score": optimal_score,
                    "optimality_gap": None if optimal_score is None else float(optimal_score - proposed_score),
                    "plan_length": len(proposed_plan),
                }
            )

    rows = add_regret_to_best_observed(rows)
    summaries = summarize_results(rows)
    return {
        "budgets": budgets,
        "scenario_rows": rows,
        "policy_summaries": summaries,
    }


def add_regret_to_best_observed(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_instance: Dict[Tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["environment"]), str(row["instance_id"]))
        best_by_instance[key] = max(best_by_instance.get(key, -float("inf")), float(row["score"]))
    for row in rows:
        key = (str(row["environment"]), str(row["instance_id"]))
        row["regret_to_best_observed"] = best_by_instance[key] - float(row["score"])
    return rows


def summarize_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["environment"]), str(row["planner"]), int(row["per_decision_budget"]))
        groups.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for (environment, planner, per_decision_budget), group in sorted(groups.items()):
        scores = np.array([float(r["score"]) for r in group], dtype=float)
        regrets = np.array([float(r["regret_to_best_observed"]) for r in group], dtype=float)
        gaps = [r["optimality_gap"] for r in group if r.get("optimality_gap") is not None]
        summaries.append(
            {
                "environment": environment,
                "planner": planner,
                "per_decision_budget": per_decision_budget,
                "mean_score": float(np.mean(scores)),
                "worst_case_score": float(np.min(scores)),
                "mean_regret_to_best_observed": float(np.mean(regrets)),
                "p90_regret_to_best_observed": float(np.quantile(regrets, 0.9)),
                "mean_optimality_gap": None if not gaps else float(np.mean(gaps)),
                "p90_optimality_gap": None if not gaps else float(np.quantile(np.array(gaps, dtype=float), 0.9)),
                "num_instances": len(group),
            }
        )
    return summaries

# ---------------------------------------------------------------------
# Result serialization and entry point
# ---------------------------------------------------------------------

def save_results(result: Dict[str, object]) -> None:
    np.save(os.path.join(WORKING_DIR, "experiment_data.npy"), result, allow_pickle=True)
    with open(os.path.join(WORKING_DIR, "summary.json"), "w") as f:
        json.dump(result["policy_summaries"], f, indent=2)

    def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(os.path.join(WORKING_DIR, "policy_metrics.csv"), result["policy_summaries"])
    write_csv(os.path.join(WORKING_DIR, "scenario_metrics.csv"), result["scenario_rows"])

    if pd is not None:
        policy_df = pd.DataFrame(result["policy_summaries"])
        scenario_df = pd.DataFrame(result["scenario_rows"])
        policy_df.to_csv(os.path.join(WORKING_DIR, "policy_metrics.csv"), index=False)
        scenario_df.to_csv(os.path.join(WORKING_DIR, "scenario_metrics.csv"), index=False)

    if pd is not None and plt is not None:
        policy_df = pd.DataFrame(result["policy_summaries"])
        for environment, env_df in policy_df.groupby("environment"):
            fig, ax = plt.subplots(figsize=(9, 4))
            labels = [f"{r.planner}\\nB={r.per_decision_budget}" for r in env_df.itertuples()]
            ax.bar(labels, env_df["mean_score"])
            ax.set_title(f"Mean score: {environment}")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(WORKING_DIR, f"{environment}_mean_score.png"), dpi=200)
            plt.close(fig)


def main() -> None:
    seed = int(os.environ.get("SEED", "0"))
    instances_per_env = int(os.environ.get("INSTANCES_PER_ENV", "6"))
    result = run_experiment(seed=seed, instances_per_env=instances_per_env)
    save_results(result)
    print("Saved MCTS discovery benchmark outputs to:", WORKING_DIR)
    for summary in result["policy_summaries"]:
        print(summary)


if __name__ == "__main__":
    main()
