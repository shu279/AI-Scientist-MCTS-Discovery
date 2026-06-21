# AI Scientist for MCTS: Discovering Budget-Efficient Planning Variants

This repository adapts the AI Scientist-v2 for
algorithm discovery in Monte Carlo Tree Search (MCTS).

The goal is to test whether an automated research agent can discover
budget-efficient MCTS variants for small synthetic planning tasks by modifying
selection, rollout, backup, expansion, or action choice. The benchmark is
intentionally lightweight and synthetic for fast algorithmic iteration.

## What This Fork Changes

This fork repurposes the original AI Scientist-v2 workflow from
deep-learning-centric experimentation to MCTS algorithm discovery.

- Added MCTS discovery topic prompt: `ai_scientist/ideas/mcts_discovery.md`
- Added runnable planning benchmark: `ai_scientist/ideas/mcts_discovery.py`
- Added launchable seed idea: `ai_scientist/ideas/mcts_discovery.json`
- Adapted tree-search process for MCTS:
  `ai_scientist/treesearch/agent_manager.py`
- Adapted parallel agent prompts to preserve the vanilla UCT/MCTS baseline:
  `ai_scientist/treesearch/parallel_agent.py`
- Adapted reviewer criteria for baseline integrity, named mechanism validity,
  and fair fixed-budget comparison: `ai_scientist/perform_llm_review.py`
- Removed dataset reference injection: `launch_scientist_bfts.py`
- Restricted the LLM/VLM model configuration to recent OpenAI-style model
  aliases (`gpt-5.4` / `gpt-5.5`)

## Benchmark

The seed code provides three deterministic synthetic planning environments:

- Grid reward collection
- Graph orienteering
- Deadline-based task scheduling

It includes these baseline planners:

- Random planning
- Greedy planning
- Beam search
- Vanilla UCT/MCTS
- A separate `proposed_mcts` slot for AI Scientist-generated variants

The benchmark reports planning metrics such as:

- Mean score
- Worst-case score
- Regret to the best observed planner on each instance
- Optimality gap where exact optimal scores are available
- Budget-sensitive comparisons across fixed per-decision MCTS budgets

Generated benchmark outputs include:

- `working/experiment_data.npy`
- `working/summary.json`
- `working/policy_metrics.csv`
- `working/scenario_metrics.csv`
- Optional mean-score plots when pandas and matplotlib are available

## Contributions

AI Scientist should propose a named MCTS mechanism with a clear rule, formula,
or pseudocode.

Allowed modification targets:

- `proposed_select_child`
- `proposed_rollout`
- `proposed_backup`
- `proposed_mcts_choose_action`
- Helper functions used only by `proposed_mcts`

Invalid contributions:

- Modifying the vanilla UCT/MCTS baseline functions
- Modifying random, greedy, or beam-search baselines
- Changing environment constants, random seeds, metrics, scenario counts, or
  fixed per-decision budgets to make results look better
- Only tuning the UCT exploration constant
- Reporting a method without a named mechanism or clear rule

## Installation

Use a virtual environment or VM. The benchmark itself is CPU-friendly and
numpy-based, but the upstream AI Scientist pipeline has broader dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your environment does not include `psutil`, the launch script skips final
child-process cleanup instead of failing.

## API Keys

Set an OpenAI API key before running the AI Scientist pipeline:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

Semantic Scholar is optional and mainly useful during ideation or citation
gathering:

```bash
export S2_API_KEY="YOUR_SEMANTIC_SCHOLAR_KEY"
```

Model names are configured in `bfts_config.yaml` and launch arguments. If a
model in the config is not available to your account, edit the config before
running.

## Run AI Scientist

To verify the synthetic benchmark without calling an LLM, run:

```bash
python ai_scientist/ideas/mcts_discovery.py
```

Expected outputs are written to `working/`.

To generate new high-level research ideas first, run ideation against the MCTS
topic file:

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/mcts_discovery.md \
  --max-num-generations 1 \
  --num-reflections 5
```

This calls an LLM and writes ideas to `ai_scientist/ideas/mcts_discovery.json`.
If that JSON already exists, the script loads the existing ideas and appends new
ones.

Then run one idea through the AI Scientist pipeline. `--idea_idx` selects the
zero-based entry in `ai_scientist/ideas/mcts_discovery.json`; `--load_code`
loads the matching seed code file,
`ai_scientist/ideas/mcts_discovery.py`.

```bash
python launch_scientist_bfts.py \
  --idea_idx 0 \
  --load_code
```

Results are saved under a timestamped directory in `experiments/`.

## Safety

This project executes LLM-generated Python code. Run it in a controlled VM,
container, or other sandboxed environment.

## Current Scope and Limitations

This is a research prototype for automated algorithm discovery. It is not a
production planner and it is not intended for operational decision making.

- The environments are synthetic and small.
- The claims should be about fixed-budget MCTS behavior on these
  benchmark families, not real-world planning.
- The internal name `HyperparamTuningIdea` is retained for compatibility with
  the original AI Scientist tree-search plumbing, but semantically it represents
  a named MCTS variant candidate in this fork.

## Relationship to AI Scientist-v2

This repository builds on AI Scientist-v2 and keeps much of the original
pipeline structure.

If you use the original AI Scientist-v2 framework, cite:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## License

This repository retains the upstream license terms in `LICENSE`. Review the
license before using or redistributing generated outputs.
