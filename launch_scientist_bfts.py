import os.path as osp
import json
import argparse
import shutil
import os
import re
import sys
from datetime import datetime
from ai_scientist.llm import create_client

from contextlib import contextmanager
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.utils.token_tracker import token_tracker


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ai_scientist/ideas/mcts_discovery.json",
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=8,
        help="Maximum number of pages for the generated writeup",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="gpt-5.4",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="gpt-5.5",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-5.4",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-5.4",
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-5.4",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    return parser.parse_args()


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    pdf_path = osp.join(idea_dir, pdf_files[0])
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    return pdf_path


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    idea = ideas[args.idea_idx]

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # Convert idea json to markdown file
    idea_path_md = osp.join(idea_dir, "idea.md")

    # If load_code is True, get the Python file with same name as JSON
    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    added_code = code

    if added_code is not None:
        print(f"Loaded additional code context ({len(added_code)} characters).")

    # Add code to idea json if it was loaded
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # Store raw idea json
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = "bfts_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )

    perform_experiments_bfts(idea_config_path)
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    copied_results_dir = osp.join(idea_dir, "experiment_results")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            copied_results_dir,
            dirs_exist_ok=True,
        )

    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    if os.path.exists(copied_results_dir):
        shutil.rmtree(copied_results_dir)

    save_token_tracker(idea_dir)

    if not args.skip_writeup:
        writeup_success = False
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            writeup_success = perform_writeup(
                base_folder=idea_dir,
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=args.page_limit,
                citations_text=citations_text,
            )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(idea_dir)

    if not args.skip_review and not args.skip_writeup:
        # Perform paper review if the paper exists
        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")

    print("Start cleaning up child processes")
    try:
        import psutil
        import signal
    except ImportError:
        print("psutil is not installed; skipping child-process cleanup.")
        sys.exit(0)

    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, alive = psutil.wait_procs(children, timeout=3)

    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    sys.exit(0)
