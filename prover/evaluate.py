"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""
import sys
sys.path.append('/home/vincentzhu/ReProver')
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache

from common import set_logger

from prover.proof_search_tree import Status, DistributedProver


def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    # data = json.load(open(os.path.join(data_path, f"time_filtered_v3_wfs.json")))
    # print(len(data))
    theorems = []
    positions = []

    if data_path.startswith("/home/vincentzhu/gfn_ntp/data/"):
        data_list = []
        for k, v in data.items():
            data_list.append(v)
        data = data_list

    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))

    # Jointly sort theorems and positions
    assert len(theorems) > 0
    theorems_and_positions = list(zip(theorems, positions))
    theorems_and_positions.sort(
        key=lambda x: hashlib.md5(
            f"{x[0].file_path}:{x[0].full_name}".encode()
        ).hexdigest()
    )
    theorems, positions = zip(*theorems_and_positions)
    theorems, positions = list(theorems), list(positions)

    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata = json.load(open(os.path.join(data_path, "metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    use_vllm: bool = False,
    gen_ckpt_path: Optional[str] = None,
    ret_ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    max_inp_seq_len: int = 2048,
    max_oup_seq_len: int = 512,
    length_penalty: float = 0.0,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    max_expansions: Optional[int] = None,
    num_workers: int = 1,
    num_gpus: int = 0,
    save_results: bool = False,
    verbose: bool = False,
) -> float:
    set_logger(verbose)

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        use_vllm,
        gen_ckpt_path,
        ret_ckpt_path,
        indexed_corpus_path,
        max_inp_seq_len,
        max_oup_seq_len,
        length_penalty,
        tactic,
        module,
        num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        max_expansions=max_expansions,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
    )
    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    if save_results:
        pickle_path = f"{exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        # choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--gen_ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--ret_ckpt_path",
        type=str,
        help="Checkpoint of the premise retriever.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--max-inp-seq-len", type=int, default=2048)
    parser.add_argument("--max-oup-seq-len", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=None,
        help="Maximum number of expansions during proof search.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="The number of GPUs for proof search."
    )
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    assert args.gen_ckpt_path or args.tactic
    assert args.num_gpus <= args.num_workers

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.use_vllm,
        args.gen_ckpt_path,
        args.ret_ckpt_path,
        args.indexed_corpus_path,
        args.max_inp_seq_len,
        args.max_oup_seq_len,
        args.length_penalty,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.max_expansions,
        args.num_workers,
        args.num_gpus,
        args.save_results,
        args.verbose,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
