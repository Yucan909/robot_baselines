import concurrent.futures as cf
import functools
import logging
import multiprocessing
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import tqdm

__worker_num = None
__queue: Optional[multiprocessing.Queue] = None

# TODO: fix typing, b.c. "ArrayLikeInt" doesn't work.
NPSeed = Union[
    None,
    int,
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


class Seedable(Protocol):
    def __call__(self, *, seed: NPSeed) -> Any:
        """A function which can accept a seed as input.

        Args:
            seed (NPSeed): A seed which can be used to seed a numpy RNG.

        Returns:
            Any: Can return anything.
        """


def _run_fn(
    fn: Seedable,
    args: Tuple[Dict, np.random.SeedSequence],
    pre_fn: Optional[Seedable] = None,
    post_fn: Optional[Callable] = None,
    debug: bool = False,
) -> Tuple[Optional[Any], bool]:
    """The function runner"""
    kwargs, seed = args
    try:
        # Run a function before, potentially to set some global seed.
        if pre_fn:
            pre_fn(seed=seed)

        # Pass the seed.
        if "seed" not in kwargs:
            kwargs["seed"] = seed

        # Run the function.
        result = fn(**kwargs)
        completed = True
    except Exception as e:
        logging.error(f"encountered an error: {type(e).__name__}:{e}")
        completed = False
        result = None
        if debug:
            raise e

    global __worker_num, __queue
    if __queue:
        __queue.put(__worker_num)

    # Run some cleanup function.
    if post_fn:
        post_fn()

    return result, completed


def _init_proc(
    q, proc_start, n_workers, n_proc_per_worker, init_fn=None, init_args=None
):
    """Process initializer, handles setting processor affinity on Linux."""
    worker_num = q.get(timeout=5)
    if sys.platform == "linux":
        s = proc_start
        n = worker_num
        N = n_workers * n_proc_per_worker
        procs = [s + (n * n_proc_per_worker + i) % N for i in range(n_proc_per_worker)]
        try:
            os.sched_setaffinity(os.getpid(), procs)
        except OSError as e:
            logging.error(f"failed to set affinity: {e}")

    global __worker_num, __queue
    __worker_num = worker_num
    __queue = q

    # Initialize.
    if init_fn is not None:
        init_fn(*init_args)


def distributed_eval(
    fn: Seedable,
    kwargs_list: List[Dict],
    init_fn: Optional[Callable] = None,
    init_args: Optional[Tuple] = None,
    pre_fn: Optional[Seedable] = None,
    post_fn: Optional[Callable] = None,
    n_workers: int = 30,
    n_proc_per_worker: int = 2,
    proc_start: int = 0,
    seed: Optional[int] = None,
    logfile: Optional[str] = None,
) -> Tuple[List[Any], List[bool]]:
    """Runs a distributed eval of a function across multiple processes. Attempts to do so
    deterministically, with a seed which gets passed around.

    NOTE: kwargs_list needs to be in the same order for full reproducibility, in the event of
    random seeds.

    Args:
        fn (Seedable): The function to run.
        kwargs_list (List[Dict]): A list of dictionaries of the arguments to pass to the function.
        pre_fn (Optional[Seedable], optional): A function to run before the function (i.e. to set global seeds). Defaults to None.
        post_fn (Optional[Callable], optional): A cleanup function afterwards, i.e. to explicitlygarbage collect or something. Defaults to None.
        n_workers (int, optional): Number of workers to distribute. Defaults to 30.
        n_proc_per_worker (int, optional): Number of cores to allocate per worker. Defaults to 2.
        proc_start (int, optional): Which processor to start at (i.e. if you want to avoid certain cores...). Defaults to 0.
        seed (int, optional): The initial random seed, which will be branched.
        logfile (str, optional): A file to dump out the parameters / seed combination which was called.

    Returns:
        Tuple[List[Any], List[bool]]:
            results: A list of results, whatever the function returns.
            completeds: A list of booleans, indicating whether the function returned (True) or threw an exception or crashed.
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(kwargs_list))

    results = []
    completeds = []

    with tqdm.tqdm(total=len(kwargs_list)) as pbar:
        # Case where we aren't doing multiprocessing.
        if n_workers == 0:
            if init_fn is not None:
                init_fn(*init_args) if init_args else init_fn()

            for kwargs, child_seed in zip(kwargs_list, child_seeds):
                result, completed = _run_fn(fn, (kwargs, child_seed), debug=True)
                results.append(result)
                completeds.append(completed)
                pbar.update(1)
        else:
            # Gotta spawn.
            mp_context = multiprocessing.get_context("spawn")

            # Construct a multiprocessing queue to pass around, containing the worker ids.
            queue: multiprocessing.Queue = mp_context.Queue()
            for i in range(n_workers):
                queue.put(i)

            with cf.ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_proc,
                initargs=(
                    queue,
                    proc_start,
                    n_workers,
                    n_proc_per_worker,
                    init_fn,
                    init_args,
                ),
                mp_context=mp_context,
            ) as executor:
                joint_args = list(zip(kwargs_list, child_seeds))

                if logfile is not None:
                    with open(logfile, "w") as f:
                        js = [f"%s: {cs}" % kwa for kwa, cs in joint_args]
                        s = "\n".join(js)
                        f.write(s)

                futures = {
                    executor.submit(
                        functools.partial(_run_fn, fn, pre_fn=pre_fn, post_fn=post_fn),
                        (kwargs, child_seed),
                    ): i
                    for i, (kwargs, child_seed) in enumerate(joint_args)
                }
                result_dict = {}
                completed_dict = {}
                for future in cf.as_completed(futures):
                    result, completed = future.result()
                    result_dict[futures[future]] = result
                    completed_dict[futures[future]] = completed
                    pbar.update(1)

                results = [result_dict[i] for i in range(len(kwargs_list))]
                completeds = [completed_dict[i] for i in range(len(kwargs_list))]
    return results, completeds
