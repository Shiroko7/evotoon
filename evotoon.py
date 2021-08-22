import random
import subprocess
from typing import Callable, List, Tuple

import numpy as np

from data_classes import CatParam, FloatParam, IntParam


def make_seed(seed: int = 765):
    """
    initialize seed
    """
    random.seed(seed)
    np.random.seed(seed)
    return seed


def latin_hypercube_sampling_num(min_val, max_val, size, dtype) -> List:
    """
    creates a latin hypercube sampling for numeric values.
    """
    partition = np.linspace(
        start=min_val, stop=max_val, num=size + 1, dtype=dtype
    )
    low = partition[:size]
    high = partition[1 : size + 1]

    if dtype == np.float64:
        points = np.random.uniform(low=low, high=high, size=size)
    elif dtype == np.int32:
        points = np.random.randint(low=low, high=high, size=size)

    np.random.shuffle(points)

    return points


def latin_hypercube_sampling_cat(options: List[str], size: int) -> List:
    """
    creates a latin hypercube sampling for categorical values.
    """
    options_length = len(options)

    if options_length < size:
        # it is not possible to use this method so a random selection is made
        points = [random.choice(options) for i in range(size)]
    else:
        # otherwhise the method is made by making a LHS of the indexes of the categorical options list
        index_sampling = latin_hypercube_sampling_num(
            0, options_length - 1, size, np.int32
        )
        points = [options[i] for i in index_sampling]

    return points


def initialization(
    poblation_size: int,
    float_parameters_list: List[FloatParam] = list(),
    int_parameters_list: List[IntParam] = list(),
    cat_parameters_list: List[CatParam] = list(),
) -> List[dict]:
    """
    creates the initial solutions for the tunning process.
    """

    # get every value
    parameter_list = list()
    for param in float_parameters_list:
        points = latin_hypercube_sampling_num(
            min_val=param.min_val,
            max_val=param.max_val,
            size=poblation_size,
            dtype=np.float64,
        )

        parameter_list.append((param.name, points))

    for param in int_parameters_list:
        points = latin_hypercube_sampling_num(
            min_val=param.min_val,
            max_val=param.max_val,
            size=poblation_size,
            dtype=np.int32,
        )

        parameter_list.append((param.name, points))

    for param in cat_parameters_list:
        points = latin_hypercube_sampling_cat(param.values, poblation_size)
        parameter_list.append((param.name, points))

    # reshape
    initial_poblation = list()
    for i in range(poblation_size):
        conf = dict()
        for name, points in parameter_list:
            conf[name] = points[i]
        initial_poblation.append(conf)

    return initial_poblation


def execute_ILSMKP(
    instance: str,
    seed: int,
    executable_path: str,
    evaluations: int,
    k: int,
) -> float:
    """
    function to execute ILSMKP program, it returns its output.
    """
    cmd = [executable_path, instance, str(seed), str(evaluations), str(k)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))
    return output


def evaluate_results(result_list: List):
    """
    evaluate results, a simple mean for the time being
    """
    return sum(result_list) / len(result_list)


def configuration_evaluation(
    algorithm: Callable, instance_list: List, seed_list, **kwargs
) -> float:
    """
    interface to call different algorithms to evaluate
    """
    result_list = [
        algorithm(instance, seed, **kwargs)
        for instance, seed in zip(instance_list, seed_list)
    ]
    return evaluate_results(result_list)


def naive_tunning(
    algorithm: Callable, configurations: List[dict], **kwargs
) -> Tuple[List, float]:
    """
    simple method to choose the best configuration given a list of configurations
    """
    best_conf = configurations[0]
    best_perf = -1.0
    for conf in configurations:
        perf = configuration_evaluation(algorithm, **{**conf, **kwargs})
        print(conf, perf)
        if perf > best_perf:
            best_perf = perf
            best_conf = conf

    return best_conf, best_perf
