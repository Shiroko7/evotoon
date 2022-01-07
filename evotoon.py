import random
import subprocess
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing

from data_classes import CatParam, FloatParam, IntParam
from scipy.stats import friedmanchisquare

import scikit_posthocs as sp
import os


def trunc_array(array: list, decs=0):
    """
    Function to truncate a numpy array
    """
    return np.trunc(array * 10 ** decs) / (10 ** decs)


def trun_float(number, decs=0) -> float:
    step = 10.0 ** decs
    return math.trunc(step * number) / step


def make_seed(seed: int = 765):
    """
    initialize seed
    """
    random.seed(seed)
    np.random.seed(seed)

    return seed


def choose_instances(path: str, n: int, seed: int):
    instance_list = []
    for file in sorted(os.listdir(path)):
        if file != ".DS_Store":
            instances = sorted(os.listdir(os.path.join(path, file)))
            instances = [os.path.join(path, file, ins) for ins in instances]
            instance_list += random.sample(instances, n)
    return instance_list


def select_seeds(instance_list: list, n_seeds: int, base_seed: int):
    new_instance_list = []
    seed_list = []
    for instance in instance_list:
        for i in range(n_seeds):
            new_instance_list.append(instance)
            seed_list.append(base_seed + i)
    return new_instance_list, seed_list


def latin_hypercube_sampling_num(min_val, max_val, size, dtype, decs=0) -> List:
    """
    creates a latin hypercube sampling for numeric values.
    """
    partition = np.linspace(start=min_val, stop=max_val, num=size + 1, dtype=dtype)

    low = partition[:size]
    high = partition[1 : size + 1]

    if dtype == np.float64:
        points = np.random.uniform(low=low, high=high, size=size)
        if decs != 0:
            points = trunc_array(points, decs)
    elif dtype == np.int32:
        if max_val - min_val <= size:
            low = min_val
            high = max_val
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
        index_sampling = latin_hypercube_sampling_num(0, options_length - 1, size, np.int32)
        points = [options[i] for i in index_sampling]

    return points


def initialization(
    poblation_size: int,
    float_parameters_list: List[FloatParam] = [],
    int_parameters_list: List[IntParam] = [],
    cat_parameters_list: List[CatParam] = [],
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
            decs=param.decs,
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


def execute_ACOTSP(
    instance: str,
    seed: int,
    executable_path: str,
    alpha: float,
    beta: float,
    rho: float,
    q0: float,
    ants: int,
    nnants: int,
    localsearch: int,
    optimum: int
):
    """
    function to execute ACOTSP program, it returns its output.
    """
    cmd = [
        executable_path,
        "--tsplibfile",
        instance,
        "--mmas",
        "--seed",
        str(seed),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--rho",
        str(rho),
        "--q0",
        str(q0),
        "--ants",
        str(ants),
        "--nnants",
        str(nnants),
        "--optimum",
        str(optimum)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))

    return output * -1


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

    return output * -1


def execute_AntKnapsack(
    instance: str,
    seed: int,
    executable_path: str,
    evaluations: int,
    total_ants: int,
    alpha: float,
    beta: float,
    tau_max: float,
    tau_min: float,
    rho: float,
) -> float:
    """
    function to execute AntKnapsackClean-master program, it returns its output.
    """
    cmd = [
        executable_path,
        instance,
        str(seed),
        str(total_ants),
        str(evaluations),
        str(alpha),
        str(beta),
        str(tau_max),
        str(tau_min),
        str(rho),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))
    return output * -1


def execute_CodGA(
    instance: str,
    seed: int,
    executable_path: str,
    p_c: float,
    p_m: float,
    N: int,
    t_max: int,
    c_0: int,
    output: str,
) -> float:
    """
    executing CodGA program, returning its output (optimal diff)
    """
    cmd = [
        executable_path,
        instance,
        output,
        str(p_c),
        str(p_m),
        str(N),
        str(t_max),
        str(seed),
        str(c_0),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    try:
        output = float(result.stdout.decode("utf-8"))
    except Exception:
        print(cmd)
        raise ValueError

    return output * -1


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
    evaluation_keys = ["instance_name", "seed", "score"]
    result_list = [
        [instance, seed, algorithm(instance, seed, **kwargs)]
        for instance, seed in zip(instance_list, seed_list)
    ]

    return pd.DataFrame(columns=evaluation_keys, data=result_list)


def evaluate_batch(
    batch: pd.DataFrame, batch_evaluations: Dict[int, pd.DataFrame], algorithm: Callable, **kwargs
) -> Dict[int, pd.DataFrame]:
    """
    given a batch of configurations and the algorithm to evaluate them
    this function returns a numpy array with its corresponding performing values
    """
    batch = batch.drop(columns=["Step_Found"], inplace=False)
    for idx, conf in batch.iterrows():
        batch_evaluations[idx] = configuration_evaluation(algorithm, **{**dict(conf), **kwargs})

    return batch_evaluations


def create_model(X_array: list):
    """
    This function creates the model to be used for tunning
    """
    # Create model
    normalizer = preprocessing.Normalization(
        axis=-1,
        input_shape=[
            X_array.shape[1],
        ],
    )
    normalizer.adapt(X_array)

    model = Sequential()
    model.add(normalizer)
    model.add(Dense(8, activation="relu", input_shape=X_array.shape))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(units=1))
    # Compile model
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


# def generate_configurations(model, index_start = 0, **kwargs) -> list:
#     """
#     Generates new configurations by selecting random configurations
#     """
#     poblation_size = kwargs["poblation_size"]
#     generated_X = pd.DataFrame(initialization(**kwargs))

#     A = model.predict(generated_X.values)
#     #A = np.squeeze(model.predict(generated_X))
#     #top_80 = np.percentile(A, 80)
#     #generated_X = generated_X[np.squeeze(np.argwhere(A > top_80))]
#     partition_A = np.argpartition(A[:, -1], -poblation_size//2)[-poblation_size//2:]
#     generated_x = generated_X.filter(items = partition_A, axis=0).reset_index(drop=True)
#     if index_start != 0:
#         generated_x.index = generated_x.index + index_start
#     return generated_x


def generate_configurations(model, batch, all_params, index_start=0) -> list:
    """
    Generates new configurations by mutating two random genes with a random value.
    """
    batch = batch.drop(columns=["Step_Found"], inplace=False)
    generated_X = pd.DataFrame()

    max_tries = 5
    p_worse = 0.25
    for _, conf in batch.iterrows():
        # Select random parameter
        n_conf = conf.copy()
        prediction = model.predict(np.array([n_conf.values]))[0][0]

        # Explore its neighbour by randomly changing it
        look_choice = random.uniform(0, 1)
        for _ in range(max_tries):
            random_choice = random.randint(0, len(batch.columns) - 1)
            param = all_params[batch.columns[random_choice]]
            if isinstance(param, IntParam):
                n_conf[random_choice] = random.randint(param.min_val, param.max_val)
            elif isinstance(param, FloatParam):
                n_conf[random_choice] = random.uniform(param.min_val, param.max_val)
                # truncate
                n_conf[random_choice] = trun_float(n_conf[random_choice], param.decs)
            else:
                choice = random.randint(0, len(param.values) - 1)
                n_conf[random_choice] = param.values[choice]

            in_generated = False
            if not generated_X.empty:
                in_generated = any([all(row) for i, row in (generated_X == n_conf).iterrows()])
            in_batch = any([all(row) for i, row in (batch == n_conf).iterrows()])

            if not in_batch and not in_generated:
                n_prediction = model.predict(np.array([n_conf.values]))[0][0]

                # Until we find a first expected (predicted) improvement
                if look_choice > p_worse:
                    if n_prediction >= prediction:
                        break
                # Controled by chance of prefering worse solutions for diversification
                else:
                    if n_prediction <= prediction:
                        break

        generated_X = generated_X.append(n_conf, ignore_index=True)

    if index_start != 0:
        generated_X.index = generated_X.index + index_start
    return generated_X


def select_configurations_by_ranking(
    batch,
    batch_evaluations,
    generated_X,
    generated_evaluations,
    instance_list,
    poblation_size,
    friedman_test=True,
):
    total_evaluations = {**batch_evaluations, **generated_evaluations}

    cols = ["instance_name"] + [f"conf_{idx}" for idx in total_evaluations]

    df = pd.DataFrame(columns=cols)
    for instance_name in instance_list:
        evaluation_dict = {
            f"conf_{idx}": total_evaluations[idx]
            .loc[total_evaluations[idx]["instance_name"] == instance_name]["score"]
            .mean()
            for idx in total_evaluations
        }
        new_row = {**{"instance_name": instance_name}, **evaluation_dict}
        df = df.append(new_row, ignore_index=True)
    df = df.rank(axis=1, numeric_only=True)

    if friedman_test:
        # compare samples
        args = tuple(df[conf] for conf in df)
        stat, p = friedmanchisquare(*args)

        # interpret
        alpha = 0.05
        if p < alpha:
            # Different distributions (reject H0)
            posthoc_matrix = sp.posthoc_conover(
                pd.melt(df, var_name="configuration", value_name="rank"),
                val_col="rank",
                group_col="configuration",
                p_adjust="holm",
            )
            different_dists = posthoc_matrix.index[
                posthoc_matrix[df.sum().idxmax()] < 0.05
            ].tolist()
            df.drop(different_dists, axis=1, inplace=True)

    best_confs = df.sum().nlargest(poblation_size)
    best_indexes = [int(idx.replace("conf_", "")) for idx, *_ in best_confs.iteritems()]

    batch = pd.concat([batch, generated_X]).filter(best_indexes, axis=0).reset_index(drop=True)
    batch_evaluations = {j: total_evaluations[idx] for j, idx in enumerate(best_indexes)}

    return batch, batch_evaluations


def select_configurations_by_mean(batch_evaluations, generated_evaluations):
    total_evaluations = {**batch_evaluations, **generated_evaluations}

    total_score = np.array([total_evaluations[i]["score"].mean() for i in total_evaluations])
    best_indexes = np.argpartition(total_score, -poblation_size)[-poblation_size:]

    batch = pd.concat([batch, generated_X]).filter(best_indexes, axis=0).reset_index(drop=True)
    batch_evaluations = {j: total_evaluations[idx] for j, idx in enumerate(best_indexes)}
    return batch, batch_evaluations


def naive_tunning(algorithm: Callable, configurations: List[dict], **kwargs) -> Tuple[List, float]:
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


def evo_tunning(
    all_params,
    budget,
    poblation_size,
    initial_batch,
    execute_algorithm,
    returning_type="RAW_VALUE",
    float_params=None,
    int_params=None,
    cat_params=None,
    **function_kwargs,
):
    # EVALUATE BATCH
    # Make inputs / outputs
    batch = pd.DataFrame(initial_batch)
    batch["Step_Found"] = 0
    evaluation_keys = ["instance_name", "seed", "score"]
    batch_evaluations = {idx: pd.DataFrame(columns=evaluation_keys) for idx, *_ in batch.iterrows()}
    batch_evaluations = evaluate_batch(
        batch, batch_evaluations, execute_algorithm, **function_kwargs
    )

    cur_budget = budget - len(batch) * len(function_kwargs["seed_list"])

    X = batch.drop(columns=["Step_Found"], inplace=False).values
    y = np.array([batch_evaluations[i]["score"].mean() for i in batch_evaluations])

    # Create model (this should consider the instance label)
    model = create_model(X)
    history = model.fit(X, y, epochs=50, verbose=0, validation_split=0.2)

    # Set queue to update
    reserve_X = pd.DataFrame()
    reserve_y = np.array([])

    i = 1
    print("Total Budget:", cur_budget)
    while cur_budget > 0:
        print("Budget left", cur_budget)
        # Generate new random configurations
        generated_X = generate_configurations(model, batch, all_params, poblation_size)
        generated_X["Step_Found"] = i
        # Evaluate them
        generated_evaluations = {
            idx: pd.DataFrame(columns=evaluation_keys)
            for idx in range(poblation_size, poblation_size + len(generated_X))
        }
        generated_evaluations = evaluate_batch(
            generated_X, generated_evaluations, execute_algorithm, **function_kwargs
        )

        cur_budget = cur_budget - len(generated_X) * len(function_kwargs["seed_list"])

        generated_y = np.array(
            [generated_evaluations[i]["score"].mean() for i in generated_evaluations]
        )

        # Update model
        # update = i % update_cycle == 0
        if cur_budget > budget / 2:
            history = model.fit(
                generated_X.drop(columns=["Step_Found"], inplace=False),
                generated_y,
                epochs=50,
                verbose=0,
                validation_split=0.2,
            )
            reserve_X = pd.DataFrame()
            reserve_y = np.array([])
        # else:
        #     reserve_X = pd.concat([reserve_X, generated_X]).reset_index(drop=True)
        #     reserve_y = np.concatenate((reserve_y, generated_y))

        # Select configurations for next step
        batch, batch_evaluations = select_configurations_by_ranking(
            batch,
            batch_evaluations,
            generated_X,
            generated_evaluations,
            function_kwargs["instance_list"],
            poblation_size,
            friedman_test=True,
        )

        # Fill if we don't have enough solutions
        if len(batch) < poblation_size:
            fillers = pd.DataFrame(
                initialization(poblation_size - len(batch), float_params, int_params, cat_params)
            )
            fillers["Step_Found"] = i
            fillers_evaluations = {
                idx: pd.DataFrame(columns=evaluation_keys)
                for idx in range(poblation_size - len(batch), poblation_size)
            }
            fillers_evaluations = evaluate_batch(
                fillers, fillers_evaluations, execute_algorithm, **function_kwargs
            )

            cur_budget = cur_budget - len(fillers) * len(function_kwargs["seed_list"])

            batch = pd.concat([batch, fillers]).reset_index(drop=True)
            batch_evaluations = {**batch_evaluations, **fillers_evaluations}

        i += 1

    batch["VALUE"] = np.array([batch_evaluations[i]["score"].mean() for i in batch_evaluations])
    if returning_type == "ABSOLUTE_OPTIMAL_DIFF":
        batch["VALUE"] = batch["VALUE"] * -1
    batch = batch.sort_values(by=["VALUE"])

    return batch
