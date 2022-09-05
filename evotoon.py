import random
import subprocess
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
import ast

from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing

from data_classes import CatParam, FloatParam, IntParam
from scipy.stats import friedmanchisquare

import scikit_posthocs as sp
import os
import re


def trunc_array(array: list, decs=0):
    """
    Function to truncate a numpy array
    """
    return np.trunc(array * 10**decs) / (10**decs)


def trun_float(number, decs=0) -> float:
    step = 10.0**decs
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
    tsp = False
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            if file != ".DS_Store":
                instances = sorted(os.listdir(file_path))
                instances = [os.path.join(path, file, ins) for ins in instances]
                instance_list += instances  # random.sample(instances, n)

        elif file.endswith(".tsp"):
            tsp = True
            instance_list.append(path + "/" + file)
    if tsp:
        instance_list = random.sample(instance_list, n)
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
        index_sampling = latin_hypercube_sampling_num(
            0, options_length - 1, size, np.int32
        )
        points = [options[i] for i in index_sampling]

    return points


def initialization(
    population_size: int,
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
            size=population_size,
            dtype=np.float64,
            decs=param.decs,
        )

        parameter_list.append((param.name, points))

    for param in int_parameters_list:
        points = latin_hypercube_sampling_num(
            min_val=param.min_val,
            max_val=param.max_val,
            size=population_size,
            dtype=np.int32,
        )

        parameter_list.append((param.name, points))

    for param in cat_parameters_list:
        points = latin_hypercube_sampling_cat(param.values, population_size)
        parameter_list.append((param.name, points))

    # reshape
    initial_population = list()
    for i in range(population_size):
        conf = dict()
        for name, points in parameter_list:
            conf[name] = points[i]
        initial_population.append(conf)

    return initial_population


def execute_ACOTSP(
    instance: str,
    seed: int,
    optimum: int,
    executable_path: str,
    alpha: float,
    beta: float,
    rho: float,
    ants: int,
    nnls: int,
    elitistants: int,
    localsearch: int,
    dlb: int,
):
    # print(instance, seed, executable_path, alpha, beta, rho, q0, ants, nnants, localsearch, optimum)
    """
    function to execute ACOTSP program, it returns its output.
    """
    cmd = [
        executable_path,
        "--tsplibfile",
        instance,
        "--eas",
        "--seed",
        str(seed),
        "--tries",
        "1",
        "--time",
        "5",
        "--tours",
        "50",
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--rho",
        str(rho),
        "--nnls",
        str(nnls),
        "--ants",
        str(ants),
        "--elitistants",
        str(elitistants),
        "--localsearch",
        str(localsearch),
        "--dlb",
        str(dlb),
        "--optimum",
        str(optimum),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))

    return output * -1


def execute_ACOTSP2(
    instance: str,
    seed: int,
    optimum: int,
    executable_path: str,
    alpha: float,
    beta: float,
    rho: float,
    ants: int,
    nnls: int,
    elitistants: int,
    localsearch: int,
    dlb: int,
):
    # print(instance, seed, executable_path, alpha, beta, rho, q0, ants, nnants, localsearch, optimum)
    """
    function to execute ACOTSP program, it returns its output.
    """
    cmd = [
        executable_path,
        "--tsplibfile",
        instance,
        "--eas",
        "--seed",
        str(seed),
        "--tries",
        "1",
        "--time",
        "5",
        "--tours",
        "50",
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--rho",
        str(rho),
        "--nnls",
        str(nnls),
        "--ants",
        str(ants),
        "--elitistants",
        str(elitistants),
        "--localsearch",
        str(localsearch),
        "--dlb",
        str(dlb),
        "--optimum",
        str(optimum),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))

    return (output / 100 + 1) * optimum * -1


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
        str(c_0),
        str(t_max),
        str(seed),
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


def pop_n_elem(any_list, n_list):
    new_list = []
    for n in n_list:
        x = any_list[n]  # .pop(n)
        new_list.append(x)
    return new_list


def configuration_evaluation(
    algorithm: Callable,
    n_seeds: int,
    instance_list: List,
    seed_list: List,
    optimal_list: List = None,
    **kwargs,
) -> float:
    """
    interface to call different algorithms to evaluate
    """
    evaluation_keys = ["instance_name", "seed", "score"]
    if optimal_list is not None:
        random_ins = sorted(
            random.sample(range(len(instance_list)), n_seeds), reverse=True
        )

        pop_instances = pop_n_elem(instance_list, random_ins)
        pop_seeds = pop_n_elem(seed_list, random_ins)
        pop_optimums = pop_n_elem(optimal_list, random_ins)

        result_list = [
            [instance, seed, algorithm(instance, seed, optimum, **kwargs)]
            for instance, seed, optimum in zip(pop_instances, seed_list, pop_optimums)
        ]
    else:
        random_ins = sorted(
            random.sample(range(len(instance_list)), n_seeds), reverse=True
        )
        pop_instances = pop_n_elem(instance_list, random_ins)
        pop_seeds = pop_n_elem(seed_list, random_ins)

        result_list = [
            [instance, seed, algorithm(instance, seed, **kwargs)]
            for instance, seed in zip(pop_instances, seed_list)
        ]

    return pd.DataFrame(columns=evaluation_keys, data=result_list)


def evaluate_batch(
    batch: pd.DataFrame,
    batch_evaluations: Dict[int, pd.DataFrame],
    algorithm: Callable,
    n_seeds: int,
    **kwargs,
) -> Dict[int, pd.DataFrame]:
    """
    given a batch of configurations and the algorithm to evaluate them
    this function returns a numpy array with its corresponding performing values
    """

    batch = batch.drop(columns=["Step_Found"], inplace=False)

    for idx, conf in batch.iterrows():
        batch_evaluations[idx] = configuration_evaluation(
            algorithm, n_seeds, **{**dict(conf), **kwargs}
        )
    return batch_evaluations


def create_model(
    X_array: list,
    layers: int,
    neurons: list,
    activation: str = "relu",
    loss: str = "mean_squared_error",
    optimizer: str = "adam",
    **kwargs
):
    """
    This function creates the model to be used for tunning
    """
    if not neurons:
        neurons = X_array.shape[1]
    if not layers:
        layers = math.ceil(X_array.shape[1] ** 0.5)
    if layers != len(neurons):
        raise ValueError("Layers differ to neurons structure lenght")

    # Create model
    normalizer = preprocessing.Normalization(
        axis=-1,
        input_shape=[
            X_array.shape[1],
        ],
    )
    normalizer.adapt(X_array.astype(np.float32))

    model = Sequential()
    model.add(normalizer)

    model.add(Dense(neurons[0], activation=activation, input_shape=X_array.shape))
    for layer in range(1, layers):
        model.add(Dense(neurons[layer], activation=activation))
    model.add(Dense(units=1))

    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    return model


def generate_configurations(
    model,
    batch,
    all_params,
    index_start=0,
    hc_max_tries=5,
    hc_p_worse=0.25,
    update_phase=True,
) -> list:
    """
    Generates new configurations by mutating two random genes with a random value.
    """
    batch = batch.drop(columns=["Step_Found"], inplace=False)
    generated_X = pd.DataFrame()

    for _, conf in batch.iterrows():
        # Select random parameter
        n_conf = conf.copy()
        prediction = model.predict(np.array([n_conf.values]).astype(np.float32))[0][0]

        # Explore its neighbour by randomly changing it
        look_choice = random.uniform(0, 1)
        for _ in range(hc_max_tries):
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
                in_generated = any(
                    [all(row) for i, row in (generated_X == n_conf).iterrows()]
                )
            in_batch = any([all(row) for i, row in (batch == n_conf).iterrows()])

            if not in_batch and not in_generated:
                n_prediction = model.predict(
                    np.array([n_conf.values]).astype(np.float32)
                )[0][0]

                # Until we find a first expected (predicted) improvement
                if (
                    update_phase and look_choice < hc_p_worse
                ):  # chance of prefering worse solutions for diversification
                    if n_prediction <= prediction:
                        break
                # inmprovement
                else:
                    if n_prediction >= prediction:
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
    population_size,
    friedman_test=True,
    alpha=0.05,
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

    for col in cols[1:]:
        df[col] = df[col].astype("float64")

    df = df.rank(axis=1, numeric_only=True)

    if friedman_test:
        # compare samples
        args = tuple(df[conf] for conf in df)

        stat, p = friedmanchisquare(*args)

        # interpret

        if p < alpha:
            # Different distributions (reject H0)
            posthoc_matrix = sp.posthoc_conover(
                pd.melt(df, var_name="configuration", value_name="rank"),
                val_col="rank",
                group_col="configuration",
                p_adjust="holm",
            )
            different_dists = posthoc_matrix.index[
                posthoc_matrix[df.sum().idxmax()] < alpha
            ].tolist()
            df.drop(different_dists, axis=1, inplace=True)

    best_confs = df.sum().nlargest(population_size)
    best_indexes = [int(idx.replace("conf_", "")) for idx, *_ in best_confs.iteritems()]
    # print(batch, generated_X)
    # print(best_confs, best_indexes)
    batch = (
        pd.concat([batch, generated_X])
        .filter(best_indexes, axis=0)
        .reset_index(drop=True)
    )
    batch_evaluations = {
        j: total_evaluations[idx] for j, idx in enumerate(best_indexes)
    }

    return batch, batch_evaluations


def select_configurations_by_mean(batch_evaluations, generated_evaluations):
    total_evaluations = {**batch_evaluations, **generated_evaluations}

    total_score = np.array(
        [total_evaluations[i]["score"].mean() for i in total_evaluations]
    )
    best_indexes = np.argpartition(total_score, -population_size)[-population_size:]

    batch = (
        pd.concat([batch, generated_X])
        .filter(best_indexes, axis=0)
        .reset_index(drop=True)
    )
    batch_evaluations = {
        j: total_evaluations[idx] for j, idx in enumerate(best_indexes)
    }
    return batch, batch_evaluations


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


def evo_tunning(
    all_params,
    budget,
    population_size,
    update_cycle,
    hc_max_tries,
    hc_p_worse,
    n_seeds,
    initial_batch,
    execute_algorithm,
    model_kwargs,
    float_params=[],
    int_params=[],
    cat_params=[],
    alpha=0.05,
    **function_kwargs,
):
    # EVALUATE BATCH
    # Make inputs / outputs
    batch = pd.DataFrame(initial_batch)
    batch["Step_Found"] = 0
    evaluation_keys = ["instance_name", "seed", "score"]
    batch_evaluations = {
        idx: pd.DataFrame(columns=evaluation_keys) for idx, *_ in batch.iterrows()
    }
    batch_evaluations = evaluate_batch(
        batch, batch_evaluations, execute_algorithm, n_seeds, **function_kwargs
    )

    cur_budget = budget - len(batch) * n_seeds

    X = batch.drop(columns=["Step_Found"], inplace=False).values
    y = np.array([batch_evaluations[i]["score"].mean() for i in batch_evaluations])

    # Create model (this should consider the instance label)
    model = create_model(X, **model_kwargs)
    history = model.fit(
        X.astype(np.float32),
        y,
        batch_size=model_kwargs["batch_size"],
        epochs=model_kwargs["epochs"],
        validation_split=model_kwargs["validation_split"],
        verbose=0,
    )

    # Set queue to update
    reserve_X = pd.DataFrame()
    reserve_y = np.array([])

    i = 1
    print("Total Budget:", cur_budget)
    while cur_budget > 0:
        print("Budget left", cur_budget)
        update_phase = cur_budget > budget / 2
        # Generate new random configurations
        generated_X = generate_configurations(
            model,
            batch,
            all_params,
            population_size,
            hc_max_tries,
            hc_p_worse,
            update_phase,
        )
        generated_X["Step_Found"] = i
        # Evaluate them
        generated_evaluations = {
            idx: pd.DataFrame(columns=evaluation_keys)
            for idx in range(population_size, population_size + len(generated_X))
        }
        generated_evaluations = evaluate_batch(
            generated_X,
            generated_evaluations,
            execute_algorithm,
            n_seeds,
            **function_kwargs,
        )

        cur_budget = cur_budget - len(generated_X) * n_seeds

        generated_y = np.array(
            [generated_evaluations[i]["score"].mean() for i in generated_evaluations]
        )

        # Update mode
        if update_phase:
            reserve_X = pd.concat([reserve_X, generated_X]).reset_index(drop=True)
            reserve_y = np.concatenate((reserve_y, generated_y))
            if i % update_cycle == 0:
                model.fit(
                    reserve_X.drop(columns=["Step_Found"], inplace=False).astype(
                        np.float32
                    ),
                    reserve_y,
                    batch_size=model_kwargs["batch_size"],
                    epochs=model_kwargs["epochs"],
                    validation_split=model_kwargs["validation_split"],
                    verbose=0,
                )
                reserve_X = pd.DataFrame()
                reserve_y = np.array([])

        # Select configurations for next step
        batch, batch_evaluations = select_configurations_by_ranking(
            batch,
            batch_evaluations,
            generated_X,
            generated_evaluations,
            function_kwargs["instance_list"],
            population_size,
            friedman_test=True,
            alpha=alpha,
        )

        # Fill if we don't have enough solutions
        if len(batch) < population_size:
            fillers = pd.DataFrame(
                initialization(
                    population_size - len(batch), float_params, int_params, cat_params
                )
            )
            fillers["Step_Found"] = i
            fillers_evaluations = {
                idx: pd.DataFrame(columns=evaluation_keys)
                for idx in range(population_size - len(batch), population_size)
            }
            fillers_evaluations = evaluate_batch(
                fillers,
                fillers_evaluations,
                execute_algorithm,
                n_seeds,
                **function_kwargs,
            )

            cur_budget = cur_budget - len(fillers) * n_seeds

            batch = pd.concat([batch, fillers]).reset_index(drop=True)
            batch_evaluations = {**batch_evaluations, **fillers_evaluations}

        i += 1

    batch["VALUE"] = np.array(
        [batch_evaluations[i]["score"].mean() for i in batch_evaluations]
    )

    batch = batch.sort_values(by=["VALUE"])

    return batch


def read_parameters(file_path):
    file = open(file_path, "r")
    cat_params = []
    float_params = []
    int_params = []

    for line in file:
        line = line.rstrip()
        if line and not line.startswith("##"):
            name, type = line.split()[0:2]
            if type == "c":
                values = (
                    line.replace("(", "").replace(")", "").replace(",", "").split()[2:]
                )
                cat_params.append(CatParam(name, values))
            elif type == "r":
                min, max, precision = (
                    line.replace("(", "").replace(")", "").replace(",", "").split()[2:]
                )
                float_params.append(
                    FloatParam(name, float(min), float(max), int(precision))
                )
            elif type == "i":
                min, max = (
                    line.replace("(", "").replace(")", "").replace(",", "").split()[2:]
                )
                int_params.append(IntParam(name, int(min), int(max)))
            else:
                raise ValueError("unkown parameter type in parameters file")

    all_params = float_params + int_params + cat_params
    all_params = {param.name: param for param in all_params}

    return all_params, float_params, int_params, cat_params


def read_setup_file(file_path):
    file = open(file_path, "r")
    setup = {}
    for line in file:
        line = line.rstrip()
        if line and not line.startswith("##"):
            key, val = re.split("\s+=\s+", line)
            setup[key] = val
    return setup


def read_model(file_path):
    model =  read_setup_file(file_path)

    model_kwargs = {}
    if model.get("layers"):
        model_kwargs["layers"] = int(model["layers"])
    if model.get("neurons"):
        model_kwargs["neurons"] = ast.literal_eval(model["neurons"])
    if model.get("loss"):
        model_kwargs["loss"] = model["loss"]
    else:
        model_kwargs["loss"] = "mean_squared_error"
    if model.get("optimizer"):
        model_kwargs["optimizer"] = model["optimizer"]
    else:
        model_kwargs["optimizer"] = "adam"
    if model.get("activation"):
        model_kwargs["activation"] = model["activation"]
    else:
        model_kwargs["activation"] = "relu"
    if model.get("batch_size"):
        model_kwargs["batch_size"] = int(model["batch_size"])
    else:
        model_kwargs["batch_size"] = 8
    if model.get("epochs"):
        model_kwargs["epochs"] = int(model["epochs"])
    else:
        model_kwargs["epochs"] = 50
    if model.get("validation_split"):
        model_kwargs["validation_split"] = float(model["validation_split"])
    else:
        model_kwargs["validation_split"] = 0.2

    return model_kwargs


def read_instance_list(file_path):
    return [f"{file_path}/{f}" for f in os.listdir(file_path)]



def import_executer(file_path, func_name):
    spec = importlib.util.spec_from_file_location("executer", file_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["executer"] = foo
    spec.loader.exec_module(foo)
    return getattr(foo, func_name)


if __name__ == "__main__":
    import sys
    import warnings
    import importlib.util

    warnings.simplefilter(action="ignore", category=FutureWarning)

    setup_path = sys.argv[1]
    SEED = int(sys.argv[2])

    scenario = read_setup_file(setup_path + "/scenario.txt")
    alg_funct = import_executer(
        scenario["executer_file"], scenario["executer_function"]
    )

    budget = int(scenario["budget"])
    population_size = int(scenario["population_size"])
    update_cycle = int(scenario["update_cycle"])
    hc_max_tries = int(scenario["hc_max_tries"])
    hc_p_worse = float(scenario["hc_p_worse"])
    n_seeds = int(scenario["n_seeds"])
    alpha = float(scenario["alpha"])

    all_params, float_params, int_params, cat_params = read_parameters(
        setup_path + "/parameters.txt"
    )

    if scenario.get("initial_batch"):
        initial_batch = scenario["initial_batch"]
    else:
        initial_batch = initialization(
            population_size, float_params, int_params, cat_params
        )

    instance_list = read_instance_list(scenario["instances_folder"])
    instance_list, seed_list = select_seeds(instance_list, n_seeds, SEED)

    function_kwargs = {
        "executable_path": "ACOTSP-master/acotsp",
        "instance_list": instance_list,
        "seed_list": seed_list,
    }
    if scenario.get("optimal_file"):
        optimal_list = read_setup_file(scenario["optimal_file"])
        function_kwargs["optimal_list"] = [
            optimal_list[instance.split(".")[0].split("/")[-1]]
            for instance in instance_list
        ]

    model_kwargs = read_model(setup_path + "/model.txt")

    print(
        evo_tunning(
            all_params=all_params,
            float_params=float_params,
            int_params=int_params,
            cat_params=cat_params,
            budget=budget,
            population_size=population_size,
            update_cycle=update_cycle,
            hc_max_tries=hc_max_tries,
            hc_p_worse=hc_p_worse,
            n_seeds=n_seeds,
            initial_batch=initial_batch,
            execute_algorithm=alg_funct,
            model_kwargs=model_kwargs,
            alpha=alpha,
            **function_kwargs,
        )
    )
