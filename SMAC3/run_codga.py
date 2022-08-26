import sys

import os
import numpy as np
import subprocess

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import NotEqualsCondition
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

SEED = int(sys.argv[1])

def choose_instances(path: str):
    instance_list = []
    tsp = False
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            if file != ".DS_Store":
                instances = sorted(os.listdir(file_path))
                instances = [os.path.join(path, file, ins) for ins in instances]
                instance_list += [instances]
                
        elif file.endswith(".tsp"):
            tsp = True
            instance_list.append(path + "/" + file)
    if tsp:
        instance_list = random.sample(instance_list)
    return instance_list

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

    return output

def codga(cs, seed, instance):
    value = execute_CodGA(
        instance=instance,
        seed=seed,
        executable_path="../CodGA/ga-nk",
		p_c=cs["p_c"],
		p_m=cs["p_m"],
		N=cs["N"],
		t_max=1000, 
		c_0=cs["c_0"],
		output="xd.txt",
    )
    return value

if __name__ == "__main__":
    path = "../InstanciasNKLandscapes/Instances-Training" 
    instance_list = choose_instances(path)

    # Define your hyperparameters
    configspace = ConfigurationSpace()

    # Float
    p_c = UniformFloatHyperparameter("p_c", 0.001, 1, log=False)
    p_m = UniformFloatHyperparameter("p_m", 0.001, 1, log=False)
    # Integer
    N = UniformIntegerHyperparameter("N", 1, 50, log=False)
	# Categorical
    c_0 = CategoricalHyperparameter('c_0', [1,2])

    parameters = [
		p_c,
		p_m,
		N,
		c_0
    ]

    for param in parameters:
        configspace.add_hyperparameter(param)

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 5000,  # Max number of function evaluations (the more the better)
        "deterministic": 0,
        "cs": configspace,
        "instances": instance_list,
    })

    # intensifier parameters
    # if no argument provided for budgets, hyperband decides them based on the number of instances available
    intensifier_kwargs = {
        'eta': 3,
        # You can also shuffle the order of using instances by this parameter.
        # 'shuffle' will shuffle instances before each SH run and 'shuffle_once'
        # will shuffle instances once before the 1st SH iteration begins
		"n_seeds": 5,
        'instance_order': None,
    }

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(SEED),
        tae_runner=codga,
        º=intensifier_kwargs
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_costs = []
    for i in instance_list:
        cost = smac.get_tae_runner().run(configspace.get_default_configuration(), i[0])[1]
        def_costs.append(cost)

    print("Value for default configuration: %.4f" % (np.mean(def_costs)))

    # Start optimization
    try:
        incumbent = smac.optimize()
    except Exception as e:
        print(e)
    finally:
        incumbent = smac.solver.incumbent

    # inc_costs = []
    # for i in instance_list:
    #     cost = smac.get_tae_runner().run(incumbent, i[0])[1]
    #     inc_costs.append(cost)
    print(incumbent)
    # print("Optimized Value: %.4f" % (np.mean(inc_costs))) 2913.38s
