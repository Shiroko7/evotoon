import sys

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

optimal_solutions = {
	"a280" : 2579,
	"ali535" : 202339,
	"att48" : 10628,
	"att532" : 27686,
	"bayg29" : 1610,
	"bays29" : 2020,
	"berlin52" : 7542,
	"bier127" : 118282,
	"brazil58" : 25395,
	"brd14051" : 469385,
	"brg180" : 1950,
	"burma14" : 3323,
	"ch130" : 6110,
	"ch150" : 6528,
	"d198" : 15780,
	"d493" : 35002,
	"d657" : 48912,
	"d1291" : 50801,
	"d1655" : 62128,
	"d2103" : 80450,
	"d15112" : 1573084,
	"d18512" : 645238,
	"dantzig42" : 699,
	"dsj1000" : 18659688, 
	"dsj1000" : 18660188,
	"eil51" : 426,
	"eil76" : 538,
	"eil101" : 629,
	"fl417" : 11861,
	"fl1400" : 20127,
	"fl1577" : 22249,
	"fl3795" : 28772,
	"fnl4461" : 182566,
	"fri26" : 937,
	"gil262" : 2378,
	"gr17" : 2085,
	"gr21" : 2707,
	"gr24" : 1272,
	"gr48" : 5046,
	"gr96" : 55209,
	"gr120" : 6942,
	"gr137" : 69853,
	"gr202" : 40160,
	"gr229" : 134602,
	"gr431" : 171414,
	"gr666" : 294358,
	"hk48" : 11461,
	"kroA100" : 21282,
	"kroB100" : 22141,
	"kroC100" : 20749,
	"kroD100" : 21294,
	"kroE100" : 22068,
	"kroA150" : 26524,
	"kroB150" : 26130,
	"kroA200" : 29368,
	"kroB200" : 29437,
	"lin105" : 14379,
	"lin318" : 42029,
	"linhp318" : 41345,
	"nrw1379" : 56638,
	"p654" : 34643,
	"pa561" : 2763,
	"pcb442" : 50778,
	"pcb1173" : 56892,
	"pcb3038" : 137694,
	"pla7397" : 23260728,
	"pla33810" : 66048945,
	"pla85900" : 142382641,
	"pr76" : 108159,
	"pr107" : 44303,
	"pr124" : 59030,
	"pr136" : 96772,
	"pr144" : 58537,
	"pr152" : 73682,
	"pr226" : 80369,
	"pr264" : 49135,
	"pr299" : 48191,
	"pr439" : 107217,
	"pr1002" : 259045,
	"pr2392" : 378032,
	"rat99" : 1211,
	"rat195" : 2323,
	"rat575" : 6773,
	"rat783" : 8806,
	"rd100" : 7910,
	"rd400" : 15281,
	"rl1304" : 252948,
	"rl1323" : 270199,
	"rl1889" : 316536,
	"rl5915" : 565530,
	"rl5934" : 556045,
	"rl11849" : 923288,
	"si175" : 21407,
	"si535" : 48450,
	"si1032" : 92650,
	"st70" : 675,
	"swiss42" : 1273,
	"ts225" : 126643,
	"tsp225" : 3916,
	"u159" : 42080,
	"u574" : 36905,
	"u724" : 41910,
	"u1060" : 224094,
	"u1432" : 152970,
	"u1817" : 57201,
	"u2152" : 64253,
	"u2319" : 234256,
	"ulysses16" : 6859,
	"ulysses22" : 7013,
	"usa13509" : 19982859,
	"vm1084" : 239297,
	"vm1748" : 336556,
}

instance_list = [
 	['../ALL_tsp/ch150.tsp'],
	['../ALL_tsp/u724.tsp'],
	['../ALL_tsp/rl1323.tsp'],
	['../ALL_tsp/st70.tsp'],
	['../ALL_tsp/d1655.tsp'],
	['../ALL_tsp/u1817.tsp'],
	['../ALL_tsp/kroB200.tsp'],
	['../ALL_tsp/gr96.tsp'],
	['../ALL_tsp/pr226.tsp'],
	['../ALL_tsp/gr431.tsp'],
	['../ALL_tsp/gr137.tsp'],
	['../ALL_tsp/att532.tsp'],
	['../ALL_tsp/rd100.tsp'],
	['../ALL_tsp/rd400.tsp'],
	['../ALL_tsp/pcb1173.tsp'],
	['../ALL_tsp/eil76.tsp'],
	['../ALL_tsp/pr264.tsp'],
	['../ALL_tsp/pr439.tsp'],
	['../ALL_tsp/u574.tsp'],
	['../ALL_tsp/bier127.tsp'],
]

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
    if not dlb:
        dlb = 1
    if not nnls:
        nnls = 1
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
        "--dlb",
        str(dlb),
        "--optimum",
        str(optimum)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))

    return output 

def acotsp(cs, seed, instance):
    optimum = optimal_solutions[instance.split("/")[2].split(".")[0]]
    nnls = []
    value = execute_ACOTSP(
        instance=instance,
        seed=seed,
        optimum=optimum,
        executable_path="../ACOTSP-master/acotsp",
        alpha=cs["alpha"],
        beta=cs["beta"],
        rho=cs["rho"],
        ants=cs["ants"],
        nnls=cs.get("nnls"),
        elitistants=cs["elitistants"],
        localsearch=cs["localsearch"],
        dlb=cs.get("dlb"),
    )
    return value

if __name__ == "__main__":
    # Define your hyperparameters
    configspace = ConfigurationSpace()

    # Float
    alpha = UniformFloatHyperparameter("alpha", 1, 10, log=False)
    beta = UniformFloatHyperparameter("beta", 1, 10, log=False)
    rho = UniformFloatHyperparameter("rho", 0, 1, log=False)
    # Integer
    ants = UniformIntegerHyperparameter("ants", 5, 100, log=False)
    nnls = UniformIntegerHyperparameter("nnls", 5, 50, log=False)
    elitistants = UniformIntegerHyperparameter("elitistants", 1, 750, log=False)
    # Categorical
    localsearch = CategoricalHyperparameter('localsearch', [0,1,2,3])
    dlb = CategoricalHyperparameter('dlb', [0,1])

    parameters = [
        alpha,
        beta,
        rho,
        ants,
        nnls,
        elitistants,
        localsearch,
        dlb,
    ]
    conditions = [
        NotEqualsCondition(nnls, localsearch, 0),
        NotEqualsCondition(dlb, localsearch, 0)
    ]

    for param in parameters:
        configspace.add_hyperparameter(param)

    for cond in conditions:
        configspace.add_condition(cond)



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
        tae_runner=acotsp,
        intensifier_kwargs=intensifier_kwargs
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

    inc_costs = []
    for i in instance_list:
        cost = smac.get_tae_runner().run(incumbent, i[0])[1]
        inc_costs.append(cost)
    print(incumbent)
    print("Optimized Value: %.4f" % (np.mean(inc_costs)))



#python3 run_acotsp.py 961  20313.98s user 357.13s system 99% cpu 5:47:13.94 total
