## IMPORTS 
import evotoon
from data_classes import CatParam, IntParam, FloatParam

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns#

import random
import warnings
warnings.filterwarnings("ignore")

## MAKE SEED
SEED = evotoon.make_seed(465)

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


def get_best_confs(folder):
	cols = ["alpha", "beta", "rho", "ants", "nnls", "elitistants", "localsearch", "dlb", "Step_Found", "VALUE"]

	df = pd.DataFrame()
	for file in sorted(os.listdir(folder)):
		path = folder + file
		df = df.append(pd.read_csv(path, usecols= cols, sep='\t').iloc[0])

	return df.reset_index(drop=True)

def run_acotsp(save_folder, seed, alpha, beta, rho, ants, nnls, elitistants, localsearch, dlb):
	instance_list = [
		"./ALL_tsp/a280.tsp",
		"./ALL_tsp/ali535.tsp",
		"./ALL_tsp/att48.tsp",
		"./ALL_tsp/d1655.tsp",
		"./ALL_tsp/eil101.tsp",
		"./ALL_tsp/eil51.tsp",
		"./ALL_tsp/gr202.tsp",
		"./ALL_tsp/gr229.tsp",
		"./ALL_tsp/gr431.tsp",
		"./ALL_tsp/gr666.tsp",
		"./ALL_tsp/kroA150.tsp",
		"./ALL_tsp/kroB100.tsp",
		"./ALL_tsp/kroB150.tsp",
		"./ALL_tsp/kroB200.tsp",
		"./ALL_tsp/p654.tsp",
		"./ALL_tsp/pcb1173.tsp",
		"./ALL_tsp/pr124.tsp",
		"./ALL_tsp/pr136.tsp",
		"./ALL_tsp/pr264.tsp",
		"./ALL_tsp/pr76.tsp",
		"./ALL_tsp/rat575.tsp",
		"./ALL_tsp/rat99.tsp",
		"./ALL_tsp/rd100.tsp",
		"./ALL_tsp/rl1304.tsp",
		"./ALL_tsp/rl1323.tsp",
		"./ALL_tsp/u1432.tsp",
		"./ALL_tsp/vm1748.tsp",
	]
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	for file in instance_list:
		f_name = file.split("/")[-1].split(".")[0]
		f = open(save_folder+"/"+f_name+"-"+str(seed)+".txt", "a")
		try:
			out = evotoon.execute_ACOTSP(file, seed, optimal_solutions[f_name], "./ACOTSP-master/acotsp", alpha, beta, rho, ants, nnls, elitistants, localsearch, dlb) * -1
		except:
			out = str((file, seed, optimal_solutions[f_name], "./ACOTSP-master/acotsp", alpha, beta, rho, ants, nnls, elitistants, localsearch, dlb))
		f.write(str(out)+"\n")
		f.close()

if __name__ == "__main__":
	read_folder = sys.argv[1]
	save_folder = sys.argv[2]
	df = get_best_confs(read_folder)

	for i, row in df.iterrows():
		alpha = row["alpha"]
		beta = row["beta"]
		rho = row["rho"]
		ants = row["ants"]
		nnls = row["nnls"]
		elitistants = row["elitistants"]
		localsearch = row["localsearch"]
		dlb = row["dlb"]
		for j in range(SEED, SEED + 5):
			run_acotsp(f"{save_folder}{i}", j, alpha, beta, rho, ants, nnls, elitistants, localsearch, dlb)