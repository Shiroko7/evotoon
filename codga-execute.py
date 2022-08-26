## IMPORTS 
import evotoon
from data_classes import CatParam, IntParam, FloatParam


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns#
import os
import sys
import random

import warnings
warnings.filterwarnings("ignore")

## MAKE SEED
SEED = evotoon.make_seed(925)

folder_list = [
		'./InstanciasNKLandscapes/Instances-Testing/nk_2_20',
		'./InstanciasNKLandscapes/Instances-Testing/nk_2_38',
		'./InstanciasNKLandscapes/Instances-Testing/nk_2_52',
		'./InstanciasNKLandscapes/Instances-Testing/nk_3_20',
		'./InstanciasNKLandscapes/Instances-Testing/nk_3_34',
		'./InstanciasNKLandscapes/Instances-Testing/nk_3_48',
		'./InstanciasNKLandscapes/Instances-Testing/nk_4_20',
		'./InstanciasNKLandscapes/Instances-Testing/nk_4_30',
		'./InstanciasNKLandscapes/Instances-Testing/nk_4_40',
		'./InstanciasNKLandscapes/Instances-Testing/nk_5_20',
		'./InstanciasNKLandscapes/Instances-Testing/nk_5_28',
		'./InstanciasNKLandscapes/Instances-Testing/nk_5_38',
		'./InstanciasNKLandscapes/Instances-Testing/nk_6_20',
		'./InstanciasNKLandscapes/Instances-Testing/nk_6_26',
		'./InstanciasNKLandscapes/Instances-Testing/nk_6_32'
 ]


def get_best_confs(folder):
	cols = ["p_c", "p_m", "N", "c_0", "Step_Found", "VALUE"]

	df = pd.DataFrame()
	for file in sorted(os.listdir(folder)):
		path = folder + file
		df = df.append(pd.read_csv(path, usecols= cols, sep='\t').iloc[0])

	return df.reset_index(drop=True)

def run_ga(save_folder, ins_folder, seed, p_c, p_m, N, c_0, t_max):
	instance_list = []
	for file in sorted(os.listdir(ins_folder)):
		file_path = os.path.join(ins_folder, file)
		instance_list.append(file_path)

	output = "lol.txt"
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	f = open(save_folder+"/"+ins_folder.split("/")[-1]+"-"+str(seed)+".txt", "a")
	for file in instance_list:
		out = evotoon.execute_CodGA(file, seed, "./CodGA/ga-nk", p_c, p_m, N, t_max, c_0, output) * -1
		f.write(str(out)+"\n")
	f.close()


if __name__ == "__main__":
	read_folder = sys.argv[1]
	save_folder = sys.argv[2]
	df = get_best_confs(read_folder)

	t_max = 100000
	for i, row in df.iterrows():
		p_c = row["p_c"]
		p_m = row["p_m"]
		N = row["N"]
		c_0 = row["c_0"]
		for j in range(SEED, SEED + 5):
			for folder in folder_list:
				run_ga(f"{save_folder}{i}", folder, j, p_c, p_m, N, c_0, t_max)