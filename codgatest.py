## IMPORTS 
import evotoon
from data_classes import CatParam, IntParam, FloatParam

import numpy as np
import pandas as pd

import random
import os

## MAKE SEED
SEED = evotoon.make_seed(765)
separator = "--------------------------------------------------------"


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
	# EVOTOON
	p_c = 0.666
	p_m = 0.122	
	N = 4
	c_0 = 1
	t_max = 1000
	for i in range(SEED, SEED + 5):
		for folder in folder_list:
			run_ga("./testing/codgatesting/evotoon/1", folder, i, p_c, p_m, N, c_0, t_max)

	p_c = 0.119
	p_m = 0.164	
	N = 45
	c_0 = 1
	t_max = 1000
	for i in range(SEED, SEED + 5):
		for folder in folder_list:
			run_ga("./testing/codgatesting/evotoon/2", folder, i, p_c, p_m, N, c_0, t_max)

	p_c = 0.990
	p_m = 0.171
	N = 8
	c_0 = 1
	t_max = 1000
	for i in range(SEED, SEED + 5):
		for folder in folder_list:
			run_ga("./testing/codgatesting/evotoon/3", folder, i, p_c, p_m, N, c_0, t_max)

	p_c = 0.517
	p_m = 0.453
	N = 38
	c_0 = 1
	t_max = 1000
	for i in range(SEED, SEED + 5):
		for folder in folder_list:
			run_ga("./testing/codgatesting/evotoon/4", folder, i, p_c, p_m, N, c_0, t_max)

	p_c = 0.245
	p_m = 0.343
	N = 34
	c_0 = 1
	t_max = 1000
	for i in range(SEED, SEED + 5):
		for folder in folder_list:
			run_ga("./testing/codgatesting/evotoon/5", folder, i, p_c, p_m, N, c_0, t_max)

	# # IRACE
	# p_c = 0.608
	# p_m = 0.353	
	# N = 34
	# c_0 = 2
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/irace/1", folder, i, p_c, p_m, N, c_0, t_max)


	# p_c = 0.616
	# p_m = 0.169
	# N = 26
	# c_0 = 2
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/irace/2", folder, i, p_c, p_m, N, c_0, t_max)


	# p_c = 0.523
	# p_m = 0.234
	# N = 22
	# c_0 = 2
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/irace/3", folder, i, p_c, p_m, N, c_0, t_max)


	# p_c = 0.616
	# p_m = 0.169
	# N = 26
	# c_0 = 2
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/irace/4", folder, i, p_c, p_m, N, c_0, t_max)

	# p_c = 0.677
	# p_m = 0.093
	# N = 33
	# c_0 = 1
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/irace/5", folder, i, p_c, p_m, N, c_0, t_max)


	# # SMAC
	# # 42
	# N = 21
	# c_0 = 1
	# p_c = 0.3150616264195509
	# p_m = 0.43468874695304377
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/smac/1", folder, i, p_c, p_m, N, c_0, t_max)

	# # 42
	# N = 49
	# c_0 = 1
	# p_c = 0.6116511379717752
	# p_m = 0.24174597998873948
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/smac/2", folder, i, p_c, p_m, N, c_0, t_max)

	# #39
	# p_c = 0.6645036294601449
	# p_m = 0.7769120878429437
	# N = 1
	# c_0 = 1
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/smac/3", folder, i, p_c, p_m, N, c_0, t_max)

	# #420
	# p_c = 0.8569581676894068
	# p_m = 0.37772192629777096
	# N = 33
	# c_0 = 1
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/smac/4", folder, i, p_c, p_m, N, c_0, t_max)

	# p_c = 0.6948840581326041
	# p_m = 0.7375712049548143
	# N = 35
	# c_0 = 1
	# t_max = 1000
	# for i in range(SEED, SEED + 5):
	# 	for folder in folder_list:
	# 		run_ga("./testing/codgatesting/smac/5", folder, i, p_c, p_m, N, c_0, t_max)