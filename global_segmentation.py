import matplotlib.pyplot as plt
import numpy as np
import math
from statistics import mean

def read_pgm(file_path:str)->list:
	with open(file_path,mode='rb') as FILE:
		records = [row.decode('utf-8') for row in FILE.readlines()]

		img = []
		for i in records[4:]:
			temp = [int(x) for x in i.split()] 
			img.append(temp)

		return img

def find_neighbor(j:int,k:int,l:int,Z:int,Y:int,X:int)->list:
	coordinates = []
	
	for z in [-1,0,1]:
		for y in [-1,0,1]:
			for x in [-1,0,1]:
				z_c, y_c, x_c = j + z, k + y, l + x

				condition_1 = (z_c >= 0) and (y_c >= 0) and (x_c >= 0) 		# Checking if all the index are positive
				condition_2 = z_c <= (Z-1) 									# Checking if z is in the range
				condition_3 = y_c <= (Y-1) 									# Checking if y is in the range
				condition_4 = x_c <= (X-1) 									# Checking if x is in the range

				if condition_1 and condition_2 and condition_3 and condition_4:
					coordinates.append((z_c,y_c,x_c))
					
	return coordinates

def mean_distance(img:list,lst_coordinates:list,cluster_center:int)->list:
	coordinates = [abs(cluster_center - img[x[0]][x[1]][x[2]]) for x in lst_coordinates]
	return mean(coordinates)


def likelihood(img:list,local_mem:list,lst_coordinates)->list:
	nume = 0
	deno = 0
	for x in lst_coordinates:
		nume += local_mem[x[0]][x[1]][x[2]]*img[x[0]][x[1]][x[2]]
		deno += img[x[0]][x[1]][x[2]]
	return (nume/deno)

	