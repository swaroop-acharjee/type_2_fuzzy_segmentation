import numpy as np
import math
import global_segmentation
import create_pgm

Z = 1					# depth
Y = 217					# height
X = 181					# col
C = [5,12,79,220]	    # Clusters

m = 2.5


# Take the 3-D image volume
img_vol = []
for j in range(Z):
	img_vol.append(global_segmentation.read_pgm('E:\\Notes\\type_2_fuzzy_segmentation\\Image Slices\\TEST_60.pgm'))


# Initial value of the membership values
global_u  = np.full((len(C),Z,Y,X),0.5).tolist()

_ = 1
while True:
	global_new_u = np.full((len(C),Z,Y,X),0).tolist()
	p_d 	  	 = np.full((Z,Y,X),0).tolist()
	N 		  	 = len(C)
	

	# Calculating the new global membership functions
	p = (1/(m-1))
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					distance = math.pow((C[i] - img_vol[j][k][l]),2)
					ln_value = math.log(math.pow(global_u[i][j][k][l],m))
					numeratr = abs(distance - ln_value - 1)
					
					temp = 0
					for r in range(N):
						distance = math.pow((C[r] - img_vol[j][k][l]),2)
						ln_value = math.log(math.pow(global_u[r][j][k][l],m))
						denomint = abs(distance - ln_value - 1)
						temp    += ((numeratr/denomint)**p)
					global_new_u[i][j][k][l] = (1/temp) 

	
	# Calculating the new cluster centers
	new_C = np.full((len(C),),0).tolist()
	for i in range(N):
		nu = 0
		de = 0
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					nu += (global_new_u[i][j][k][l]**1)*img_vol[j][k][l]
					de += (global_new_u[i][j][k][l]**1)
		new_C[i] = (nu/de)	


	# Calculating the error in the iternation
	sume = 0
	for pair in zip(C,new_C):
		sume += abs(pair[0] - pair[1])

	e = sume/2
	C = new_C
	global_u = global_new_u

	
	print(f"Iteraton: {_}. Error: {e}")
	if e < 0.00001 or _ >= 1000:
		break
	else:
		_ += 1


for j in range(Z):
	for k in range(Y):
		for l in range(X):
			max_member = []
			for i in range(len(C)):
				max_member.append((global_u[i][j][k][l],i))
			max_clus = max(max_member)[1]
			for i in range(len(C)):
				if i == max_clus:
					global_u[i][j][k][l] = 255
				else:
					global_u[i][j][k][l] = 0



for i in range(len(C)):
	print()
	create_pgm.create_pgm_file(X,Y,f"E:\\Notes\\type_2_fuzzy_segmentation\\Results\\cluster{i}.pgm",f"Segmentation result of first cluster",global_u[i])
	print()

print("Final Cluster Centers: ",C)



