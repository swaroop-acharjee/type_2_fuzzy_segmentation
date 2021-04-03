from numpy import (
	full,
)

from create_pgm import (
	create_pgm_file,
)

from statistics import (
	mean,
	variance,
)

from math import (
	pow,
	log,
	exp,
)

from global_segmentation import (
	read_pgm,
	find_neighbor,
	mean_distance,
	likelihood,
)


'''
======================================
			CONSTANTS
======================================
'''
Z = 1					# depth
Y = 217					# height
X = 181					# col
C = [5,12,79,220]	   	# Clusters
A = 0.7
P = 1
Q = 3
M = 2.5



p = (1/(M-1))


# Take the 3-D image volume
# Change this into function
img_vol = []
for j in range(Z):
	img_vol.append(read_pgm('E:\\Notes\\type_2_fuzzy_segmentation\\Image Slices\\TEST_60.pgm'))


# Initial value of the membership values
global_u  = full((len(C),Z,Y,X),0.5).tolist()
local_u   = full((len(C),Z,Y,X),0.5).tolist()
final_u   = full((len(C),Z,Y,X),0.5).tolist()

_ = 1
while True:
	global_new_u 		= full((len(C),Z,Y,X),0).tolist()
	local_new_u  		= full((len(C),Z,Y,X),0).tolist()
	type_2_u     		= full((len(C),Z,Y,X),0).tolist()
	type_2_u_normalized = full((len(C),Z,Y,X),0).tolist()
	p_d 	  	 		= full((Z,Y,X),0).tolist()
	N 		  	 		= len(C)
	

	# Calculating the new global membership functions
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					distance = pow((C[i] - img_vol[j][k][l]),2)
					ln_value = log(pow(global_u[i][j][k][l],m))
					numeratr = abs(distance - ln_value - 1)
					
					temp = 0
					for r in range(N):
						distance = pow((C[r] - img_vol[j][k][l]),2)
						ln_value = log(pow(global_u[r][j][k][l],m))
						denomint = abs(distance - ln_value - 1)
						temp    += ((numeratr/denomint)**p)
					global_new_u[i][j][k][l] = (1/temp) 


	# Calculating the new local membership functions
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)
					f 			= likelihood(img_vol,local_u[i],coordinates)
					d 		    = mean_distance(img_vol,coordinates,C[i])
					ln_value    = log(pow(local_u[i][j][k][l],m))
					numeratr    = abs(f*d - ln_value - 1)

					temp = 0
					for r in range(N):
						coordinates = find_neighbor(j,k,l,Z,Y,X)
						f 			= likelihood(img_vol,local_u[r],coordinates)
						d 		    = mean_distance(img_vol,coordinates,C[r])
						ln_value    = log(pow(local_u[r][j][k][l],m))
						denomint    = abs(f*d - ln_value - 1)
						temp       += ((numeratr/denomint)**p)
					local_new_u[i][j][k][l] = (1/temp) 


	# Calculation of the type 2 fuzzy values
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)
					lst_u = []
					
					for x in coordinates:
						lst_u.append(local_new_u[i][x[0]][x[1]][x[2]])

					S = variance(lst_u)

					lst_y = []
					for x in lst_u:
						temp = pow((x-local_new_u[i][j][k][l]),2)
						lst_y.append(exp((-1/2)*(temp/S)))

					m     = max(lst_y)
					lst_z = [(row/m) for row in lst_y]

					a     = sum([row[0]*row[1] for row in zip(lst_y,lst_z)])
					b     = sum(lst_z)

					type_2_u[i][j][k][l] = (a/b)


	# Normalization	
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					s = 0
					for r in range(N):
						s += type_2_u[i][j][k][l]
					type_2_u_normalized[i][j][k][l] = (type_2_u[i][j][k][l]/s) 


	# Calculating the new cluster centers
	new_C = full((len(C),),0).tolist()
	for i in range(N):
		nu = 0
		de = 0
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)
					f 			= likelihood(img_vol,type_2_u_normalized[i],coordinates)
					nu += (A*(global_new_u[i][j][k][l]**m)*img_vol[j][k][l]) + ((1-A)*((type_2_u_normalized[i][j][k][l]**m)*f*img_vol[j][k][l]))
					de += (A*(global_new_u[i][j][k][l]**m)) + ((1-A)*((type_2_u_normalized[i][j][k][l]**m)*f))
		new_C[i] = (nu/de)	


	# Calculating the error in the iternation
	sume = 0
	for pair in zip(C,new_C):
		sume += abs(pair[0] - pair[1])

	e = sume/2
	C        = new_C
	global_u = global_new_u
	local_u  = type_2_u_normalized
	
	print(f"Iteraton: {_}. Error: {e}")
	if e < 0.01 or _ >= 1000:
		break
	else:
		_ += 1



# Calculating the final Membership values
for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				su = 0
				for r in range(N):
					su += (pow(global_u[r][j][k][l],P)*pow(local_u[r][j][k][l],Q))
				final_u[i][j][k][l] = ((pow(global_u[i][j][k][l],P)*pow(local_u[i][j][k][l],Q))/su)



# Return a 3-D list with the classifying label.
for j in range(Z):
	for k in range(Y):
		for l in range(X):
			max_member = []
			for i in range(len(C)):
				max_member.append((final_u[i][j][k][l],i))
			max_clus = max(max_member)[1]

			for i in range(len(C)):
				if i == max_clus:
					final_u[i][j][k][l] = 255
				else:
					final_u[i][j][k][l] = 0


for i in range(len(C)):
	print()
	create_pgm_file(X,Y,f"E:\\Notes\\type_2_fuzzy_segmentation\\Results\\cluster{i}.pgm",f"Segmentation result of first cluster",final_u[i])
	print()

print("Final Cluster Centers: ",C)