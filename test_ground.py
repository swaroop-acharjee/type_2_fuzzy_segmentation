import numpy as np
import math
import global_segmentation
import create_pgm
import statistics


Z = 1					# depth
Y = 217					# height
X = 181					# col
C = [5,12,79,220]	   	# Clusters


m = 2.5
p = (1/(m-1))

def find_neighbor(j,k,l,Z,Y,X):
	coordinates = []
	for z in [-1,0,1]:
		for y in [-1,0,1]:
			for x in [-1,0,1]:
				z_c = j + z
				y_c = k + y
				x_c = l + x
				if ((z_c >= 0) and (y_c >= 0) and (x_c >= 0)) and ((z_c <= (Z-1)) and (y_c <= (Y-1)) and (x_c <= (X-1))):
					coordinates.append((z_c,y_c,x_c))
	return coordinates

def mean_distance(img,lst_coordinates,cluster_center):
	coordinates = [abs(cluster_center - img[x[0]][x[1]][x[2]]) for x in lst_coordinates]
	return statistics.mean(coordinates)


def likelihood(img,local_mem,lst_coordinates):
	nume = 0
	deno = 0
	for x in lst_coordinates:
		nume += local_mem[x[0]][x[1]][x[2]]*img[x[0]][x[1]][x[2]]
		deno += img[x[0]][x[1]][x[2]]
	return (nume/deno)


#Take the 3-D image volume
img_vol = []
for j in range(Z):
	img_vol.append(global_segmentation.read_pgm('E:\\Notes\\type_2_fuzzy_segmentation\\Image Slices\\TEST_60.pgm'))


# Initial value of the membership values
global_u  = np.full((len(C),Z,Y,X),0.5).tolist()
local_u   = np.full((len(C),Z,Y,X),0.5).tolist()
final_u   = np.full((len(C),Z,Y,X),0.5).tolist()

_ = 1
while True:
	global_new_u 		= np.full((len(C),Z,Y,X),0).tolist()
	local_new_u  		= np.full((len(C),Z,Y,X),0).tolist()
	type_2_u     		= np.full((len(C),Z,Y,X),0).tolist()
	type_2_u_normalized = np.full((len(C),Z,Y,X),0).tolist()
	p_d 	  	 		= np.full((Z,Y,X),0).tolist()
	N 		  	 		= len(C)
	

	# Calculating the new global membership functions
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


	# Calculating the new local membership functions
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)
					f 			= likelihood(img_vol,local_u[i],coordinates)
					d 		    = mean_distance(img_vol,coordinates,C[i])
					ln_value    = math.log(math.pow(local_u[i][j][k][l],m))
					numeratr    = abs(f*d - ln_value - 1)

					temp = 0
					for r in range(N):
						coordinates = find_neighbor(j,k,l,Z,Y,X)
						f 			= likelihood(img_vol,local_u[r],coordinates)
						d 		    = mean_distance(img_vol,coordinates,C[r])
						ln_value    = math.log(math.pow(local_u[r][j][k][l],m))
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

					S = statistics.variance(lst_u)

					lst_y = []
					for x in lst_u:
						temp = math.pow((x-local_new_u[i][j][k][l]),2)
						lst_y.append(math.exp((-1/2)*(temp/S)))

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
	new_C = np.full((len(C),),0).tolist()
	for i in range(N):
		nu = 0
		de = 0
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)
					f 			= likelihood(img_vol,type_2_u_normalized[i],coordinates)
					nu += (0.7*(global_new_u[i][j][k][l]**m)*img_vol[j][k][l]) + (0.3*((type_2_u_normalized[i][j][k][l]**m)*f*img_vol[j][k][l]))
					de += (0.7*(global_new_u[i][j][k][l]**m)) + (0.3*((type_2_u_normalized[i][j][k][l]**m)*f))
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

p = 1
q = 3
for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				su = 0
				for r in range(N):
					su += (math.pow(global_u[r][j][k][l],p)*math.pow(local_u[r][j][k][l],q))
				final_u[i][j][k][l] = ((math.pow(global_u[i][j][k][l],p)*math.pow(local_u[i][j][k][l],q))/su)

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
	create_pgm.create_pgm_file(X,Y,f"E:\\Notes\\type_2_fuzzy_segmentation\\Results\\cluster{i}.pgm",f"Segmentation result of first cluster",final_u[i])
	print()

print("Final Cluster Centers: ",C)



