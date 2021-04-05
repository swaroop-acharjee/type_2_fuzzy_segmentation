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
	intersection,
)


'''
======================================
			CONSTANTS
======================================
'''
START = 50
END   = 52
Z 	  = (END-START)			# depth
Y 	  = 217					# height
X     = 181					# col
C     = [5,12,79,220]	   	# Values of the initial Clusters
A     = 0.7					# Value of alpha
P     = 1
Q     = 3
M     = 2.5
E     = 0.01
number_of_iterations = 2





p = (1/(M-1))


# Take the 3-D image volume
# Change this into function
img_vol = []
for j in range(START,END):
	img_vol.append(read_pgm(f'Data\\Brain Volume\\TEST_{j}.pgm'))

print(len(img_vol))

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
					ln_value = log(pow(global_u[i][j][k][l],M))
					numeratr = abs(distance - ln_value - 1)
					
					temp = 0
					for r in range(N):
						distance = pow((C[r] - img_vol[j][k][l]),2)
						ln_value = log(pow(global_u[r][j][k][l],M))
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
					ln_value    = log(pow(local_u[i][j][k][l],M))
					numeratr    = abs(f*d - ln_value - 1)

					temp = 0
					for r in range(N):
						coordinates = find_neighbor(j,k,l,Z,Y,X)
						f 			= likelihood(img_vol,local_u[r],coordinates)
						d 		    = mean_distance(img_vol,coordinates,C[r])
						ln_value    = log(pow(local_u[r][j][k][l],M))
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

					lst_z = [(row/max(lst_y)) for row in lst_y]

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
					f 	= likelihood(img_vol,type_2_u_normalized[i],coordinates)
					nu += (A*(global_new_u[i][j][k][l]**M)*img_vol[j][k][l]) + ((1-A)*((type_2_u_normalized[i][j][k][l]**M)*f*img_vol[j][k][l]))
					de += (A*(global_new_u[i][j][k][l]**M)) + ((1-A)*((type_2_u_normalized[i][j][k][l]**M)*f))
		new_C[i] = (nu/de)	


	# Calculating the error in the iternation
	sume = 0
	for pair in zip(C,new_C):
		sume += abs(pair[0] - pair[1])

	e 		 = sume/2
	C        = new_C
	global_u = global_new_u
	local_u  = type_2_u_normalized
	
	print(f"Iteraton: {_}\t\t. Error: {e}")

	
	if e < E or _ >= number_of_iterations:
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



classifiend_img   	  = full((Z,Y,X),0).tolist()
sum_vpc = 0
sum_vpe = 0

# Return a 3-D list with the classifying label.
for j in range(Z):
	for k in range(Y):
		for l in range(X):
			max_member = []
			for i in range(len(C)):
				max_member.append((final_u[i][j][k][l],i))
				sum_vpc += pow(final_u[i][j][k][l],2)
				sum_vpe -= final_u[i][j][k][l]*log(final_u[i][j][k][l])

			classifiend_img[j][k][l]   = max(max_member)[1] 



print("\n\nCreating the PGM files\n\n")
ground_truth = []
for j in range(Z):
	print(START+j)
	create_pgm_file(X,Y,f"Data\\Test Results\\TEST_{START+j}.pgm",f"Label {START + j}",classifiend_img[j],3)
	ground_truth.append(read_pgm(f'Data\\Ground Truth\\TEST_{START+j}.pgm'))
	print()

correct_classified = 0
mis_classified     = 0


# Misclassification Error
test_img_cluster   = {0:[],1:[],2:[],3:[]}
ground_img_cluster = {0:[],1:[],2:[],3:[]}
sum_sa  = 0
sum_dsc = 0


for j in range(Z):
	for k in range(Y):
		for l in range(X):
			if ground_truth[j][k][l] in [0,1,2,3]:
				test_img_cluster[classifiend_img[j][k][l]].append((j,k,l))
				ground_img_cluster[ground_truth[j][k][l]].append((j,k,l))
				
				if ground_truth[j][k][l] == classifiend_img[j][k][l]:
					correct_classified += 1
				else:
					mis_classified += 1



for key in test_img_cluster:
    common_points = intersection(test_img_cluster[key],ground_img_cluster[key])
    x = len(common_points)
    y = len(ground_img_cluster[key])
    sum_sa += (x/y)

    dsc_nume = 2*len(common_points)
    dsc_deno = (len(test_img_cluster[key]) + len(ground_img_cluster[key]))
    sum_dsc += (dsc_nume/dsc_deno)

dsc    			 = (sum_dsc/4)
avg_sa 			 = (sum_sa/4)
error_percentage = ((mis_classified/(correct_classified+mis_classified))*100)
vpc 			 = sum_vpc/(X*Y*Z)
vpe 			 = sum_vpe/(X*Y*Z)	





print("\n=========================================================\n")
print("\t\t\tRESULTS\n\n\n")
print("\n=========================================================\n")

print("Final Cluster Centers: ",C)
print("Total Misclassification Error: ",error_percentage)
print("Average Segmentation Accuracy: ",avg_sa)
print("Dice Similarity Coefficient: ",dsc)
print("Partition Coefficient: ",vpc)
print("Partitition Entropy: ",vpe)
