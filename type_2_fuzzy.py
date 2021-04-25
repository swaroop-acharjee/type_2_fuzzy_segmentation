from global_segmentation import (
	read_pgm,
	find_neighbor,
	mean_distance,
	likelihood,
	intersection,
	mean_pixels,
)

from numpy import (
	full,
	log as ln,
)

from math import (
	log,
	exp,
	pow,
)

from statistics import (
		variance,
		mean,
	)

START = 50
END   = 101
Z 	  = (END-START)						# depth
Y 	  = 217								# height
X     = 181								# col
N_IT  = 100

C     = [13.01,45.01,100.01,115.01]	   	# Values of the initial Clusters
A     = 0.7								# Value of alpha
N     = len(C)
P     = 1
Q     = 3
M     = 2.5
E     = 0.01
p 	  = (1/(M-1))


# Creating a 3-D Matrix of the Image Voxels
img_vol 	 = []
ground_truth = []
for j in range(START,END):
    img_vol.append(read_pgm(f'Data\\Brain Volume\\TEST_{j}.pgm'))
    ground_truth.append(read_pgm(f'Data\\Ground Truth\\TEST_{j}.pgm'))


global_u = full((N,Z,Y,X),0).tolist()
local_u  = full((N,Z,Y,X),0).tolist()
final_u  = full((N,Z,Y,X),0).tolist()


# Calculating the initial values of the membership functions
for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				temp = 0
				numerator = (img_vol[j][k][l] - C[i])**2
				for r in range(N):
					denominator =  (img_vol[j][k][l] - C[r])**2
					temp        += ((numerator/denominator)**p)
				local_u[i][j][k][l]  = (1/temp)	
				global_u[i][j][k][l] = (1/temp)


for _ in range(N_IT):

	# Calculation of the global membership functions
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					distance  = ((C[i] - img_vol[j][k][l])**2)
					ln_value  = log(global_u[i][j][k][l]**M)
					numerator = abs(distance - ln_value - 1)

					temp 	  = 0
					for r in range(N):
						distance    = ((C[r] - img_vol[j][k][l])**2)
						ln_value    = log(global_u[r][j][k][l]**M)
						denominator = abs(distance - ln_value - 1)

						temp       += ((numerator/denominator)**p) 
					
					global_u[i][j][k][l] = (1/temp) 

	# Calculate the Local membership functions
	for i in range(N):
		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates = find_neighbor(j,k,l,Z,Y,X)

					f           = likelihood(img_vol,local_u[i],coordinates)
					d           = mean_distance(img_vol,coordinates,C[i])

					ln_value    = log(local_u[i][j][k][l]**M)
					numerator   = abs(f*d - ln_value - 1)


					temp = 0
					for r in range(N):
						f     	    = likelihood(img_vol,local_u[r],coordinates)
						d           = mean_distance(img_vol,coordinates,C[r])

						ln_value    = log(local_u[r][j][k][l]**M)
						denominator = abs(f*d - ln_value - 1)

						temp 	   += (numerator/denominator)**p

					local_u[i][j][k][l]  = (1/temp)


	# Calculating the new cluster centers.
	new_C 		= [0,0,0,0]

	for i in range(N):	
		numerator   = 0
		denominator = 0

		for j in range(Z):
			for k in range(Y):
				for l in range(X):
					coordinates   = find_neighbor(j,k,l,Z,Y,X)

					f             = likelihood(img_vol,local_u[i],coordinates)
					a_mean        = mean_pixels(img_vol,coordinates)

					n_a = (A*(global_u[i][j][k][l]**M))
					n_b = ((1-A)*(local_u[i][j][k][l]**M*f))

					numerator   += n_a*img_vol[j][k][l] + n_b*a_mean
					denominator += n_a + n_b

		new_C[i] = numerator/denominator


	e_0 = abs(C[0] - new_C[0])
	e_1 = abs(C[1] - new_C[1])
	e_2 = abs(C[2] - new_C[2])
	e_3 = abs(C[3] - new_C[3])
	avg_error = ((e_0 + e_1 + e_2 + e_3)/4)
	


	C  = new_C

	print()
	print((_+1))
	print(C)
	print(avg_error)
	print()


	if avg_error < E:
		break


# Normalization of the final membership functions
for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				su = 0
				for r in range(N):
					su += pow(global_u[r][j][k][l],P) * pow(local_u[r][j][k][l],Q)
				final_u[i][j][k][l] = ((pow(global_u[i][j][k][l],P) * pow(local_u[i][j][k][l],Q))/su)


# Calculating Partition Coefficient and Partition Entropy
sum_vpc = 0 
sum_vpe = 0

for j in range(Z):
	for k in range(Y):
		for l in range(X):
			for i in range(N):
				sum_vpc += pow(final_u[i][j][k][l],2)
				sum_vpe -= (final_u[i][j][k][l]*ln(final_u[i][j][k][l]))

vpc  = (sum_vpc/(Z*Y*X))
vpe  = (sum_vpe/(Z*Y*X))

# Return a 3-D image matrix with the classifying label.
# 0 BG, 1 CSF, 2 GM 3 WM
classified_img  = full((Z,Y,X),0).tolist()
for j in range(Z):
	for k in range(Y):
		for l in range(X):
			c_group        = 0
			max_membership = 0
			for i in range(N):
				if max_membership < final_u[i][j][k][l]:
					max_membership = final_u[i][j][k][l]
					c_group = i
			classified_img[j][k][l]   = c_group


# Misclassification Error
correct_classified = 0
mis_classified     = 0
test_img_cluster   = {0:[],1:[],2:[],3:[]}
ground_img_cluster = {0:[],1:[],2:[],3:[]}
sum_sa  		   = 0
sum_dsc 		   = 0

# Calculating the Misclassifed voxels
for j in range(Z):
	for k in range(Y):
		for l in range(X):
			if ground_truth[j][k][l] in [0,1,2,3]:
				test_img_cluster[classified_img[j][k][l]].append((j,k,l))
				ground_img_cluster[ground_truth[j][k][l]].append((j,k,l))
				
				if ground_truth[j][k][l] == classified_img[j][k][l]:
					correct_classified += 1
				else:
					mis_classified += 1

for key in test_img_cluster:
    common_points = intersection(test_img_cluster[key],ground_img_cluster[key])
    x = len(common_points)
    y = len(ground_img_cluster[key])

    print(f"Segmentation Accuracy {key} {(x/y)}")
    sum_sa += (x/y)

    dsc_nume = 2*len(common_points)
    dsc_deno = (len(test_img_cluster[key]) + len(ground_img_cluster[key]))
    sum_dsc += (dsc_nume/dsc_deno)

dsc    			 = (sum_dsc/4)
avg_sa 			 = (sum_sa/4)
error_percentage = ((mis_classified/(correct_classified+mis_classified))*100)	



# Print the final results.
print(f"Final Cluster Centers: {C}")
print(f"Total Misclassification Error: {error_percentage}")
print(f"Average Segmentation Accuracy: {avg_sa}")
print(f"Dice Similarity Coefficient: {dsc}")
print(f"Partition Coefficient: {vpc}")
print(f"Partitition Entropy: {vpe}")