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
)

from math import (
	log,
	exp,
)

from statistics import (
		variance,
	)

START = 50
END   = 53
Z 	  = (END-START)			# depth
Y 	  = 217					# height
X     = 181					# col
N_IT  = 100

C     = [13,45,100,115]	   	# Values of the initial Clusters
A     = 0.7					# Value of alpha
N     = len(C)
P     = 1
Q     = 3
M     = 2.5
E     = 0.01
p 	  = (1/(M-1))


img_vol 	 = []
ground_truth = []
for j in range(START,END):
    img_vol.append(read_pgm(f'Data\\Brain Volume\\TEST_{j}.pgm'))
    ground_truth.append(read_pgm(f'Data\\Ground Truth\\TEST_{j}.pgm'))

print("Z = ",len(img_vol))
print("Y = ",len(img_vol[0]))
print("X = ",len(img_vol[0][0]))


global_u = full((len(C),Z,Y,X),0).tolist()
local_u  = full((len(C),Z,Y,X),0).tolist()
final_u  = full((len(C),Z,Y,X),0).tolist()

for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				temp = 0
				numerator = (img_vol[j][k][l] + 0.001 - C[i])**2
				for r in range(N):
					denominator =  (img_vol[j][k][l] + 0.001 - C[r])**2
					temp        += ((numerator/denominator)**p)
				local_u[i][j][k][l]  = (1/temp)	
				global_u[i][j][k][l] = (1/temp)


for i in range(N):
	global_new_u 		= full((Z,Y,X),0).tolist()
	local_new_u  		= full((Z,Y,X),0).tolist()
	type_2_u     		= full((Z,Y,X),0).tolist()
	type_2_u_normalized = full((Z,Y,X),0).tolist()
		
			for j in range(Z):
				for k in range(Y):
					for l in range(X):
						coordinates   = find_neighbor(j,k,l,Z,Y,X)

						distance   	  = (C[i] - img_vol[j][k][l])**2
						ln_value_1    = log(global_u[i][j][k][l]**M)
						numerator_1   = abs(distance - ln_value_1 - 1)

						
						f             = likelihood(img_vol,local_u[i],coordinates)
						d             = mean_distance(img_vol,coordinates,C[i])
						ln_value_2    = log(local_u[i][j][k][l]**M)
						numerator_2   = abs(f*d - ln_value_2 - 1)


						temp_1 = 0
						temp_2 = 0

						for r in range(N):
							distance_r_1  = (C[r] - img_vol[j][k][l])**2
							ln_value_r_1  = log(global_u[r][j][k][l]**M)
							denominator_1 = abs(distance_r_1 - ln_value_r_1 - 1)

							f_r_2         = likelihood(img_vol,local_u[r],coordinates)
							d_r_2         = mean_distance(img_vol,coordinates,C[r])
							ln_value_r_2  = log(local_u[r][j][k][l]**M)
							denominator_2 = abs(f_r_2*d_r_2 - ln_value_r_2 - 1)

							temp_1 += ((numerator_1/denominator_1)**p) 
							temp_2 += ((numerator_2/denominator_2)**p)


						global_new_u[i][j][k][l] = (1/temp_1)
						local_new_u[i][j][k][l]  = (1/temp_2)


		for i in range(N):
			for j in range(Z):
				for k in range(Y):
					for l in range(X):
						coordinates  = find_neighbor(j,k,l,Z,Y,X)

						list_local_u = [local_new_u[i][j][k][l],]

						for x_j,x_k,x_l in coordinates:
							list_local_u.append(local_new_u[i][x_j][x_k][x_l])

						S = ((-2)*variance(list_local_u))

						primary_fuzzy_set = []
						for x in list_local_u:
							temp 		= (x-local_new_u[i][j][k][l])**2
							temp_term 	= exp(temp/S) 
							primary_fuzzy_set.append(temp_term)

						max_y = max(primary_fuzzy_set)
						sum_numerator   = 0
						sum_denominator = 0
						for x in primary_fuzzy_set:
							term 			 = x/max_y
							sum_numerator   += (term*x)
							sum_denominator += term
						type_2_u[i][j][k][l] = (sum_numerator/sum_denominator) 

		for i in range(N):
			for j in range(Z):
				for k in range(Y):
					for l in range(X):
						su = 0
						for r in range(N):
							su += type_2_u[r][j][k][l]
						type_2_u_normalized[i][j][k][l] = type_2_u[i][j][k][l]/su



		local_new_u = type_2_u_normalized[::]

		new_C 		= [0,0,0,0]

		for i in range(N):	
			numerator   = 0
			denominator = 0
			for j in range(Z):
				for k in range(Y):
					for l in range(X):
						# coordinates  = find_neighbor(j,k,l,Z,Y,X)
						
						# f            = likelihood(img_vol,local_new_u[i],coordinates)
						# a_mean       = mean_pixels(img_vol,coordinates)

						n_a = (A*(global_new_u[i][j][k][l]**M))
						n_b = ((1-A)*(local_new_u[i][j][k][l]**M))

						numerator   += n_a*img_vol[j][k][l] + n_b*img_vol[j][k][l]
						denominator += n_a + n_b
						
			new_C[i] = round(numerator/denominator,2)

		e_0 = abs(C[0] - new_C[0])
		e_1 = abs(C[1] - new_C[1])
		e_2 = abs(C[2] - new_C[2])
		e_3 = abs(C[3] - new_C[3])
		avg_error = ((e_0 + e_1 + e_2 + e_3)/4)
		

		C 		 = new_C[::]
		global_u = global_new_u[::]
		local_u  = local_new_u[::]

		print()
		print((_+1))
		print(e_0,"   ",e_1,"   ",e_2,"   ",e_3,"   ")
		print(C)
		print(avg_error)
		print()


		if avg_error < E:
			break




# Normalization of the membership functions
for i in range(N):
	for j in range(Z):
		for k in range(Y):
			for l in range(X):
				su = 0
				for r in range(N):
					su += (global_u[r][j][k][l]**P)*(local_u[r][j][k][l]**Q)
				final_u[i][j][k][l] = (((global_u[i][j][k][l]**P)*(local_u[i][j][k][l]**Q))/su)



sum_vpc = 0
sum_vpe = 0
classified_img  = full((Z,Y,X),0).tolist()

# Return a 3-D list with the classifying label.
for j in range(Z):
	for k in range(Y):
		for l in range(X):
			max_member = []
			for i in range(len(C)):
				max_member.append((final_u[i][j][k][l],i))
				sum_vpc += final_u[i][j][k][l]**2
				sum_vpe -= final_u[i][j][k][l]*log(final_u[i][j][k][l])

			classified_img[j][k][l]   = max(max_member)[1]

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
vpc 			 = sum_vpc/(Z*Y*X)
vpe 			 = sum_vpe/(Z*Y*X)	



print(f"Final Cluster Centers: {C}")
print(f"Total Misclassification Error: {error_percentage}")
print(f"Average Segmentation Accuracy: {avg_sa}")
print(f"Dice Similarity Coefficient: {dsc}")
print(f"Partition Coefficient: {vpc}")
print(f"Partitition Entropy: {vpe}")