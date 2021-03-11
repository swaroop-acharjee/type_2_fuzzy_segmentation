import matplotlib.pyplot as plt
import numpy as np
import math

def read_pgm(file_path:str)->list:
	'''
	Function to read the data 
	from a pgm file and return 
	a 2-D matrix of the pixel intensity
	values.
	'''
	with open(file_path,mode='rb') as FILE:

		records = [row.decode('utf-8') for row in FILE.readlines()]

		img = []
		for i in records[4:]:
			temp = [int(x) for x in i.split()] 
			img.append(temp)

		return img


def initial_cluster_center(img:list):
	'''
	Function to calculate the initial
	cluster center and return a tuple
	containing 4 elements. 
	1st Element: Intensity of Background
	2nd Element: Intensity of Gray matter
	3rd Element: Intensity of White Matter
	4th Element: Intensity of CSF
	'''
	freq_hist = dict()

	for row in img:
		for col in row:
			if col not in freq_hist:
				freq_hist[col]  = 1
			else:
				freq_hist[col] += 1

	
	a = list(freq_hist.keys())
	b = list(freq_hist.values())	

	c = sorted(list(zip(a,b)))

	freq_hist = dict(c)

	for intensity in freq_hist:
		print(intensity,freq_hist[intensity])

	return 5,12,79,200


# def cal_global_memberships(m,img,c,u):
# 	a = 0.3
# 	d = dict()
# 	for i in range(len(c)):
# 		d[i] = []
# 		for k in range(len(img)):
# 			temp = []
# 			for l in range(len(img[0])):
# 				print(c[i])
# 				distance    = math.pow((c[i] - img[k][l]),2)
# 				ln_value    = math.log(math.pow(u[i][k][l],m))
# 				final_value = a*(distance - ln_value - 1) 
# 				temp.append(final_value)
# 			d[i].append(temp)


	
# 	pixel_deno = []
# 	for k in range(len(img)):
# 		temp = []
# 		for l in range(len(img[0])):
# 			s = 0
# 			for i in range(len(c)):
# 				s += d[i][k][l]
# 			temp.append(s)
# 		pixel_deno.append(temp)


# 	new_u = dict()
# 	for i in range(2):
# 		temp = np.full((20,20),0).tolist()
# 		new_u[i] = temp

# 	p = (1/(m-1))
# 	for i in range(len(c)):
# 		for k in range(len(img)):
# 			for l in range(len(img[0])):
# 				n = d[i][k][l]
# 				new_u[i][k][l] = (1/((n/(a*pixel_deno[k][l]))**p)) 
	

# 	# Calculating the cluster centers
# 	N = [0,0]
# 	D = [0,0]
# 	new_clusters = [0,0]

# 	for i in range(len(c)):
# 		for k in range(len(img)):
# 			for l in range(len(img[0])):
# 				N[i] += new_u[i][k][l]**m*img[k][l]
# 				D[i] += new_u[i][k][l]**m
# 		new_clusters[i] = (N[i]/D[i])


# 	return new_clusters,new_u



# if __name__ == '__main__':
# 	file_path = 'small_pgm.pgm'
	
# 	img 	  = read_pgm(file_path)
	

# 	# Assuming the intensity of the cluster centers
# 	c = (5,70)
# 	m = 2.5
# 	e = 0.1

# 	# Initializing the value of the
# 	# membership function
# 	# distance matrix

# 	u = dict()
# 	for i in range(2):
# 		temp = np.full((20,20),0.5).tolist()
# 		u[i] = temp

# 	for _ in range(5):
# 		x = cal_global_memberships(m,img,c,u)
# 		print(x[0])

# 		c = x[0]
# 		u = x[1]



	