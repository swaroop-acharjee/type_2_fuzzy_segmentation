from global_segmentation import (
	read_pgm,
)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

START = 50
END   = 52
Z 	  = (END-START)			# depth
Y 	  = 217					# height
X     = 181					# col


test_img     = []
ground_truth = []
for j in range(START,END):
    test_img.append(read_pgm(f'Data\\Test Results\\TEST_{j}.pgm'))
    ground_truth.append(read_pgm(f'Data\\Ground Truth\\TEST_{j}.pgm'))

correct_classified = 0
mis_classified     = 0


# Misclassification Error
for j in range(Z):
    for k in range(Y):
        for l in range(X):
            if ground_truth[j][k][l] in [0,1,2,3]:
                if ground_truth[j][k][l] == test_img[j][k][l]:
                    correct_classified += 1
                else:
                    mis_classified += 1

print(correct_classified)
print(mis_classified)
percentage = ((correct_classified/(correct_classified+mis_classified))*100)
print(percentage)



# Segmentation accuracy
test_img_cluster   = {0:[],1:[],2:[],3:[]}
ground_img_cluster = {0:[],1:[],2:[],3:[]}

for j in range(Z):
    for k in range(Y):
        for l in range(X):
            if ground_truth[j][k][l] in [0,1,2,3]:
                test_img_cluster[test_img[j][k][l]].append((j,k,l))
                ground_img_cluster[ground_truth[j][k][l]].append((j,k,l))


sum_segmentation_accuracy = 0
sum_dsc = 0
for key in test_img_cluster:
    common_points = intersection(test_img_cluster[key],ground_img_cluster[key])
    x = len(common_points)
    y = len(ground_img_cluster[key])
    sum_segmentation_accuracy += (x/y)

    dsc_nume = 2*len(common_points)
    dsc_deno = (len(test_img_cluster[key]) + len(ground_img_cluster[key]))
    sum_dsc += (dsc_nume/dsc_deno)

dsc = (sum_dsc/4)
avg_segmentation_accuracy = (sum_segmentation_accuracy/4)

print(avg_segmentation_accuracy)
print(dsc)
