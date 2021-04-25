from global_segmentation import (
	read_pgm,
)

GT = read_pgm(f'Data\\Ground Truth\\TEST_55.pgm')
IM = read_pgm(f'Data\\Brain Volume\\TEST_55.pgm')

Y 	  = 217			# height
X     = 181		    # col


intensity = {}
for k in range(Y):
    for l in range(X):
        if GT[k][l] not in intensity:
            intensity[GT[k][l]] = [IM[k][l]]
        else:
            intensity[GT[k][l]].append(IM[k][l])

for ele in intensity:
    print(ele," ",sum(intensity[ele])/len(intensity[ele]))

