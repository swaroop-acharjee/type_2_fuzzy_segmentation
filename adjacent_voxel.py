from statistics import (variance,stdev)
from math import (exp)

lst_u = [0.24851338232736886, 0.4983450662578147, 0.4838903706498718, 0.48869666561357983, 0.49922401271013, 0.4924096039244918, 0.4934159792146924, 0.5033863918070528, 0.49234558822094504, 0.4882008674490168, 0.4985113034655582, 0.4874088621442156, 0.48884082116221805, 0.4991382997843284, 0.4955188577847399, 0.5053283131366655, 0.49234558822094515, 0.48966066136692954]
val   = 0.24851338232736886


S     = ((-1)*2*(stdev(lst_u)**2))

lst_y = []	# Primary Fuzzy Function

for x in lst_u:
	temp 		= (x - 0.24851338232736886)**2
	temp_term 	= exp(temp/S)
	print(S,temp_term)

# max_y = max(lst_y)

# sum_numerator   = 0
# sum_denominator = 0
# for x in lst_y:
# 	term = x/max_y
# 	sum_numerator   += (term*x)
# 	sum_denominator += term


# print(sum_numerator/sum_denominator)



