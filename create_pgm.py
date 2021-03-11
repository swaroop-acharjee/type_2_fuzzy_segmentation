import numpy as np


def read_raw_data(file_name:str,ROWS:int,COLS:int,OFFSET=0)->list:
	FILE = open(file_name,mode='r')
	
	# Reading the data in the Single Dimensional form
	img = np.fromfile(FILE, dtype = np.uint8, count = ROWS*COLS, offset=((ROWS*COLS)*OFFSET))
	
	# Shaping the data to the two dimensional format
	img = np.reshape(img,(ROWS,COLS)).tolist()

	return img

	FILE.close()

def create_pgm_file(width:int,height:int,file_name:str,comment:str,img:list,greylevel=255)->None:
	FILE = open(file_name,'wb')
	
	# Defining the PGM Headers
	pgm_header = f"P2\n#{comment}\n{str(width)} {str(height)}\n{str(greylevel)}\n" 
	pgmHeader_byte = bytearray(pgm_header,'utf-8')

	# Writing the PGM Headers into the file
	FILE.write(pgmHeader_byte)

	# Creating the rows of the data
	for row in img:
		row = [str(x) for x in row]
		FILE.write(bytearray(' '.join(row)+"\n",'utf-8'))

	print(f"{file_name} successfully created!!")
	FILE.close()	


if __name__ == "__main__":
	# ROWS = 217
	# COLS = 181

	img = read_raw_data('data.rawb',20,20,8)
	create_pgm_file(20,20,"small_pgm.pgm","test.pgm",img)
	
	# for i in range(181):	
	# 	img = read_raw_data('data.rawb',ROWS,COLS,i)
	# 	create_pgm_file(COLS,ROWS,f"E:\\Notes\\type_2_fuzzy_segmentation\\Image Slices\\TEST_{i}.pgm",f"TEST_{i}.pgm",img)