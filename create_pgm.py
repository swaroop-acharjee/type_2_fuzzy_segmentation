import numpy as np


def read_raw_data(file_name:str,ROWS:int,COLS:int,OFFSET=0) -> list:
	'''
	This function reads a rawb file and converts it into a 2-D matrix
	'''

	FILE = open(file_name,mode='r')
	
	# Reading the data in the Single Dimensional form
	img = np.fromfile(FILE, dtype = np.uint8, count = ROWS*COLS, offset=((ROWS*COLS)*OFFSET))
	
	# Shaping the data to the two dimensional format
	img = np.reshape(img,(ROWS,COLS)).tolist()

	FILE.close()
	return img

	

def create_pgm_file(width:int,height:int,file_name:str,comment:str,img:list,greylevel=255)->None:
	'''
	This function takes a 2-D matrix
	and converts into a pgm file.
	'''
	
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
	ROWS 			  = 217
	COLS 			  = 181
	
	path_to_raw_data  = 'Data\\Raw Files\\'+'t1_icbm_normal_1mm_pn9_rf40.rawb'
	path_to_save_file = 'Data\\Brain Volume\\'
	# Creating the brain volume
	for i in range(181):	
		img = read_raw_data(path_to_raw_data,ROWS,COLS,i)
		create_pgm_file(COLS,ROWS,path_to_save_file+f"TEST_{i+1}.pgm",f"TEST_{i+1}.pgm",img)
	

	path_to_raw_data  = 'Data\\Raw Files\\'+'phantom_1.0mm_normal_crisp.rawb'
	path_to_save_file = 'Data\\Groud Truth\\'
	# Create the discrete model
	for i in range(181):	
		img = read_raw_data(path_to_raw_data,ROWS,COLS,i)
		create_pgm_file(COLS,ROWS,path_to_save_file+f"TEST_{i+1}.pgm",f"TEST_{i+1}.pgm",img,9)