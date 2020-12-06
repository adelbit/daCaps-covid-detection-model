#Libraries Used
import pandas as pd
import numpy as np
import os
from skimage import transform
import matplotlib.pyplot as plt

#Resizeing Images

#Replacement of multiple characters to use in the createPostName function
def replaceMul(mainString, toBeReplaces, newString):
    #Iterate over the strings to be replaced
    for elem in toBeReplaces :
        #Check if string is in the main string
        if elem in mainString :
            #Replace the string
            mainString = mainString.replace(elem, newString)
    
    return  mainString 

#Preprocessing Data   
def preprocessData(n_to_process=-1, img_shape=(224,224)):

	os.makedirs(f'./database_preprocessed/', exist_ok=True)

	train_data = pd.read_csv(r'./dataset//train.txt', header=None, index_col=None)[0].str.split(' ', 1)
	val_data   = pd.read_csv(r'./dataset/validate.txt', header=None, index_col=None)[0].str.split(' ', 1)
	test_data  = pd.read_csv(r'./dataset/test.txt', header=None, index_col=None)[0].str.split(' ', 1)

	# number of samples to process
	train_data = train_data if (n_to_process==-1 or n_to_process>len(train_data)) else train_data[:n_to_process]
	validate_data   = validate_data if (n_to_process ==-1 or n_to_process>len(validate_data)) else validate_data[:n_to_process]
	test_data  = test_data if (n_to_process ==-1 or n_to_process>len(test_data)) else test_data[:n_to_process]


	train_paths = train_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	validate_paths   = validate_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	test_paths  = test_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	all_paths   = np.hstack((train_paths, validate_paths, test_paths))

	i=0
	for img_path in all_paths:
		i += 1
		if  i % max(1, int(len(all_paths)/1000))==0: print(i, '/', len(all_paths))
		new_path = img_path.replace('database', 'database_preprocessed'); new_path = replaceMul(new_path,['images_001/', 'images_002/', 'images_003/','images_004/', 'images_005/','images_006/', 'images_007/','images_008/', 'images_009/', 'images_010/', 'images_011/', 'images_012/'],'')
		img = plt.imread(img_path)
		img = transform.resize(img, img_shape, anti_aliasing=True)
		plt.imsave(fname=new_path, arr=img, cmap='gray')

preprocessData(n_to_process=-1, img_shape=(128,128))
