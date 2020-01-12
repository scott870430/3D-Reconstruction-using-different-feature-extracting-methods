import os
import glob

import cv2
import numpy as np
import shutil

from Feature_Matching import process
from function import *

if __name__ == '__main__':

	feature = ['KLT','SIFT','SURF','ORB']
	t = 1
	print("capture feature: ",feature[t])

	#choose what you want to do
	get_new_feature = True
	show_feature = True
	feactorization = True
	#your image path 
	img_names = glob.glob('images/*.jpg')
	img_names.sort(key = len)

	#get featrue from algorithm which you choose
	if get_new_feature:
		print("----get new feature from ",feature[t],"\n")
		img_names = glob.glob('images/*.jpg')

		dirName = feature[t]+'_Features'
		if  os.path.exists(dirName):
			shutil.rmtree(dirName)

		# load image
		imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in img_names]   #your image
		results_array = process(imgs[:], feature=feature[t] , scale = 0.5 , manual = False) #get feature

		os.system("pause")

	#mark these features on the picture
	if show_feature:
		print("----show feature from ",feature[t],"\n")
		fpt = glob.glob(feature[t]+'_Features'+'/*.txt')

		fpt.sort(key = len)

		img_count = 1
		for img,pts in  zip(img_names,fpt):
			image = cv2.imread(img)
			gray = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
			points = get_point(pts)

			for e in points:
				cv2.circle(gray,(int(e[1]), int(e[0])),10, (0, 0, 255), 1)
			cv2.imwrite(feature[t]+'_Features'+'/'+'revise'+str(img_count)+".jpg",gray)

			img_count += 1

	#3D reconstruction
	if feactorization:
		print("do 3D reconstruction with feactorization\n")
		feature_files = glob.glob(feature[t]+'_Features'+'/*.txt')


		U = np.zeros(1)
		V = np.zeros(1)
		for e in feature_files:
			points = get_point(str(e))
			p_t = points - np.mean(points, axis=0)

			if U.all() == np.zeros(1):
				U = p_t[:,0:1].reshape(1,-1)
				V = p_t[:,1:2].reshape(1,-1)
			else :
				tempU = p_t[:,0:1].reshape(1,-1)
				tempV = p_t[:,1:2].reshape(1,-1)
				U = np.concatenate((U,tempU))
				V = np.concatenate((V,tempV))


		print("feature number: ",len(U[0]))
		W = np.concatenate((U,V))

		u, s, vh = np.linalg.svd(W)
		u = u[:, 0: 3]
		vh = vh[0: 3, :]
		s = np.diag(s[0: 3])

		motion = np.dot(u, np.sqrt(s))
		structure = np.dot(np.sqrt(s), vh)

		X,Y,Z = structure[0,:], structure[1,:], structure[2,:]
		dirName = feature[t] + '_result'


		if not os.path.exists(dirName):
			os.mkdir(dirName)

			print("Directory " , dirName ,  " Created ")

		with open(dirName+'/'+feature[t]+'result.txt', 'w+') as output_file:
			for x,y,z in zip(X,Y,Z):
				print(x,y,z,file = output_file)
		
		#Plot the structure
		visualization(X,Y,Z,feature[t])
