import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def get_point(filename):
	with open(filename) as file:
				lines = file.read().splitlines()
	points_number = int(lines[0])
	points = np.ones((points_number, 2))

	for i in range(points_number):
		y,x = lines[i+1].split()
		points[i,0] = x 
		points[i,1] = y
	return points


def visualization(X,Y,Z,feature):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	ax.scatter(X, Y, Z)
	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

	mid_x = (X.max()+X.min()) * 0.5
	mid_y = (Y.max()+Y.min()) * 0.5
	mid_z = (Z.max()+Z.min()) * 0.5

	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	ax.set_title('Factorization Method with '+ feature)

	plt.show()