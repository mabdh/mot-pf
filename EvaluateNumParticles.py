#!/usr/bin/python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os.path as ph
import os
import sys
from bokeh.plotting import figure, HBox, output_file, show, VBox
from bokeh.models import Range1d

pathDir = "Evaluate"
dirname = os.listdir(pathDir)

result = [np.zeros(7) for j in range(0,10)]
# result = np.array(result)
# print result

for x in xrange(1,len(dirname)):
	filename = pathDir + "/" + dirname[x] + "/Evalfile.txt"
	with open(filename) as f:
		data = f.read()
	data = data.split('\n')
	data = [row.split(' ') for row in data]
	data = np.array(data)
	data = np.delete(data,len(data)-1,0)
	for i in range(0,10):
		result[i] = np.vstack((result[i], data[i]))
result = [np.delete(result[i],0,axis=0) for i in range (0,len(result))]
result = np.array(result, dtype=np.float32)

result = np.delete(result,(1,2,3,4),axis=2)

# print result[0]
data_mean_avg = [np.mean(mat, axis=0) for mat in result]
data_mean_avg = np.array(data_mean_avg)
data_std = [np.std(mat, axis=0) for mat in result]
data_std = np.array(data_std)
data_mean_avg = np.transpose(data_mean_avg)
data_std = np.transpose(data_std)

plt.figure(1)
plt.xlabel('Number of particles')
plt.ylabel('MOTA')
plt.errorbar(data_mean_avg[0], data_mean_avg[1],yerr=data_std[1],ecolor='r')
plt.plot(data_mean_avg[0], data_mean_avg[1],color='b')
plt.xscale('symlog')
plt.xticks(np.arange(min(data_mean_avg[0]), max(data_mean_avg[0]) + 10, 10.0))


plt.grid(True)

plt.figure(2)
plt.xlabel('Number of particles')
plt.ylabel('MOTP')
plt.errorbar(data_mean_avg[0], data_mean_avg[2],yerr=data_std[2],ecolor='r')
plt.plot(data_mean_avg[0], data_mean_avg[2],color='g')
plt.xscale('symlog')
plt.xticks(np.arange(min(data_mean_avg[0]), max(data_mean_avg[0]) + 10, 10.0))
plt.grid(True)
print data_std[1]
print data_std[2]
plt.show()
