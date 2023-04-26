import random
import matplotlib.pyplot as plt 

import pandas as pd
import warnings
from tqdm import tqdm

if __name__ == "__main__":
	# test
	
	dataset_path = 'simulator/workload/datasets/bitbrain/'
	az_dpath = 'simulator/workload/datasets/azure_2019/'
	
	max_sla = 30
	nums0, nums1, nums2 = [],[],[]

	max_df2 = 0

	for index in range(1, 500):
		df = pd.read_csv(dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
		df2 = pd.read_csv(az_dpath+str(index)+'.csv', header=None)

		if max(df2.to_numpy()[:max_sla,0]) > max_df2:
			max_df2 = max(df2.to_numpy()[:max_sla,0])
			print(max_df2)

		ips = df['CPU capacity provisioned [MHZ]'].to_numpy()[:max_sla] * df2.to_numpy()[:max_sla, 0] / 100
		temp = max(ips)

		if temp < 400 or temp > 6000:
			continue
		elif temp < 800:
			nums0.append(temp)
		elif temp < 1600:
			nums1.append(temp)
		elif temp < 3200:
			nums2.append(temp)

	plt.hist(nums0, bins=200)
	plt.savefig("test0.png")

	
	plt.hist(nums1, bins=200)
	plt.savefig("test1.png")

	
	plt.hist(nums2, bins=200)
	plt.savefig("test2.png")

	
	print(len(nums0),len(nums1),len(nums2))
