import glob
from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *
import random
import math
import numpy as np
from os import path, makedirs, listdir, remove
import wget
from zipfile import ZipFile
import shutil
import pandas as pd
import warnings
from utils.ColorUtils import color
warnings.simplefilter("ignore")

from tqdm import tqdm


from time import time
import concurrent.futures

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
#ips_multiplier = 2054.0 / (2 * 600)
ips_multiplier = 1

# Found:  0.36363636363636365 40.4040404040404 73 [30, 28, 15]
#ips_multiplier = 1 / 0.20711071107110712
#ram_multiplier = 1 / 62.511251125112516

# Found:  0.4141641641641642 89.13913913913913 78 [31, 35, 12]
#ips_multiplier = 1 / 0.4141641641641642
#ram_multiplier = 1 / 89.13913913913913

def createfiles(df):
	vmids = df[1].unique()[:1000].tolist()
	df = df[df[1].isin(vmids)]
	vmid = 0
	for i in tqdm(range(1, 501), ncols=80):
		trace = []
		bitbraindf = pd.read_csv(f'simulator/workload/datasets/bitbrain/rnd/{i}.csv')
		reqlen = len(bitbraindf)
		while len(trace) < reqlen:
			vmid = (vmid + 1) % len(vmids)
			trace += df[df[1] == vmids[vmid]][4].tolist()
		trace = trace[:reqlen]
		pd.DataFrame(trace).to_csv(f'simulator/workload/datasets/azure_2019/{i}.csv', header=False, index=False)

class MyAzure2019Workload(Workload):
	def __init__(self, numContainers):
		super().__init__()

		#start_time = time()
		#test1()
		#print(f"Time taken: {time() - start_time}")
		#exit()

		#test2()
		#exit()

		## FAILURES
		self.creationFailure_id = 0
		self.createdFailures = []
		self.deployedFailures = []
		## END FAILURES

		self.num = numContainers
		dataset_path = 'simulator/workload/datasets/bitbrain/'
		az_dpath = 'simulator/workload/datasets/azure_2019/'
		possible_path = 'simulator/workload/indices/'
		if not path.exists(dataset_path):
			makedirs(dataset_path)
			print('Downloading Bitbrain Dataset')
			url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
			url_alternate = 'https://www.dropbox.com/s/xk047xqcq9ue5hc/rnd.zip?dl=1'
			try: filename = wget.download(url)
			except: filename = wget.download(url_alternate)
			zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
			for f in listdir(dataset_path+'rnd/2013-9/'): shutil.move(dataset_path+'rnd/2013-9/'+f, dataset_path+'rnd/')
			shutil.rmtree(dataset_path+'rnd/2013-7'); shutil.rmtree(dataset_path+'rnd/2013-8')
			shutil.rmtree(dataset_path+'rnd/2013-9'); remove(filename)
		if not path.exists(az_dpath):
			makedirs(az_dpath)
			print('Downloading Azure 2019 Dataset')
			url = 'https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz'
			filename = wget.download(url)
			df = pd.read_csv(filename, header=None, compression='gzip')
			createfiles(df); remove(filename)
		self.dataset_path = dataset_path
		self.az_dpath = az_dpath
		self.disk_sizes = [1, 2, 3]
		self.meanSLA, self.sigmaSLA = 20, 3
		self.meanSLA, self.sigmaSLA = 4, 1
		self.max_sla = round(self.meanSLA + 4 * self.sigmaSLA) # it cathes 99.9% of the data


		self.possible_indices = [[],[],[]]	# 3 types
		
		if path.exists(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy"):
			self.possible_indices = np.load(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy",allow_pickle=True)

		else:
			cpus_ips = [4029, 8102, 16111]	# from MyFog
			rams = [4295, 17180, 34360]		# from MyFog
			max_containers = [20, 30, 50]	# manually

			all_ips = []
			for i in range(1, 500):
				df = pd.read_csv(self.dataset_path+'rnd/'+str(i)+'.csv', sep=';\t')
				df2 = pd.read_csv(az_dpath+str(i)+'.csv', header=None)

				ips = df['CPU capacity provisioned [MHZ]'].to_numpy()[:self.max_sla] * df2.to_numpy()[:self.max_sla, 0] / 100
				ram = df['Memory usage [KB]'].to_numpy()[:self.max_sla]

				temp_ips = ips_multiplier * np.max(ips) * 0.9
				all_ips.append(temp_ips)
				temp_ram = np.max(ram) / 4000
				
				
				if cpus_ips[0]*0.01 < temp_ips < cpus_ips[0]/max_containers[0] and temp_ram < rams[0]*0.1:
						self.possible_indices[0].append(i)
				if cpus_ips[1]*0.01 < temp_ips < cpus_ips[1]/max_containers[1] and temp_ram < rams[1]*0.1:
						self.possible_indices[1].append(i)
				if cpus_ips[2]*0.01 < temp_ips < cpus_ips[2]/max_containers[2] and temp_ram < rams[2]*0.1:
						self.possible_indices[2].append(i)

				"""
						
				if cpus_ips[0]*0.001 < temp_ips < cpus_ips[0]*0.01 and temp_ram < rams[0]*0.1:
						self.possible_indices[0].append(i)
						self.possible_indices[1].append(i)
						self.possible_indices[2].append(i)

				
						

			# histogram of all ips, x from 0 to n_max
			from matplotlib import pyplot as plt

			n_max = np.ceil(cpus_ips[2]/max_containers[2])
			n_min = np.floor(cpus_ips[0]*0.01)

			plt.figure(figsize=(10,10))

			plt.subplot(2,1,1)
			temp = [i for i in all_ips if n_min<i<n_max]
			plt.hist(temp, bins=1000)

			print(f'count1={len(temp)}')

			plt.subplot(2,1,2)
			temp = [i * 2054.0 / (2 * 600) for i in all_ips if n_min < i * 2054.0 / (2 * 600) < n_max]
			plt.hist(temp, bins=1000)
			plt.savefig("all_ips.png")

			print(f'count2={len(temp)}')

			exit()
				
			"""

			

			np.save(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy", self.possible_indices)


		print(len(self.possible_indices[0]),len(self.possible_indices[1]),len(self.possible_indices[2]))

		#exit()


	def generateNewContainers(self, interval, layer_type = 0):
		#
		# layer_type:	0 - edge
		#				1 - fog
		#				2 - cloud
		#
		workloadlist = []
		# generates 1 container per interval
		#for _ in range(1):

		# poisson arrival with mean 1.5
		for _ in range(max(1,np.random.poisson(self.num))):
			CreationID = self.creation_id
			#index = self.possible_indices[randint(0,len(self.possible_indices)-1)]
			index = random.choice(self.possible_indices[layer_type])
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			df2 = pd.read_csv(self.az_dpath+str(index)+'.csv', header=None)
			sla = round(random.gauss(self.meanSLA, self.sigmaSLA))
			#TODO: ver linha a baixo
			ips = df['CPU capacity provisioned [MHZ]'].to_numpy() * df2.to_numpy()[:, 0] / 100
			IPSModel = IPSMBitbrain((ips_multiplier*ips).tolist(), max((ips_multiplier*ips).tolist()[:max(sla,self.max_sla)]), sla, interval + sla)
			RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/500).to_list(), (df['Network transmitted throughput [KB/s]']/500).to_list())
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/4000).to_list(), (df['Disk write throughput [KB/s]']/12000).to_list())
			workloadlist.append((CreationID, layer_type, interval, IPSModel, RAMModel, DiskModel))
			self.creation_id += 1

			# if layer_type != 0 we want to leave the loop, because we only want to generate 1 container
			if layer_type != 0:
				break

		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()
	
	## FAILURES INJECTION
	def getUndeployedFailures(self):
		undeployed = []
		for i,deployed in enumerate(self.deployedFailures):
			if not deployed:
				undeployed.append(self.createdFailures[i])
		return undeployed

	def updateDeployedFailures(self, creationIDs):
		for cid in creationIDs:
			assert not self.deployedFailures[cid]
			self.deployedFailures[cid] = True

	def generateNewFailures(self, interval, host, failure_type = ['CPU', 'RAM'], max_duration=20):
		#
		# layer_type:	0 - edge
		#				1 - fog
		#				2 - cloud
		#

		failurelist = []
		for _ in range(2):
			CreationID = self.creationFailure_id
			index = random.choice(self.possible_indices[host.layer_type])
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			df2 = pd.read_csv(self.az_dpath+str(index)+'.csv', header=None)
			sla = max_duration
			zeros = [0] * max_duration

			ips = df['CPU capacity provisioned [MHZ]'].to_numpy() * df2.to_numpy()[:, 0] / 100
			if 'CPU' in failure_type:
				IPSModel = IPSMBitbrain((ips_multiplier*ips).tolist(), max((ips_multiplier*ips).tolist()[:max_duration]), sla, interval + sla)
			else:
				IPSModel = IPSMBitbrain(zeros, 0, sla, interval + sla)

			if 'RAM' in failure_type:
				RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/500).to_list(), (df['Network transmitted throughput [KB/s]']/500).to_list())
			else:
				RAMModel = RMBitbrain(zeros, zeros, zeros)
				
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, zeros, zeros)
			failurelist.append((CreationID, host.layer_type, interval, IPSModel, RAMModel, DiskModel))
			self.creationFailure_id += 1

		self.createdFailures += failurelist
		self.deployedFailures += [False] * len(failurelist)
		return self.getUndeployedFailures()
	

def test1():
	# Test 1: Find the best combination of IPS and RAM models the dataset
	# OBJECTIVE: maximize the number of csv files that can be used

	dataset_path = 'simulator/workload/datasets/bitbrain/'
	az_dpath = 'simulator/workload/datasets/azure_2019/'
	
	meanSLA, sigmaSLA = 3, 0.5
	max_sla = math.ceil(meanSLA + 3 *  sigmaSLA)

	# Load data
	file_paths = glob.glob(dataset_path + 'rnd/*.csv')
	df_list = [pd.read_csv(file_path, sep=';\t') for file_path in file_paths]

	az_file_paths = glob.glob(az_dpath + '*.csv')
	df2_list = [pd.read_csv(file_path, header=None) for file_path in az_file_paths]

	# Compute all_rams and all_ips using NumPy operations
	all_rams = np.array([np.median(df['Memory usage [KB]'].to_numpy()[:max_sla]) for df in df_list])
	all_ips = np.array([np.max(df['CPU capacity provisioned [MHZ]'].to_numpy()[:max_sla] * df2.to_numpy()[:max_sla, 0] / 100) for df, df2 in zip(df_list, df2_list)])

	print('min and max ips', np.min(all_ips), np.max(all_ips))
	print('min and max rams', np.min(all_rams), np.max(all_rams))

	# check for all ips and rams
	mins_ram = np.array([400, 1200, 3400])
	maxs_ram = np.array([1200, 3400, 6800])

	mins_ips = np.array([400, 800, 1600])
	maxs_ips = np.array([800, 1600, 3200])

	def compute_indexes(rams, cpus, mul_cpu, mul_ram, min_ram, max_ram, min_cpu, max_cpu):
		indexes = [0,0,0]
		indexes[0] = np.sum(rams[min_ram[0] < rams / mul_ram < max_ram[0]] & cpus[min_cpu[0] < cpus / mul_cpu < max_cpu[0]])
		indexes[1] = np.sum(rams[min_ram[1] < rams / mul_ram < max_ram[1]] & cpus[min_cpu[1] < cpus / mul_cpu < max_cpu[1]])
		indexes[2] = np.sum(rams[min_ram[2] < rams / mul_ram < max_ram[2]] & cpus[min_cpu[2] < cpus / mul_cpu < max_cpu[2]])
		return indexes

	"""
	def compute_indexes(rams, cpus, mul_cpu, mul_ram, min_ram, max_ram, min_cpu, max_cpu):
		indexes = [0,0,0]
		
		for i in range(len(rams)):
			ram_value = rams[i] / mul_ram
			cpu_value = cpus[i] / mul_cpu
			if min_ram[0] < ram_value < max_ram[2] and min_cpu[0] < cpu_value < max_cpu[2]:
				if ram_value < max_ram[0]:
					if cpu_value < max_cpu[0]:
						indexes[0] += 1
				elif ram_value < max_ram[1]:
					if min_cpu[1] <= cpu_value < max_cpu[1]:
						indexes[1] += 1
				else:
					if min_cpu[2] <= cpu_value:
						indexes[2] += 1
		return indexes
	
	"""

	bounds = [(0, 6), (0, 2000)]  # Bounds for mul_cpu and mul_ram
	# Result: Found:  0.36363636363636365 40.4040404040404 73 [30, 28, 15]
	bounds = [(0, 1.25), (0, 650)]  # Bounds for mul_cpu and mul_ram part 2

	# for for linespace of bounds
	num_values = 1000
	matrix = np.zeros((num_values, num_values))
	best_value = 0

	muls_cpu = np.linspace(bounds[0][0], bounds[0][1], num_values)
	muls_ram = np.linspace(bounds[1][0], bounds[1][1], num_values)

	for c, mul_cpu in enumerate(muls_cpu):
		print(c)
		for r, mul_ram in enumerate(muls_ram):
			indexes = compute_indexes(all_rams, all_ips, mul_cpu, mul_ram, mins_ram, maxs_ram, mins_ips, maxs_ips)
			matrix[c][r] = np.sum(indexes)
			if np.sum(indexes) > best_value:
				best_value = np.sum(indexes)
				print("Found: ", mul_cpu, mul_ram, best_value, indexes)
				
	#plot matrix
	import matplotlib.pyplot as plt
	plt.figure(figsize=(10,10))
	plt.imshow(matrix, cmap='hot', interpolation='nearest')
	plt.colorbar()
	plt.xlabel("mul_ram")
	plt.xticks(np.linspace(0, num_values, 5), np.linspace(bounds[1][0], bounds[1][1], 5))
	plt.ylabel("mul_cpu")
	plt.yticks(np.linspace(0, num_values, 5), np.linspace(bounds[0][0], bounds[0][1], 5))
	plt.savefig("matrix.png")


def test2():
	# TEST 2: test distribution of number of containers per node

	from random import gauss
	from numpy.random import poisson
	import matplotlib.pyplot as plt


	num_sim = 100000

	# duration of container, follows a gaussian distribution
	dur_mean = 5
	dur_std = 3

	# arrival of containers, follows a poisson distribution
	arr_mean = 1.5

	containers1 = [[max(1,round(gauss(dur_mean,dur_std))) for _ in range(max(1,poisson(arr_mean)))]]
	containers2 = [[]]
	containers3 = [[]]

	for i in range(1,num_sim):
		containers1.append([c-1 for c in containers1[i-1] if c!=1])
		containers2.append([c-1 for c in containers2[i-1] if c!=1])
		containers3.append([c-1 for c in containers3[i-1] if c!=1])
		
		containers1[i].extend([max(1,round(gauss(dur_mean,dur_std))) for _ in range(max(1,poisson(arr_mean)))])
		containers2[i].extend([max(1,round(gauss(dur_mean,dur_std))) for _ in range(containers1[i-1].count(1))])
		containers3[i].extend([max(1,round(gauss(dur_mean,dur_std))) for _ in range(containers2[i-1].count(1))])
		

	num_containers1 = [len(cs) for cs in containers1]
	num_containers2 = [len(cs) for cs in containers2]
	num_containers3 = [len(cs) for cs in containers3]

	nc1 = {n:num_containers1.count(n) for n in set(num_containers1)}
	print(nc1)
	nc2 = {n:num_containers2.count(n) for n in set(num_containers2)}
	print(nc2)
	nc3 = {n:num_containers3.count(n) for n in set(num_containers3)}
	print(nc3)

	#plot nc
	plt.figure()
	plt.plot(list(nc1.keys()), list(nc1.values()), label="nc1")
	plt.plot(list(nc2.keys()), list(nc2.values()), label="nc2")
	plt.plot(list(nc3.keys()), list(nc3.values()), label="nc3")
	plt.legend()
	plt.savefig("nc.png")
