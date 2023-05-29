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

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)

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
		self.meanSLA, self.sigmaSLA = 3, 0.5
		self.max_sla = math.ceil(self.meanSLA + 3 *  self.sigmaSLA)


		self.possible_indices = [[],[],[]]	# 3 types

		if path.exists(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy"):
			self.possible_indices = np.load(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy",allow_pickle=True)

		else:
			for i in range(1, 500):
				df = pd.read_csv(self.dataset_path+'rnd/'+str(i)+'.csv', sep=';\t')
				df2 = pd.read_csv(az_dpath+str(i)+'.csv', header=None)

				ips = df['CPU capacity provisioned [MHZ]'].to_numpy()[:self.max_sla] * df2.to_numpy()[:self.max_sla, 0] / 100
				temp = ips_multiplier * max(ips)

				if 400 < temp < 3200:
					if temp < 800:
						self.possible_indices[0].append(i)
					elif temp < 1600:
						self.possible_indices[1].append(i)
					elif temp < 3200:
						self.possible_indices[2].append(i)

			np.save(possible_path + f"{self.meanSLA}-{self.sigmaSLA}.npy", self.possible_indices)

		#print(len(self.possible_indices[0]),len(self.possible_indices[1]),len(self.possible_indices[2]))		

	def generateNewContainers(self, interval, layer_type = 0):
		#
		# layer_type:	0 - edge
		#				1 - fog
		#				2 - cloud
		#
		workloadlist = []
		#for i in range(max(1,int(gauss(self.mean, self.sigma)))):
		# generates 1 container per interval
		for _ in range(1):
			CreationID = self.creation_id
			#index = self.possible_indices[randint(0,len(self.possible_indices)-1)]
			index = random.choice(self.possible_indices[layer_type])
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			df2 = pd.read_csv(self.az_dpath+str(index)+'.csv', header=None)
			sla = random.gauss(self.meanSLA, self.sigmaSLA)
			#TODO: ver linha a baixo
			ips = df['CPU capacity provisioned [MHZ]'].to_numpy() * df2.to_numpy()[:, 0] / 100
			IPSModel = IPSMBitbrain((ips_multiplier*ips).tolist(), max((ips_multiplier*ips).tolist()[:self.max_sla]), int(1.2*sla), interval + sla)
			RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/1000).to_list(), (df['Network transmitted throughput [KB/s]']/1000).to_list())
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/4000).to_list(), (df['Disk write throughput [KB/s]']/12000).to_list())
			workloadlist.append((CreationID, layer_type, interval, IPSModel, RAMModel, DiskModel))
			self.creation_id += 1
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

	def generateNewFailures(self, interval, host, max_duration=20):
		#
		# layer_type:	0 - edge
		#				1 - fog
		#				2 - cloud
		#

		failurelist = []
		for _ in range(1):
			CreationID = self.creationFailure_id
			index = random.choice(self.possible_indices[host.layer_type])
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			df2 = pd.read_csv(self.az_dpath+str(index)+'.csv', header=None)
			sla = max_duration
			zeros = [0] * max_duration

			ips = df['CPU capacity provisioned [MHZ]'].to_numpy() * df2.to_numpy()[:, 0] / 100
			IPSModel = IPSMBitbrain((ips_multiplier*ips).tolist(), max((ips_multiplier*ips).tolist()[:self.max_sla]), int(1.2*sla), interval + sla)
			RAMModel = RMBitbrain(zeros, zeros, zeros)
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, zeros, zeros)
			failurelist.append((CreationID, host.layer_type, interval, IPSModel, RAMModel, DiskModel))
			self.creationFailure_id += 1

		self.createdFailures += failurelist
		self.deployedFailures += [False] * len(failurelist)
		return self.getUndeployedFailures()