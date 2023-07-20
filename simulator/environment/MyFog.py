import numpy as np
from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *

#Powermodels
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *

class MyFog():
	def __init__(self, num_hosts):
		self.num_hosts = num_hosts
		# based on COSCO paper
		self.types = {
			'B2s':
				{
					'IPS': 4029,
					'RAMSize': 4295,
					'RAMRead': 372.0,
					'RAMWrite': 200.0,
					'Latency': 0.003,
					'DiskSize': 4026,
					'DiskRead': 13.42,
					'DiskWrite': 1.011,
					'BwUp': 1000,
					'BwDown': 1000,
					'Power': 'PMB2s'
				},
			'B4ms':
				{
					'IPS': 8102,
					'RAMSize': 17180,
					'RAMRead': 360.0,
					'RAMWrite': 305.0,
					'Latency': 0.020,
					'DiskSize': 16105,
					'DiskRead': 10.38,
					'DiskWrite': 0.619,
					'BwUp': 2000,
					'BwDown': 2000,
					'Power': 'PMB4ms'
				},
			'B8ms':
				{
					'IPS': 16111,
					'RAMSize': 34360,
					'RAMRead': 376.54,
					'RAMWrite': 266.75,
					'Latency': 0.076,
					'DiskSize': 32212,
					'DiskRead': 11.64,
					'DiskWrite': 1.164,
					'BwUp': 2500,
					'BwDown': 2500,
					'Power': 'PMB8ms'
				}
 		}


	"""
	def generateHosts(self):
		# Linear SFC - old - no pointers to parent or replica
		hosts = []
		keys = ['B2s', 'B4ms', 'B8ms']
		# first half is normal, second half is replica
		for i in range(self.num_hosts):
			typeID = i%3
			key = keys[i%3]
			IPS = self.types[key]['IPS']
			Ram = RAM(self.types[key]['RAMSize'], self.types[key]['RAMRead'], self.types[key]['RAMWrite'])
			Disk_ = Disk(self.types[key]['DiskSize'], self.types[key]['DiskRead']*5, self.types[key]['DiskWrite']*10)
			Bw = Bandwidth(self.types[key]['BwUp'], self.types[key]['BwDown'])
			#Power = eval(self.types[key]['Power']+'()')
			Latency = self.types[key]['Latency']
			hosts.append((IPS, Ram, Disk_, Bw, Latency, typeID))
		return hosts
	"""
	
	
	def generateHosts(self):
		# Linear SFC
        if self.num_hosts != 6:
            raise Exception("MyTreeFog supports only 6 hosts")
	 
		hosts = []
		keys = ['B2s', 'B4ms', 'B8ms']
		# first half is normal, second half is replica
		for i in range(self.num_hosts):
			typeID = i%3
			parent = (i+1)%3 if (i+1)%3 != 0 else None
			replica = i+3 if i < 3 else None
			key = keys[i%3]
			IPS = self.types[key]['IPS']
			Ram = RAM(self.types[key]['RAMSize'], self.types[key]['RAMRead'], self.types[key]['RAMWrite'])
			Disk_ = Disk(self.types[key]['DiskSize'], self.types[key]['DiskRead']*5, self.types[key]['DiskWrite']*10)
			Bw = Bandwidth(self.types[key]['BwUp'], self.types[key]['BwDown'])
			#Power = eval(self.types[key]['Power']+'()')
			Latency = self.types[key]['Latency']
			hosts.append((IPS, Ram, Disk_, Bw, Latency, typeID, parent, replica))
		return hosts