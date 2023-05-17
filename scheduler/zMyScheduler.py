from .Scheduler import *
import numpy as np

class MyScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.result_cache = None

	def selection(self):
		selectedContainerIDs_targetHost = []
		half_hosts_len = int(len(self.env.hostlist)/2)
		# runs only the firs half of hosts (others are replicas)
		for hostID, host in enumerate(self.env.hostlist[:half_hosts_len]):
			# FAULT DETECTION
			if host.getCPU() > 80:
				# host with CPU usage above 80%
				containerIDs = self.env.getContainersOfHost(hostID)
				if containerIDs:
					# all the containers in host (Instructions per second)
					containerIPs = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
					# selects the container that consumes more cpu resources (has more instructon per second (IPs))
					selectedContainerIDs_targetHost.append((containerIDs[np.argmax(containerIPs)], hostID + half_hosts_len))
					
					print(f'HOST {hostID} has cpu usage of {host.getCPU()}: {containerIPs}\t{host.ipsCap}\t{host.getBaseIPS()}')
		
		return selectedContainerIDs_targetHost

	def placement(self, containerIDs):
		print('-------------place in')
		decisions = []
		# List with the cpu usafe of each host
		#scores = [(hostID, host.getCPU()) for hostID, host in enumerate(self.env.hostlist)]
		
		for cid in containerIDs:
			# run simple simulation will return energy consumption (we probably can use something that takes into account the CPU or "availability" instead)
			#scores = [self.env.stats.runSimpleSimulation([(cid, hostID)]) for hostID, _ in enumerate(self.env.hostlist)]

			
			#print(cid, end='\t')
			#print(scores)

			#leastFullHost = min(scores, key = lambda t: t[1])

			#decision.append((cid, leastFullHost[0]))
			#scores.remove(leastFullHost)

			container_ltype = self.env.getContainerByID(cid).getLType()
			print(f"Container with ID {cid} has type {container_ltype}")

			# Will send to the layer type
			decisions.append((cid, container_ltype))

		print(decisions)
		print('-------------place out')

		return decisions