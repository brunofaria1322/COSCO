from .Scheduler import *
import numpy as np

class DetectionScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		selectedContainerIDs = []
		for hostID, host in enumerate(self.env.hostlist):
			# FAULT DETECTION
			if host.getCPU() > 80:
				# host with CPU usage above 80%
				containerIDs = self.env.getContainersOfHost(hostID)
				if containerIDs:
					# all the containers in host (Instructions per second)
					containerIPs = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
					# selects the container that consumes more cpu resources (has more instructon per second (IPs))
					selectedContainerIDs.append(containerIDs[np.argmax(containerIPs)])
					
					print(f'HOST {hostID} has cpu usage of {host.getCPU()}: {containerIPs}\t{host.ipsCap}\t{host.getBaseIPS()}')
		
		return selectedContainerIDs

	def placement(self, containerIDs):
		print('-------------place in')
		decision = []
		# List with the cpu usafe of each host
		scores = [(hostID, host.getCPU()) for hostID, host in enumerate(self.env.hostlist)]
		
		for cid in containerIDs:
			# run simple simulation will return energy consumption (we probably can use something that takes into account the CPU or "availability" instead)
			#scores = [self.env.stats.runSimpleSimulation([(cid, hostID)]) for hostID, _ in enumerate(self.env.hostlist)]

			
			print(cid, end='\t')
			print(scores)

			leastFullHost = min(scores, key = lambda t: t[1])

			decision.append((cid, leastFullHost[0]))
			scores.remove(leastFullHost)

		print(decision)
		print('-------------place out')

		return decision