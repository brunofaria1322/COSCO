
class Failure():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, l_type, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID = -1):
		self.id = ID
		self.creationID = creationID
		self.l_type = l_type
		self.ipsmodel = IPSModel
		self.ipsmodel.allocContainer(self)
		self.sla = self.ipsmodel.SLA
		self.rammodel = RAMModel
		self.rammodel.allocContainer(self)
		self.diskmodel = DiskModel
		self.diskmodel.allocContainer(self)
		self.hostid = HostID
		self.env = Environment
		self.createAt = creationInterval
		self.startAt = self.env.interval
		self.totalExecTime = 0
		self.active = True
		self.destroyAt = -1
		self.lastContainerSize = 0

	def getLType(self):
		return self.l_type

	def getBaseIPS(self):
		return self.ipsmodel.getIPS()

	def getApparentIPS(self):
		if self.hostid == -1: return self.ipsmodel.getMaxIPS()
		canUseIPS = self.getHost().getIPSAvailable() / len(self.env.getFailuresOfHost(self.hostid))
		if canUseIPS < 0:
			return 0
		return min(self.ipsmodel.getMaxIPS(), self.getBaseIPS() + canUseIPS)

	def getRAM(self):
		rsize, rread, rwrite = self.rammodel.ram()
		self.lastContainerSize = rsize
		return rsize, rread, rwrite

	def getDisk(self):
		return self.diskmodel.disk()

	def getContainerSize(self):
		if self.lastContainerSize == 0: self.getRAM()
		return self.lastContainerSize

	def getHostID(self):
		return self.hostid

	def getHost(self):
		return self.env.getHostByID(self.hostid)

	def allocate(self, hostID, _):
		# Failures are not migrated
		self.hostid = hostID
		return 0

	def execute(self, _):
		# Migration time is the time to migrate to new host
		# Thus, execution of task takes place for interval
		# time - migration time with apparent ips
		assert self.hostid != -1
		execTime = self.env.intervaltime
		apparentIPS = self.getApparentIPS()
		requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
		self.totalExecTime += min(execTime, requiredExecTime)
		self.ipsmodel.completedInstructions += apparentIPS * self.totalExecTime

	def allocateAndExecute(self, hostID, allocBw):
		self.execute(self.allocate(hostID, allocBw))

	def destroy(self):
		self.destroyAt = self.env.interval
		self.hostid = -1
		self.active = False


