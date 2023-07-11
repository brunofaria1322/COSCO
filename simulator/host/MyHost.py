from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *


class MyHost:
    # IPS = Million Instructions per second capacity
    # RAM = Ram in MB capacity
    # Disk = Disk characteristics capacity
    # Bw = Bandwidth characteristics capacity
    # Latency = Latency of host
    # Environment = Environment object
    # layer_type = 0 = edge, 1 = fog, 2 = cloud
    # parentID = parent host ID
    # replicaID = replica host ID
    def __init__(
        self,
        ID,
        IPS,
        RAM,
        Disk,
        Bw,
        Latency,
        Environment,
        layer_type,
        parentID,
        replicaID=None,
    ):
        self.id = ID
        self.ipsCap = IPS
        self.ramCap = RAM
        self.diskCap = Disk
        self.bwCap = Bw
        self.latency = Latency
        self.env = Environment
        self.layer_type = layer_type  # 0 = edge, 1 = fog, 2 = cloud
        self.parentID = parentID  # parentID host ID
        self.replicaID = replicaID  # replica host ID

        # print(f'HOST {ID} with layer Type {layer_type}, parent {parentID}, replica {replicaID} created')
        # print(f'\tIPS: {IPS}, RAM: {RAM.size}, Disk: {Disk.size}, Latency: {Latency}')

    def getCPU(self):
        ips = self.getApparentIPS()
        return 100 * (ips / self.ipsCap)

    def getBaseIPS(self):
        # Get base ips count as sum of min ips of all containers
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).getBaseIPS()
        # assert ips <= self.ipsCap
        return ips

    def getApparentIPS(self):
        # Give containers remaining IPS for faster execution
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).getApparentIPS()
        # assert int(ips) <= self.ipsCap

        # FAILURES
        failures = self.env.getFailuresOfHost(self.id)
        for failureID in failures:
            ips += self.env.getFailureByID(failureID).getApparentIPS()
        return min(self.ipsCap, int(ips))

    def getIPSAvailable(self):
        # IPS available is ipsCap - baseIPS
        # When containers allocated, existing ips can be allocated to
        # the containers
        return self.ipsCap - self.getBaseIPS()

    def getRAM(self):
        size, _, _ = self.getCurrentRAM()
        return 100 * (size / self.ramCap.size)

    def getCurrentRAM(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getRAM()
            size += s
            read += r
            write += w
        # assert size <= self.ramCap.size
        # assert read <= self.ramCap.read
        # assert write <= self.ramCap.write
        return size, read, write

    def getRAMAvailable(self):
        size, read, write = self.getCurrentRAM()
        return (
            self.ramCap.size - size,
            self.ramCap.read - read,
            self.ramCap.write - write,
        )

    def getCurrentDisk(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getDisk()
            size += s
            read += r
            write += w
        # assert size <= self.diskCap.size
        # assert read <= self.diskCap.read
        # assert write <= self.diskCap.write
        return size, read, write

    def getDiskAvailable(self):
        size, read, write = self.getCurrentDisk()
        return (
            self.diskCap.size - size,
            self.diskCap.read - read,
            self.diskCap.write - write,
        )

    def __str__(self):
        return f"Host {self.id} with layer type {self.layer_type}, parent {self.parentID}, replica {self.replicaID}"
