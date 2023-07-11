import random
from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *


class MyFog:
    def __init__(self, num_hosts):
        ips_multiplier = 2048
        gib_multiplier = 1024
        self.num_hosts = num_hosts
        # based on COSCO paper
        self.types = {
            "edge": {  # B1ms - B2s
                "IPS": [1 * ips_multiplier, 2 * ips_multiplier],
                "RAMSize": [2 * gib_multiplier, 4 * gib_multiplier],
                "RAMRead": 372.0,
                "RAMWrite": 200.0,
                "Latency": 0.003,
                "DiskSize": [4 * gib_multiplier, 8 * gib_multiplier],
                "DiskRead": 13.42,
                "DiskWrite": 1.011,
                "BwUp": 1000,
                "BwDown": 1000,
            },
            "fog": {  # B8ms - B12ms
                "IPS": [8 * ips_multiplier, 12 * ips_multiplier],
                "RAMSize": [32 * gib_multiplier, 48 * gib_multiplier],
                "RAMRead": 360.0,
                "RAMWrite": 305.0,
                "Latency": 0.020,
                "DiskSize": [64 * gib_multiplier, 96 * gib_multiplier],
                "DiskRead": 10.38,
                "DiskWrite": 0.619,
                "BwUp": 2000,
                "BwDown": 2000,
            },
            "cloud": {  # B20ms
                "IPS": 20 * ips_multiplier,
                "RAMSize": 80 * gib_multiplier,
                "RAMRead": 376.54,
                "RAMWrite": 266.75,
                "Latency": 0.076,
                "DiskSize": 160 * gib_multiplier,
                "DiskRead": 11.64,
                "DiskWrite": 1.164,
                "BwUp": 2500,
                "BwDown": 2500,
            },
        }

    def generateHosts(self):
        # Linear SFC
        hosts = []
        keys = list(self.types)
        
        # first half is normal, second half is replica
        for i in range(self.num_hosts):
            typeID = i % 3
            parent = (i + 1) % 3 if (i + 1) % 3 != 0 else None
            replica = i + 3 if i < 3 else None
            key = keys[i % 3]
            if typeID == 2:  # cloud
                IPS = self.types[key]["IPS"]
                Ram = RAM(
                    self.types[key]["RAMSize"],
                    self.types[key]["RAMRead"],
                    self.types[key]["RAMWrite"],
                )
                Disk_ = Disk(
                    self.types[key]["DiskSize"],
                    self.types[key]["DiskRead"],
                    self.types[key]["DiskWrite"],
                )
            else:
                IPS = random.randint(
                    self.types[key]["IPS"][0], self.types[key]["IPS"][1]
                )
                Ram = RAM(
                    random.randint(self.types[key]["RAMSize"][0], self.types[key]["RAMSize"][1]),
                    self.types[key]["RAMRead"],
                    self.types[key]["RAMWrite"],
                )
                Disk_ = Disk(
                    random.randint(self.types[key]["DiskSize"][0], self.types[key]["DiskSize"][1]),
                    self.types[key]["DiskRead"] * 5,
                    self.types[key]["DiskWrite"] * 10,
                )
            Bw = Bandwidth(self.types[key]["BwUp"], self.types[key]["BwDown"])

            # Power = eval(self.types[key]['Power']+'()')
            Latency = self.types[key]["Latency"]
            hosts.append((IPS, Ram, Disk_, Bw, Latency, typeID, parent, replica))
        return hosts
