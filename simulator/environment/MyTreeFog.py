from matplotlib import patches
from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

    def plotTree(self, hosts):
        """
        Plot the tree of hosts

        :param hosts: hosts to plot
        """

        G = nx.DiGraph()

        # add nodes
        for i, host in enumerate(hosts):
            G.add_node(i, host=host)

        # add edges
        for i, host in enumerate(hosts):
            parentID = host[6]
            replicaID = host[7]
            if parentID is not None:
                G.add_edge(i, parentID)
            if replicaID is not None:
                G.add_edge(i, replicaID)

        # Set node positions for each level
        # hosts with same type are placed on the same level

        pos = {}

        # first display the cloud and replica on the right

        # cloud
        pos[0] = (-1, 0)
        # replica
        pos[1] = (1, 0)

        n_fog = np.sum([host[5] == 1 for host in hosts])
        n_edge = np.sum([host[5] == 0 for host in hosts])

        # Edge hosts (they are the ones with the highest number of nodes)
        prev_fog = hosts[2 + n_fog][6]
        prev_fog_count = 0
        prev_fog_start = -n_edge
        for e in range(n_edge):
            fog = hosts[e + 2 + n_fog][6]

            if fog != prev_fog:
                print(prev_fog, fog, end="\t")
                print(prev_fog_start, -n_edge + 1 + e * 2)
                pos[prev_fog] = (prev_fog_start + 1 + prev_fog_count - 2, -1)
                pos[prev_fog + 1] = (prev_fog_start + 1 + prev_fog_count, -1)
                prev_fog = fog
                prev_fog_count = 0
                prev_fog_start = -n_edge + 1 + e * 2
            else:
                prev_fog_count += 1

            pos[e + 2 + n_fog] = (-n_edge + 1 + e * 2, -2)

        # last fog
        pos[prev_fog] = (prev_fog_start + 1 + prev_fog_count - 2, -1)
        pos[prev_fog + 1] = (prev_fog_start + 1 + prev_fog_count, -1)

        # Draw the graph
        plt.figure(figsize=(int(0.9 * n_edge), 4))
        node_labels = {i: f"{i}" for i in G.nodes}
        colors = ["lightblue", "lightgreen"]
        node_colors = [colors[1 if hosts[i][7] else 0] for i in G.nodes]
        nx.draw_networkx(
            G, pos, with_labels=False, node_size=2000, node_color=node_colors, alpha=0.8
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=10, font_weight="bold"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.legend(
            handles=[
                patches.Patch(color=colors[1], label="Host"),
                patches.Patch(color=colors[0], label="Replica"),
            ],
            prop={"size": 10, "weight": "bold"},
        )

        plt.savefig("tree.png")
        plt.savefig("tree.svg")

    def generateHostAndReplica(self, layer, hostID, parentID=None):
        """
        Generate a host and its replica

        :param layer: layer of the host
        :param parentID: ID of the parent host

        :return: host and replica
        """

        bw = Bandwidth(self.types[layer]["BwUp"], self.types[layer]["BwDown"])

        # power = eval(self.types[layer]['Power']+'()')
        latency = self.types[layer]["Latency"]

        if layer == "cloud":
            h_ips = r_ips = self.types[layer]["IPS"]
            h_ram = r_ram = RAM(
                self.types[layer]["RAMSize"],
                self.types[layer]["RAMRead"],
                self.types[layer]["RAMWrite"],
            )
            h_disk = r_disk = Disk(
                self.types[layer]["DiskSize"],
                self.types[layer]["DiskRead"],
                self.types[layer]["DiskWrite"],
            )
        else:
            if parentID is None:
                raise Exception(f"Parent ID cannot be None for {layer} layer hosts")

            h_ips = random.randint(
                self.types[layer]["IPS"][0], self.types[layer]["IPS"][1]
            )
            r_ips = random.randint(
                self.types[layer]["IPS"][0], self.types[layer]["IPS"][1]
            )

            h_ram = RAM(
                random.randint(
                    self.types[layer]["RAMSize"][0], self.types[layer]["RAMSize"][1]
                ),
                self.types[layer]["RAMRead"],
                self.types[layer]["RAMWrite"],
            )
            r_ram = RAM(
                random.randint(
                    self.types[layer]["RAMSize"][0], self.types[layer]["RAMSize"][1]
                ),
                self.types[layer]["RAMRead"],
                self.types[layer]["RAMWrite"],
            )

            h_disk = Disk(
                random.randint(
                    self.types[layer]["DiskSize"][0], self.types[layer]["DiskSize"][1]
                ),
                self.types[layer]["DiskRead"] * 5,
                self.types[layer]["DiskWrite"] * 10,
            )
            r_disk = Disk(
                random.randint(
                    self.types[layer]["DiskSize"][0], self.types[layer]["DiskSize"][1]
                ),
                self.types[layer]["DiskRead"] * 5,
                self.types[layer]["DiskWrite"] * 10,
            )

        layer_id = list(self.types.keys()).index(layer)

        host = (h_ips, h_ram, h_disk, bw, latency, layer_id, parentID, hostID + 1)
        replica = (r_ips, r_ram, r_disk, bw, latency, layer_id, parentID, None)

        return host, replica

    def generateHosts(self):
        # FIXED TREE STRUCTURE
        # 1 Cloud, 2 Fog, 4 Edge
        if self.num_hosts != 14:
            raise Exception("MyTreeFog supports only 14 hosts")

        hosts = []
        keys = list(self.types)

        # Cloud
        c_host, c_replica = self.generateHostAndReplica("cloud", 0)
        hosts.append(c_host)  # 0
        hosts.append(c_replica)  # 1

        for i in range(2):
            # Fog
            f_host, f_replica = self.generateHostAndReplica("fog", 2 + 2 * i, 0)

            hosts.append(f_host)  # 2, 4
            hosts.append(f_replica)  # 3, 5

        for j in range(4):
            # Edge
            e_host, e_replica = self.generateHostAndReplica(
                "edge", 6 + 2 * j, 2 + 2 * (j // 2)
            )

            hosts.append(e_host)  # 6, 8, 10, 12
            hosts.append(e_replica)  # 7, 9, 11, 13

        self.plotTree(hosts)
        
        return hosts
