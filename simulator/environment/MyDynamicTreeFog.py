from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *

import random
import numpy as np
import networkx as nx
from matplotlib import patches
import matplotlib.pyplot as plt


class MyFog:
    def __init__(self, num_hosts):
        ips_multiplier = 2054
        gib_multiplier = 1024
        self.num_hosts = num_hosts

        # based on COSCO paper
        self.types = {
            "edge": {  # B1ms - B4ms with B2ms
                "IPS": [1 * ips_multiplier, 4 * ips_multiplier],
                "RAMSize": [2 * gib_multiplier, 8 * gib_multiplier],
                "RAMRead": 372.0,
                "RAMWrite": 200.0,
                "Latency": 0.003,
                "DiskSize": [4 * gib_multiplier, 16 * gib_multiplier],
                "DiskRead": 13.42,
                "DiskWrite": 1.011,
                "BwUp": 1000,
                "BwDown": 1000,
            },
            "fog": {  # B12ms - B20ms
                "IPS": [12 * ips_multiplier, 20 * ips_multiplier],
                "RAMSize": [48 * gib_multiplier, 80 * gib_multiplier],
                "RAMRead": 360.0,
                "RAMWrite": 305.0,
                "Latency": 0.020,
                "DiskSize": [96 * gib_multiplier, 160 * gib_multiplier],
                "DiskRead": 10.38,
                "DiskWrite": 0.619,
                "BwUp": 2000,
                "BwDown": 2000,
            },
            # removed fog, since it has infinite resources
        }

    def plotTree(self, hosts):
        """
        Plot the tree of hosts

        :param hosts: hosts to plot
        """

        cloud_id = self.num_hosts

        G = nx.DiGraph()

        # add cloud node
        G.add_node(cloud_id, host=None)

        # add nodes
        for i, host in enumerate(hosts):
            G.add_node(i, host=host)

        # add edges
        for i, host in enumerate(hosts):
            parentID = host[6]
            replicaID = host[7]
            if parentID is None:
                G.add_edge(i, cloud_id)
            else:
                G.add_edge(i, parentID)
            if replicaID is not None:
                G.add_edge(i, replicaID)

        # Set node positions for each level
        # hosts with same type are placed on the same level

        pos = {}

        # first display the cloud and replica on the right

        # cloud
        pos[cloud_id] = (0, 0)

        edges = [[host, e] for e, host in enumerate(hosts) if host[5] == 0]
        # print(edges)
        n_edge = len(edges)

        # Edge hosts (they are the ones with the highest number of nodes)
        prev_fog = edges[0][0][6]
        prev_fog_count = 0
        prev_fog_start = -n_edge
        for e, edge_supp in enumerate(edges):
            edge = edge_supp[0]
            edge_id = edge_supp[1]
            fog = edge[6]

            if fog != prev_fog:
                #   print(prev_fog, fog, end="\t")
                #   print(prev_fog_start, -n_edge + 1 + e * 2)
                pos[prev_fog] = (prev_fog_start + 1 + prev_fog_count - 2, -1)
                pos[prev_fog + 1] = (prev_fog_start + 1 + prev_fog_count, -1)
                prev_fog = fog
                prev_fog_count = 0
                prev_fog_start = -n_edge + 1 + e * 2
            else:
                prev_fog_count += 1

            pos[edge_id] = (-n_edge + 1 + e * 2, -2)

        # last fog
        pos[prev_fog] = (prev_fog_start + 1 + prev_fog_count - 2, -1)
        pos[prev_fog + 1] = (prev_fog_start + 1 + prev_fog_count, -1)

        # Draw the graph
        plt.figure(figsize=(int(0.9 * n_edge), 4))
        node_labels = {i: f"{i}" for i in G.nodes}
        node_labels[cloud_id] = "Cloud"
        colors = ["lightblue", "lightgreen", "lightcoral"]
        node_colors = [
            colors[2 if i == cloud_id else 1 if hosts[i][7] else 0] for i in G.nodes
        ]
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

        plt.xlim(-n_edge, n_edge)

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

        h_ips = random.randint(self.types[layer]["IPS"][0], self.types[layer]["IPS"][1])
        r_ips = random.randint(self.types[layer]["IPS"][0], self.types[layer]["IPS"][1])

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
        # DYNAMIC TREE STRUCTURE

        hosts = []
        hosts_counter = 0

        # Cloud (root)
        # Cloud does not count now because it has infinite resources

        # We want to hava exactly self.num_hosts hosts!
        # Each fog has 1 to 5 edge hosts

        while hosts_counter < self.num_hosts:
            print(hosts_counter, self.num_hosts, end="\t")
            remaining_hosts = self.num_hosts - hosts_counter
            if remaining_hosts < 4:
                raise Exception("Remaining hosts must be at least 4")

            # Fog
            fog_id = hosts_counter
            f_host, f_replica = self.generateHostAndReplica("fog", fog_id, None)
            hosts.append(f_host)
            hosts.append(f_replica)
            hosts_counter += 2

            fog_ips = f_host[0]
            fog_max_children = fog_ips // (3 * 2048)
            # minimum max children supported by a fog is 4 and maximum is 10

            # Edge
            remaining_hosts -= 2
            if remaining_hosts < 6:
                n_edge = remaining_hosts // 2
            else:
                n_edge = random.randint(1, min(fog_max_children, remaining_hosts // 2))

            print(n_edge)

            for _ in range(n_edge):
                e_host, e_replica = self.generateHostAndReplica(
                    "edge", hosts_counter, fog_id
                )
                hosts.append(e_host)
                hosts.append(e_replica)
                hosts_counter += 2

            # if remaining hosts are less than 4, add them to the last fog
            if 0 < self.num_hosts - hosts_counter < 4:
                if self.num_hosts - hosts_counter != 2:
                    raise Exception("Remaining hosts must be 2")
                e_host, e_replica = self.generateHostAndReplica(
                    "edge", hosts_counter, fog_id
                )
                hosts.append(e_host)
                hosts.append(e_replica)
                hosts_counter += 2

        self.plotTree(hosts)

        return hosts
