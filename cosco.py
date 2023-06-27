import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from time import time
from subprocess import call
from os import system, rename


# Simulator imports
from simulator.Simulator import *
from simulator.environment.MyFog import *

#from simulator.workload.MyBitbrainWorkload import *
from simulator.workload.MyAzure2019Workload import *

# Scheduler imports
from scheduler.zMyScheduler import MyScheduler


# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp

usage = "usage: python main.py"


# Global constants
NUM_SIM_STEPS = 100
#HOSTS = 10 * 5 if opts.env == '' else 10
HOSTS =  3 * 2

<<<<<<< HEAD
CONTAINERS = HOSTS * 20
FAILURES = HOSTS * 20
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
#NEW_CONTAINERS = 0 if HOSTS == 10 else 5
NEW_CONTAINERS = 2
=======
CONTAINERS = HOSTS * 5
FAILURES = HOSTS * 5
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
#NEW_CONTAINERS = 0 if HOSTS == 10 else 5
NEW_CONTAINERS = 1
>>>>>>> bc28d1d734d4c1688ed2ae3d1cb951eedfa65b8c

FAULT_RATE = 0.3
FAULT_TIME = 6
FAULT_INCREASE_TIME = 2
RECOVER_TIME = 18
FAULTY_HOSTS = [0,1,2]
FAILURE_TYPES = ['CPU', 'RAM']
ACCUMULATIVE_FAULTS = True

def initalizeEnvironment(prints= True):

	# Initialize simple fog datacenter
	if prints:
		print('Initializing environment...')
	''' Can be SimpleFog, BitbrainFog, AzureFog // Datacenter '''
	datacenter = MyFog(HOSTS)

	# Initialize workload
	if prints:
		print('Initializing workload...')
	''' Can be SWSD, BWGD2, Azure2017Workload, Azure2019Workload // DFW, AIoTW '''
	#workload = BWGD2(NEW_CONTAINERS, 1.5)
	#workload = MyBW(NEW_CONTAINERS)
	#workload = Azure2019Workload(NEW_CONTAINERS, 1.5)
	workload = MyAzure2019Workload(NEW_CONTAINERS)

	# Initialize scheduler
	if prints:
		print('Initializing scheduler...')
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
	#scheduler = GOBIScheduler('energy_latency_'+str(HOSTS)) # GOBIScheduler('energy_latency_'+str(HOSTS))
	scheduler = MyScheduler()

	# Initialize Environment
	if prints:
		print('Generating hosts...')
	hostlist = datacenter.generateHosts()


	if prints:
		print('Initializing simulator...')
	env = Simulator(ROUTER_BW, scheduler, CONTAINERS, FAILURES, INTERVAL_TIME, hostlist)


	# Initialize stats
	if prints:
		print('Initializing stats...')
	stats = Stats(env, workload, datacenter, scheduler)


	# Execute first step
	if prints:
		print('Executing first step...')
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	start = time()
	decision = scheduler.placement(deployed) # Decide placement using container ids
	schedulingTime = time() - start
	migrations = env.allocateInit(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	if prints:
		print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
		print("Containers in host:", env.getContainersInHosts())
		print("Schedule:", env.getActiveContainerList())
		printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
	return datacenter, workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats, prints= True):
	if prints:
		print(f"STEP {env.interval}")

	destroyed = env.destroyCompletedContainers()
	
	if prints:
		print(f"Destroyed = {[c.id for c in destroyed]}")

	# Create new container in the next layer
	if destroyed:
		for container in destroyed:
			cont_ltype = container.getLType()
			if cont_ltype < 2:
				workload.generateNewContainers(env.interval, cont_ltype+1)	#new container for the next layer

	
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	if prints:
		print([(c[0],c[1]) for c in newcontainerinfos])
	deployed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	
	if prints:
		print(f"Deployed = {deployed}")

	start = time()
	replica_decision = scheduler.selection() # Select container IDs for migration to replica
	selected = [cid for cid,_ in replica_decision]
	decision = scheduler.filter_placement(scheduler.placement(deployed)) # Decide placement for selected container ids
	decision = replica_decision + decision
	schedulingTime = time() - start


	## Failures injection
	failuredecision = []
	failuresdeployed = []

	CYCLE_TIME = FAULT_TIME + RECOVER_TIME
	if FAULT_RATE:
		cycle_stage = env.interval % CYCLE_TIME
		
		if ACCUMULATIVE_FAULTS:
			if cycle_stage == 0:
				# clear all faults
				env.clearFailures()

			elif cycle_stage >= RECOVER_TIME and (cycle_stage - RECOVER_TIME) % FAULT_INCREASE_TIME == 0:

				# inject faults
				if FAULTY_HOSTS :
					newfailuresinfo = []

					for targetID in FAULTY_HOSTS:

						failure_type = []

						for f_type in FAILURE_TYPES:
							if random.random() < FAULT_RATE:
								failure_type.append(f_type)

						newfailuresinfo = workload.generateNewFailures(env.interval, env.hostlist[targetID], failure_type = failure_type, max_duration = FAULT_TIME)

					failuresdeployed = env.addFailures(newfailuresinfo)

					# When increasing the scenario complexity, the failuredecision list will need to be changed (passing targetID as "parameter"?)
					failuredecision += [(fid, env.failurelist[fid].getLType()) for fid in failuresdeployed]

		
		else:
			if cycle_stage == 0:
				# clear all faults
				env.clearFailures()

			elif cycle_stage == RECOVER_TIME:
				#targetID = 0
				
				if FAULTY_HOSTS:
					newfailuresinfo=[]

					for targetID in FAULTY_HOSTS:
						
						if random.random() < FAULT_RATE:
							#targetID = 0
							newfailuresinfo = workload.generateNewFailures(env.interval, env.hostlist[targetID])
						#print(newfailuresinfo)

					failuresdeployed = env.addFailures(newfailuresinfo)

					failuredecision += [(fid, env.failurelist[fid].getLType()) for fid in failuresdeployed]


	if prints:
		print(f"Failures Deployed = {failuresdeployed}")
	
	migrations, failures = env.simulationStep(decision, failuredecision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload deployed using creation IDs
	workload.updateDeployedFailures(env.getFailuresCreationIDs(failures, failuresdeployed)) # Update workload deployed using creation IDs
	
	if prints:
		print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
		print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
		print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
		print("Containers in host:", env.getContainersInHosts())
		print("Num active containers:", env.getNumActiveContainers())
		print("Host allocation:", [(c.getHostID() if c else -1) for c in env.containerlist])
		printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)

def saveStats(stats, datacenter, workload, env, save_essential=False):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)

	if save_essential:
		stats.generateCompleteDataset(dirname, stats.hostinfo, 'hostinfo')
		return
	
	stats.generateDatasets(dirname)

	stats.generateGraphs(dirname)
	stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':
	datacenter, workload, scheduler, env, stats = initalizeEnvironment()

	for step in range(NUM_SIM_STEPS):
		print(color.BOLD+"Simulation Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)
		
<<<<<<< HEAD
	saveStats(stats, datacenter, workload, env)

def runCOSCO(prints = False, save_essential = True):
=======

	saveStats(stats, datacenter, workload, env)

def runCOSCO(prints = False, save_essential = True):	
>>>>>>> bc28d1d734d4c1688ed2ae3d1cb951eedfa65b8c

	datacenter, workload, scheduler, env, stats = initalizeEnvironment(prints)

	for step in range(NUM_SIM_STEPS):
		if prints:
			print(color.BOLD+"Simulation Interval:", step, color.ENDC)

		stepSimulation(workload, scheduler, env, stats, prints)

	saveStats(stats, datacenter, workload, env, save_essential = save_essential)