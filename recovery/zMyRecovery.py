import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *

class MyRecovery(Recovery):
    def __init__(self, hosts, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator'
        self.training = training

    def run_model(self, time_series, original_decision):
        
        return original_decision

