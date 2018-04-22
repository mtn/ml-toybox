#!/usr/bin/env python3

import random
import numpy as np

class Network(Object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes


