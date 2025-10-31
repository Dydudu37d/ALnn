__version__ = "0.1.0"
__author__ = "Andylisg"
__email__ = "lifuan666@gmail.com"

from .nn import *
import sys
import os
__all__ = ["nn","load_num_data","MiniCNN","MultiLogicNet"]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
