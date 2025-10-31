__version__ = "0.1.0"
__author__ = "Andylisg"
__email__ = "lifuan666@gmail.com"

from .nn import *
from .test_alnn import *
import sys
import os
__all__ = ["nn","load_num_data","MiniCNN","MultiLogicNet","has_gpu","cuda","test_multi_logic_net","test_mini_cnn"]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
