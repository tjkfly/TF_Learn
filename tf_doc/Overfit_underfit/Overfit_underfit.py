# encoding: utf-8
"""
@author: tjk
@contact: tjk@email.com
@time: 2020/4/8 下午7:33
@file: Overfit_underfit.py
@desc: 
"""
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


