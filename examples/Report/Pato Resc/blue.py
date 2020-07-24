from matplotlib.colors import LinearSegmentedColormap
import warnings
import numpy as np

warnings.filterwarnings("ignore",category=UserWarning)

def colour():

    data = np.array([[30,120,180],[30,120,180]])/255+0.05

    blueMap = LinearSegmentedColormap.from_list('blue',data)
    return blueMap