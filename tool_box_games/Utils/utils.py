import numpy as np

def get_HW(tabular):
        """ return the HW of 8bits values array """
        HW_val = np.array([bin(n).count("1") for n in range(0,256)],dtype=int)
        return HW_val[tabular]
