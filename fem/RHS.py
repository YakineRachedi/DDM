import numpy as np
from numpy import linalg as la
from numpy import pi
na = np.newaxis

def point_source(sp, k):    
    def ps(x):
        v = np.zeros(np.size(x,0), float)
        for s in sp:
            v += s[2]*np.exp(-10*(k/(2.0*pi))**2 * la.norm(x - s[na,0:2], axis=1)**2)
        return v
    return ps 