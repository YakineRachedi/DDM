import numpy as np
from fem.RHS import point_source

Lx, Ly = 1.0, 2.0
nx, ny = 33, 65
J = 4
k = 16
ns = 8
sp_list = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in range(ns)]
ps = point_source(sp_list, k)
