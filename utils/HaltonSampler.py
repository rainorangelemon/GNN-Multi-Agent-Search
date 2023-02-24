from scipy.stats import qmc
halton_samplers = {}
halton_samplers[2] = qmc.Halton(d=2, scramble=True)
halton_samplers[7] = qmc.Halton(d=7, scramble=True)