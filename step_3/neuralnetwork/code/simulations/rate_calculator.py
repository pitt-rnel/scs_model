import numpy as np
def firing_from_spn():
    pass


def pressure_from_volume():
    pass



def pel_firing_from_bladder(x):
    FRlow = 0.015 * x ** 3 - 0.0002 * x ** 2 + 0.05924 * x
    if FRlow <= 0:
        fr = np.Inf
    else:
        fr = FRlow
    if fr <= 0:
        interval = 99999.
    elif fr >= 200:
        interval = 1000.0 / 200
    elif fr < 200 and noise:
        mean = 1000.0 / fr  # ms
        sigma = mean * 0.2
        interval = rnd.normalvariate(mean, sigma)
    else:
        interval = 1000.0 / fr  # ms

