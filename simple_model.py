import numpy as np


muabo = np.genfromtxt("Lab3/muabo.txt", delimiter=",")
muabd = np.genfromtxt("Lab3/muabd.txt", delimiter=",")

#found values from datasheet: red=600nm, green=520nm, blue=460nm
red_wavelength = 600
green_wavelength = 520
blue_wavelength = 460

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

def transmittance_rate(mua,musr,depth):
    return np.exp(-np.sqrt(3*mua*(musr+mua))*depth)


# TODO calculate penetration depth
#oppgave 2.1a
delta = np.sqrt(1/(3*(musr+mua)*mua))

#oppgave 2.1b
transmittance_finger = transmittance_rate(mua,musr,13e-3)

#oppgave 2.1c

#oppgave 2.1d
bvf_100 = 1
mua_100 = mua_blood*bvf_100 + mua_other
blood_vessel_diameter = 300e-6
blood_vessel_transmittance = transmittance_rate(mua_100,musr,blood_vessel_diameter)
tissue_transmittance = transmittance_rate(mua,musr,blood_vessel_diameter)

print(delta)
print(transmittance_finger)

print(blood_vessel_transmittance)
print(tissue_transmittance)
