# NewForge
Emulator of the boost factor for the dark matter power spectrum in f(R) gravity models

# Installation
It is recommended to install the package in a dedicated python3 environment. The package requires:
- numpy,
- scipy.

To install the package, access the package directory from a terminal window and execute:

    pip install .

# Using the emulator
The emulator model can be imported in python with

    from NewForge import BoostPredictor

Instantiating a BoostPredictor object loads the MLP model and the auxiliary data

    model = BoostPredictor()

The model can be used to predict the boost factor with the syntax

    Bk = model.predict(fR0, z, cosmo_params)

where fR0, z, cosmo_params are user defined paramters. 

The emulator assumes a flat $\Lambda$-CDM cosmology and neglets the energy density of radiation and neutrinos. The cosmological parameters required in the dictonary cosmo_params to obtain the nDGP boost factors and their allowed ranges are: 
- $\Omega_{\rm m} \in [0.107210, 0.547250]$,
- $sigma_8 \in [0.49342, 1.3159]$.

Notice that the parameter $\Omega_{\rm m}$ accounts for the sum of CDM and baryonic matter. Also the modified gravity parameter $f_{R0}$ and the redsfhit $z$ are required and their interpolation ranges are:
- $f_{R0} \in [6.6853e-07, 3.1185e-05]$.
- $z \in [0,2]$,


A minimal working example for the emulator is shown in the [example notebook](notebooks/example.ipynb).
