# Import required packages/functions --- The emulator requires sklearn installed ---
import pickle as pk
import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')


# Bounds of the interpolation range
input_bounds = {'fR0':[6.6853e-07, 3.1185e-05],
                'Om':[0.107210, 0.547250],
                'sigma8':[0.49342, 1.3159],
                'a':[1/3,1],
                'z':[0,2]}

required_params = ['Om', 'sigma8']

id_fun = lambda x: x # naive identity function

def rescale_param(cosmo_params,key):
    '''
    Rescale cosmological paramter in [0,1].

    Input
        cosmo_params: dict of cosmological parameters keys and corresponding values. Must contain Om, sigma8.
        key: str identifier of cosmological paramter amongst Om, sigma8.

    Output
        rescaled_value: float; normalised value of the cosmological parameter.
    '''
    value = cosmo_params[key]
    param_min, param_max = input_bounds[key]
    rescaled_value = (value-param_min)/(param_max-param_min)
    if rescaled_value<0 or rescaled_value>1:
        raise ValueError(f'Cosmological parameter ({key}={value}) outside the interpolation range: {key}_range = {input_bounds[key]}')
    return rescaled_value

class BoostPredictor:
    def __init__(self):
        print ("Loading model and related data")
        with open(os.path.join(CACHE_DIR, 'weights.pkl'), 'rb') as f:
            self.weights = pk.load(f)
        self.act_fun_list=[np.tanh, np.tanh, np.tanh, id_fun]
        self.table_mean = np.load(os.path.join(CACHE_DIR, 'TableMean.npy'), allow_pickle=True)
        self.k_vals = np.load(os.path.join(CACHE_DIR, 'k_vals.npy'), allow_pickle=True)
        with open(os.path.join(CACHE_DIR, 'pca.pkl'), 'rb') as f:
            self.pca_components = pk.load(f)

    def model(self, in_array):
        n_layers = len(self.weights)//2
        out_arr = in_array.copy()
        for i in range(n_layers):
            act_fun = self.act_fun_list[i]
            out_arr = act_fun(np.einsum('i,ij->j', out_arr, self.weights[2*i])+self.weights[2*i+1])
        return out_arr
    
    def predict(self, fR0, z, cosmo_params, k_out=None, ext=2):
        '''
        Computes the f(R) boost factor (P_fR/P_GR).

        Input
            fR0: float; value of the f(R) parameter fR0
            z: float; redshift value.
            cosmo_params: dict of cosmological parameters keys and corresponding values. Must contain Om, sigma8.
                          Notice that Om is the sum of CDM and baryon densities. 
                          The cosmology is assumed to be flat LCDM, i.e. O_Lambda = 1-Om 
            k_out: 1D-array, defualt None; custom wavenumber array for which the boost factor is estimated.
                   If None, the algorithm uses the training values of the wavenumber.
            ext : int or str, optional;
                Controls the extrapolation mode for scipy.interpolate.InterpolatedUnivariateSpline;

                if ext=0 or 'extrapolate', return the extrapolated value.
                if ext=1 or 'zeros', return 0
                if ext=2 or 'raise', raise a ValueError
                if ext=3 of 'const', return the boundary value.
                    The default value is 2.
        Output
            Bk: 1D-array; boost factor values
        '''
        if z<input_bounds['z'][0] or z>input_bounds['z'][1]:
            raise ValueError(f'Redshift value (z={z}) outside the interpolation range: z_range = {input_bounds["z"]}')
        a = 1/(1+z)
        if fR0>input_bounds['fR0'][1] or fR0<input_bounds['fR0'][0]:
            raise ValueError(f'f(R) parameter (fR0={fR0}) outside the interpolation range: fR0_range = {input_bounds["fR0"]}', )
        log_fR0 = abs(np.log10(fR0))
        if any(key not in cosmo_params for key in required_params):
            missing_keys = [key for key in required_params if key not in cosmo_params]
            raise KeyError(f'The following keys are missing from the cosmo_params dictonary: {missing_keys}.')
        cosmo_list = [rescale_param(cosmo_params,key) for key in required_params]
        input_arr = np.concatenate([cosmo_list,[a], [log_fR0]])
        raw_predics = self.model(input_arr)
        pca_predics = np.einsum('i,ij->j', raw_predics, self.pca_components)
        predictions = 10**(-3*(pca_predics+self.table_mean))+0.999
        if k_out is not None:
            if ext==2:
                if k_out.min()<self.k_vals.min() or k_out.max()>self.k_vals.max() :
                    raise ValueError(f'Requested k_value is outside of the interpolation range: k_range = {[self.k_vals.min(),self.k_vals.max()]}\n'+
                                    'Set an extrapolation rule with the "ext" keyword to avoid this warning.' )
            return InterpolatedUnivariateSpline(self.k_vals, predictions, ext=ext, k=1)(k_out)
        else:
            return predictions
    