import numpy as np
from pkg_resources import resource_stream

def test_predict(model):
    # set reference parameters
    cosmo_params = {'Om':0.32,
                'sigma8':0.8}
    fR0 = 1e-5
    z = 1

    Bk = model.predict(fR0,z,cosmo_params)
    Bk_ref = np.load(resource_stream('NewForge','cache/Test_Bk.npy'), allow_pickle=True)

    assert all(abs(Bk-Bk_ref)<1e-7) , f"Test failed: the model could not reproduce the reference boost factor."

    print("All tests passed successfully.")