"""
Time series related functions

"""
import numpy as np



def skew(x, axis=0):
    """
    Calculate the skewness of the input time series.
    
    Parameters:
        x : array_like
            Input time series.

        axis : int, optional
            Axis in x corresponding to time. Default: 0.

    Returns:
        y : float
            Skewness of the input time series.    

    Raises:
            
    """
    
    # Preparation
    shp = np.ones(len(x.shape), dtype=int)
    shp[0] = x.shape[axis]
    xm = np.mean(x, axis=axis)
    xm = np.tile(xm, shp)
    xm = np.moveaxis(xm, 0, axis)
    
    # Calculation
    a = np.mean((x-xm)**3, axis=axis)
    b = np.mean((x-xm)**2, axis=axis)**(3/2)
    y = a/b
    
    return y



def asym(x, axis=0):
    """
    Calculate the asymmetry of the input time series.
    
    Parameters:
        x : array_like
            Input time series.

        axis : int, optional
            Axis in x corresponding to time. Default: 0.

    Returns:
        y : float
            Skewness of the input time series.    

    Raises:
            
    """
    from scipy.signal import hilbert

    # Preparation
    shp = np.ones(len(x.shape), dtype=int)
    shp[0] = x.shape[axis]
    xm = np.mean(x, axis=axis)
    xm = np.tile(xm, shp)
    xm = np.moveaxis(xm, 0, axis)
    
    # Calculation
    a = np.mean(hilbert(x-xm).imag**3, axis=axis)
    b = np.mean((x-xm)**2, axis=axis)**(3/2)
    y = a/b
    
    return y

