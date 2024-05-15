"""
Functions for general use

"""
import numpy as np



def disper(w, h, g=9.80665, tol=1e-9, maxiter=100):
    """
    Calculate wave number k from the first order dispersion relation, 
    w^2 = gk tanh(kh), using Newton's method.
    
    Parameters:
        w : array_like
            Angular frequency [s^-1].

        h : float or array_like
            Water depth [m]. Can be a floating-point number or an array with 
            the same size as w.

        g : float, optional
            Acceleration of gravity [m^2 s^-2]. Default: 9.80665.

        tol : float, optional
            Tolerance [m] for termination. Default: 1e-9.

        maxiter : int, optional
            Maximum number of iterations [-]. Default: 100.

    Returns:
        k : ndarray
            Wave number [m^-1]. An array with the same size as w.      

    Raises:
        ValueError
            Input arrays w and h have different size.
            
    """
    
    # Preparation
    w, h = np.array(w), np.array(h)
    w_shp = w.shape
    w, h = w.flatten(), h.flatten()
    k = np.empty(w.shape)
    k[:] = np.nan
    
    # Check inputs
    if len(w) != len(h):
        if len(h) == 1:
            h = h * np.ones(w.shape)
        else:
            raise ValueError('Sizes of w and h do not match.')

    # Loop through all w
    for i in range(len(w)):
        wi = w[i]
        hi = h[i]
        # Solve k using Newton's method
        if wi == 0:
            k[i] = 0
        else:
            err = 1
            ct = 0
            ki0 = wi**2/(g*np.sqrt(np.tanh(wi**2*hi/g))) # Initial guess
            while err > tol and ct < maxiter:
                f = wi**2 - g*ki0*np.tanh(ki0*hi)
                df = -g*np.tanh(ki0*hi) - g*ki0*hi/np.cosh(ki0*hi)**2
                ki = ki0 - f / df
                err = abs(ki - ki0)
                ct += 1
                ki0 = ki
            k[i] = ki
    k = np.reshape(k, w_shp)
    
    return k

