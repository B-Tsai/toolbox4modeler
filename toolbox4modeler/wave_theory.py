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



def cnoidal_wave(h, H, T, t, x=0., g=9.80665, tol=1e-9):
    """
    Calculate the surface elevation time series of a cnoidal wave, using the 
    approximation of decimals method.
    
    Parameters:
        h : float
            Water depth [m].

        H : float
            Wave height [m].

        T : float
            Wave period [s].

        t : array_like
            Desired output time [s].            

        x : float
            Desired output location [m]. Default: 0.
 
        g : float, optional
            Acceleration of gravity [m^2 s^-2]. Default: 9.80665.

        tol : float, optional
            Tolerance [-] for termination. Default: 1e-9.

    Returns:
        wl : ndarray
            Time series of the surface elevation  [m]. An array with the same 
            size as t.      

        L : float
            Wave length [m].

        c : float
            Wave celerity [m/s].
            
        m: float
            Elliptic parameter [-].

    Raises:
            
    """
    from scipy import special
    
    # Approximate m to the desired tolerence
    m0 = 0.5
    dm = 0.1
    m = np.arange(m0, 1+dm/10, dm)
    while dm >= tol:
        K = special.ellipk(m)
        E = special.ellipe(m)
        c = np.sqrt(g*h)*(1+H/(m*h)*(1-m/2-3/2*E/K))
        L = h*np.sqrt(16/3*m*h/H*c/np.sqrt(g*h))*K
        err = L-c*T;
        m0 = m[err<0]
        dm = dm/10
        m = np.arange(m0[-1], m0[-1]+dm*10.1, dm)
    idx = np.argmin(np.abs(err))
    K = K[idx]
    E = E[idx]
    L = L[idx]
    c = c[idx]
    m = m[idx]
    
    # Calculate time series
    wl2 = H/m*(1-m-E/K)
    _, cn, _, _ = special.ellipj(2*K*(x-c*t)/L, m)
    wl = wl2 + H*cn**2
    
    return wl, L, c, m



def fifth_stokes_wave(h, a, T, g=9.80665):
    """
    Calculate the wave component of a 5th order Stokes wave.
    
    Parameters:
        h : float
            Water depth [m].

        a : float
            Wave amplitude [m].

        T : float
            Wave period [s].
 
        g : float, optional
            Acceleration of gravity [m^2 s^-2]. Default: 9.80665.

    Returns:
        wl : ndarray
            Time series of the surface elevation  [m]. An array with the same 
            size as t.      

        L : float
            Wave length [m].

        c : float
            Wave celerity [m/s].
            
        m: float
            Elliptic parameter [-].

    Raises:
            
    """
    