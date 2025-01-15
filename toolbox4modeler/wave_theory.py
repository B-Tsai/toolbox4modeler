"""
Functions for general use

"""
import numpy as np



def disper_1st(h, T, g=9.80665):
    """
    Description:
        Calculate wave number k from the first order dispersion relation, 
        w^2 = gk tanh(kh).
    
    References:
        
    Parameters:
        h : float
            Water depth [m].

        T : float
            Wave peroid [s].

        g : float, optional
            Acceleration of gravity [m^2 s^-2]. Default: 9.80665.

    Returns:
        L : float
            Wave length [m].   

    Raises:
            
    """
    from scipy.optimize import least_squares
    
    # Preparation
    w = 2*np.pi/T
    
    # least-squares function
    def fun_disper(k):
        r = np.sqrt(g*k*np.tanh(k*h)) - w
        return r
    
    # Initial guess
    k0 = w**2/(g*np.sqrt(np.tanh(w**2*h/g)))
    
    # Calculation
    res_lsq = least_squares(fun_disper, k0).x
    L = 2*np.pi/res_lsq[0]
    
    return L



def disper_5th(h, T, H, g=9.80665):
    """
    Description:
        Calculate wave number k from the fifth order dispersion relation, 
        w^2 = gk tanh(kh) (1 + k^2 a^2 w2 + k^4 a^4 w4)^2
    
    References:
        Zhao, K., & Liu, P. L.-F. (2022). On Stokes wave solutions. Proceedings
        of the Royal Society A: Mathematical, Physical and Engineering 
        Sciences, 478(2258), 20210732. https://doi.org/10.1098/rspa.2021.0732
        
    Parameters:
        h : float
            Water depth [m].

        T : float
            Wave peroid [s].

        H : float
            Wave height [m].
            
        g : float, optional
            Acceleration of gravity [m^2 s^-2]. Default: 9.80665.

    Returns:
        L : float
            Wave length [m].   

    Raises:
            
    """
    from scipy.optimize import least_squares
    
    # Preparation
    w = 2*np.pi/T

    # least-squares function
    def fun_disper(x):
        k = x[0]
        a = x[1]
        r = np.zeros(2)
        sgm = np.tanh(k*h)
        a1 = np.cosh(2*k*h)
        w0 = np.sqrt(g*k*sgm)
        w2 = (2*a1**2 + 7)/(4*(a1 - 1)**2)
        w4 = (20*a1**5 + 112*a1**4 - 100*a1**3 - 68*a1**2 - 211*a1 + 328)/(32*(a1 - 1)**5)
        B31 = (3 + 8*sgm**2 - 9*sgm**4)/(16*sgm**4)
        B33 = (27 - 9*sgm**2 + 9*sgm**4 - 3*sgm**6)/(64*sgm**6)
        B51 = (121*a1**5 + 263*a1**4 + 376*a1**3 - 1999*a1**2 + 2509*a1 - 1108)/(192*(a1 - 1)**5)
        B53 = 9*(57*a1**7 + 204*a1**6 - 53*a1**5 - 782*a1**4 - 741*a1**3 - 52*a1**2 + 371*a1 + 186)/(128*(3*a1 + 2)*(a1 - 1)**6)
        B55 = 5*(300*a1**8 + 1579*a1**7 + 3176*a1**6 + 2949*a1**5 + 1188*a1**4 + 675*a1**3 + 1326*a1**2 + 827*a1 + 130)/(384*(12*a1**2 + 11*a1 + 2)*(a1 - 1)**6)
        r[0] = w0 + w0*(k*a)**2*w2 + w0*(k*a)**4*w4 - w
        r[1] = 2*a + 2*a*(B31 + B33)*(k*a)**2 + 2*a**(B51 + B53 + B55)*(k*a)**4 - H        
        return r

    # Initial guess
    k0 = w**2/(g*np.sqrt(np.tanh(w**2*h/g)))
    a0 = H/2
    
    # Calculation
    res_lsq = least_squares(fun_disper, np.array([k0, a0])).x
    L = 2*np.pi/res_lsq[0]
    a = res_lsq[1]
    
    return L, a



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
    