"""
Functions for wave filtering

"""
import numpy as np



def wave_no_filt(h, wl, dt, dx=0):
    """
    Calculate wave component.
    
    References:
        
    Parameters:
        h : float
            Water depth [m].
            
        wl : 2-D ndarray of float
            Water level [m]. One of the dimension should be the same
            as x, and the other dimension corresponds to time.

        dt : float
            Time step [s] of wl.

        dx : float
            Relative distance [m] from the desired output location to the 
            location where wl is obtained. Default: 0.
            
    Returns:
        a : 1-D ndarray of float
            Wave amplitude [m] of each wave component (a = H/2).

        k : 1-D ndarray of float
            Wave number [1/m] of each wave component.
            
        w : 1-D ndarray of float
            Wave angular frequency [rad/s] of each wave component (w = 2*pi/T).
                
        ps : 1-D ndarray of float
            Phase shift [rad] of each wave component.

    Raises:
            
    """ 
    from toolbox4modeler import wave
    
    # Preparation
    n = len(wl)
    w = 2*np.pi/dt*np.arange(n)/n
    k = wave.disper(w, h)
    
    # Calculation
    y = np.fft.fft(wl)
    
    # Output
    idx = range(int(np.floor(n/2)))
    a = np.abs(y[idx])/n*2
    k = k[idx]
    w = w[idx]
    ps = np.angle(y[idx]*np.exp(-1j*k*dx))

    return a, k, w, ps



def wave_filt_1d_multi_wg(x, h, wl, dt, x0=0):
    """
    Separate reflected waves using multiple wave gauges.
    
    References:
        
    Parameters:
        x : 1-D ndarray of float
            Locations [m] of wl obtained (along the wave propagation 
            direction). Must have at least two points.

        h : float
            Water depth [m].
            
        wl : 2-D ndarray of float
            Water level [m]. One of the dimension should be the same
            as x, and the other dimension corresponds to time.

        dt : float
            Time step [s] of wl.

        x0 : float
            Location [m] of the output. Default: 0.
            
    Returns:
        a : 1-D ndarray of float
            Wave amplitude [m] of each wave component (a = H/2).

        k : 1-D ndarray of float
            Wave number [1/m] of each wave component.
            
        w : 1-D ndarray of float
            Wave angular frequency [rad/s] of each wave component (w = 2*pi/T).
                
        ps : 1-D ndarray of float
            Phase shift [rad] of each wave component.

    Raises:
            
    """
    from toolbox4modeler import wave
        
    # Preparation
    n = len(wl)
    w = 2*np.pi/dt*np.arange(n)/n
    k = wave.disper(w, h)
    
    # Calculation
    y = np.fft.fft(wl, axis=0)
    Fir = np.zeros((n,2), dtype=complex)
    for i in range(n):
        if i == 0:
            Fir[i,:] = np.array([0, 0])
        else:
            eikxmT = np.array([np.exp(-1j*k[i]*x), np.exp(1j*k[i]*x)])
            Fm = eikxmT @ np.reshape(y[i,:], (-1,1))
            eikxm = eikxmT @ eikxmT.transpose()
            Fir[i,:] = np.linalg.solve(eikxm, Fm).transpose()
    
    # Output
    idx = range(int(np.floor(n/2)))
    a = np.abs(Fir[idx,:])/n*2
    k = k[idx]
    w = w[idx]
    ps = np.angle(Fir[idx,:]*np.array([np.exp(-1j*k*x0),np.exp(-1j*k*x0)]).transpose())

    return a, k, w, ps