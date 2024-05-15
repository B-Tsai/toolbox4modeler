"""
Functions for model pre-processing

"""
import numpy as np



def sw_bthy_inp(xb, zb, h, x, outp_dir, file_name):
    """
    Generate a user-defined input file of bathymetry for SWASH.
    
    References:
        
    Parameters:
        xb : 1-D ndarray of float
            x coordinate of the raw bathymetry [m].
            
        zb : 1-D ndarray of float
            z coordinate of the raw bathymetry [m].

        h : float
            Water depth [m].
                
        x : 1-D ndarray of float
            x coordinate of the model domain [m].
        
        outp_dir : str
            Directory of the output file
            
        file_name : str
            File name of the output file
            
    Returns:

    Raises:
            
    """
    import os
    from scipy import interpolate
    
    # Preparation
    outp = os.path.join(outp_dir, file_name)
    interp1 = interpolate.interp1d(xb, zb, bounds_error=False, fill_value=np.nan)
    z = interp1(x)
    bot = h - z
    con = ['']    
    for i in range(len(bot)):
       con[0] = con[0] + ('{:.6e}\n'.format(bot[i]))

    # Output
    with open(outp, "w") as file:
        file.write(con[0])



def sw_wave_inp(a0, a, w, ps, outp_dir, file_name):
    """
    Generate a user-defined input file of wave spectrum for SWASH.
    
    References:
        
    Parameters:
        a0 : float
            Wave amplitude [m] of the zero harmonic.
            
        a : 1-D ndarray of float
            Wave amplitude [m] of each wave component (a = H/2).
            
        w : 1-D ndarray of float
            Wave angular frequency [rad/s] of each wave component (w = 2*pi/T).
                
        ps : 1-D ndarray of float
            Phase shift [deg] of each wave component.
        
        outp_dir : str
            Directory of the output file
            
        file_name : str
            File name of the output file
            
    Returns:

    Raises:
            
    """
    import os
    
    # Preparation
    outp = os.path.join(outp_dir, file_name)
    con = ('{:.6e} &\n'.format(a0))
    #con = ['']
    for i in range(len(w)):
       con[0] = con[0] + ('{:.6e} {:.6e} {:11.6f} &\n'.format(a[i],w[i],ps[i]))

    # Output
    with open(outp, "w") as file:
        file.write('BOU SIDE W CCW BTYPE WEAK SMOO 1 SEC CON FOUR ' + con[0])
        #file.write('BOU SIDE W CCW BTYPE WEAK SMOO 1 SEC CON FOUR 0 &\n' + con[0])