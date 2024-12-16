"""
Functions for model pre-processing

"""
import numpy as np
import os



def sw_bthy_inp(xb, zb, h0, x, outp_dir, file_name):
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
    from scipy import interpolate
    
    # Preparation
    outp = os.path.join(outp_dir, file_name)
    interp1 = interpolate.interp1d(xb, zb, bounds_error=False, fill_value=np.nan)
    z = interp1(x)
    h = h0 - z
    con = ''
    for i in range(len(h)):
       con = con + ('{:.6e}\n'.format(h[i]))

    # Output
    with open(outp, "w") as file:
        file.write(con)



def sw_wave_inp(a0, a, w, phs, outp_dir, file_name):
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
                
        phs : 1-D ndarray of float
            Phase shift [deg] of each wave component.
        
        outp_dir : str
            Directory of the output file
            
        file_name : str
            File name of the output file
            
    Returns:

    Raises:
            
    """
    
    # Preparation
    outp = os.path.join(outp_dir, file_name)
    con = ''
    for i in range(len(w)):
        con = con + ('{:.6e} {:.6e} {:11.6f} &\n'.format(a[i],w[i],phs[i]))

    # Output
    with open(outp, "w") as file:
        file.write('BOU SIDE W CCW BTYPE WEAK SMOO 1 SEC CON FOUR ' + '{:.6e} &\n'.format(a0) + con)



def cr_bthy_inp(xb, zb, h0, x, outp_dir, file_name):
    """
    Generate a user-defined input file of bathymetry for CROCO.
    
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
    from datetime import datetime
    from scipy import interpolate
    
    import netCDF4 as nc
    
    # Preparation
        
    nx = len(x) - 2
    ny = 1
    dx = np.mean(np.diff(x))
    dy = dx
    size_x = nx*dx
    size_y = ny*dy
    y = np.linspace(-dy, dy, ny+2)
    x_rho, y_rho = np.meshgrid(x, y)
    interp1 = interpolate.interp1d(xb, zb, bounds_error=False, fill_value=np.nan)
    z0 = interp1(x)
    z = np.reshape(np.tile(z0, ny+2), [ny+2,-1])
    h = h0 - z
    

    # Output
    # Create and save a new CROCO grid file
    path_grd = os.path.join(outp_dir, file_name+".nc")
    nc_grd = nc.Dataset(path_grd, 'w', format='NETCDF4')
    nc_grd.created = datetime.now().isoformat()
    nc_grd.type = 'CROCO grid file produced by prep_croco_case.py'

    # create dimensions
    nc_grd.createDimension('one', 1)
    nc_grd.createDimension('xi_rho', nx + 2)
    nc_grd.createDimension('eta_rho', ny + 2)
    nc_grd.createDimension('xi_u', nx + 1)
    nc_grd.createDimension('eta_v', ny + 1)
    nc_grd.createDimension('xi_psi', nx + 1)
    nc_grd.createDimension('eta_psi', ny + 1)
    
    nc_grd.createVariable('xl', 'f8', ('one'))
    nc_grd.variables['xl'].long_name = 'Domain length in the XI-direction'
    nc_grd.variables['xl'].units = 'm'
    nc_grd.variables['xl'][:] = size_x

    nc_grd.createVariable('el', 'f8', ('one'))
    nc_grd.variables['el'].long_name = 'Domain length in the ETA-direction'
    nc_grd.variables['el'].units = 'm'
    nc_grd.variables['el'][:] = size_y

    # create variables and attributes
    nc_grd.createVariable('spherical', 'S1', ('one'))
    nc_grd.variables['spherical'].long_name = 'Grid type logical switch'
    nc_grd.variables['spherical'].option_T = 'spherical'
    nc_grd.variables['spherical'][:] = 'F'

    nc_grd.createVariable('h', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['h'].long_name = 'Bottom depth at RHO-points.'
    nc_grd.variables['h'].units = 'm'
    nc_grd.variables['h'][:] = h

    nc_grd.createVariable('f', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['f'].long_name = 'Coriolis parameter'
    nc_grd.variables['f'].units = 's-1'
    nc_grd.variables['f'][:] = np.zeros(h.shape)

    nc_grd.createVariable('pm', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['pm'].long_name = 'Coordinate transformation metric "m" associated with the differential distances in XI.'
    nc_grd.variables['pm'].units = 'm-1'
    nc_grd.variables['pm'][:] = nx/size_x*np.ones(h.shape)

    nc_grd.createVariable('pn', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['pn'].long_name = 'Coordinate transformation metric "n" associated with the differential distances in ETA.'
    nc_grd.variables['pn'].units = 'm-1'
    nc_grd.variables['pn'][:] = ny/size_y*np.ones(h.shape)

    nc_grd.createVariable('x_rho', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['x_rho'].long_name = 'XI-coordinates at RHO-points.'
    nc_grd.variables['x_rho'].units = 'm'
    nc_grd.variables['x_rho'][:] = x_rho

    nc_grd.createVariable('y_rho', 'f8', ('eta_rho', 'xi_rho'))
    nc_grd.variables['y_rho'].long_name = 'ETA-coordinates at RHO-points.'
    nc_grd.variables['y_rho'].units = 'm'
    nc_grd.variables['y_rho'][:] = y_rho

    nc_grd.close()



def cr_wave_inp(a, w, phs, k, outp_dir, file_name):
    """
    Generate a user-defined input file of wave spectrum for CROCO.
    
    References:
        
    Parameters:            
        a : 1-D ndarray of float
            Wave amplitude [m] of each wave component (a = H/2).
            
        w : 1-D ndarray of float
            Wave angular frequency [rad/s] of each wave component (w = 2*pi/T).
                
        phs : 1-D ndarray of float
            Phase shift [rad] of each wave component.
            
        k : 1-D ndarray of float
            Wave number [rad/m] of each wave component (k = 2*pi/L).
        
        outp_dir : str
            Directory of the output file
            
        file_name : str
            File name of the output file
            
    Returns:

    Raises:
            
    """
    
    # Preparation
    outp = os.path.join(outp_dir, file_name)
    con = ''
    for i in range(len(w)):
        con = con + ('{:.6e} {:.6e} {:.6e} {:.6e}\n'.format(a[i],w[i],phs[i],k[i]))

    # Output
    with open(outp, "w") as file:
        file.write(con)        