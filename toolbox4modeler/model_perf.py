"""
Functions for prediction-observation comparison

"""
import numpy as np



def rmse(o, p, axis=0):
    """
    Calculate the root mean squared error (RMSE).
    
    References:
        
    Parameters:
        o : array_like
            Observed variate.

        p : array_like
            Predicted variate.
            
        axis : int, optional
            Axis to be averaged. Default: 0.

    Returns:
        rmse : float
            Root mean squared error of the input variates.

    Raises:
            
    """
        
    # Calculation
    rmse = np.sqrt(np.mean((p-o)**2, axis=axis))
    
    return rmse



def mae(o, p, axis=0):
    """
    Calculate the mean absolute error (MAE).
    
    References:
        
    Parameters:
        o : array_like
            Observed variate.

        p : array_like
            Predicted variate.
            
        axis : int, optional
            Axis to be averaged. Default: 0.

    Returns:
        rmse : float
            Root mean squared error of the input variates.

    Raises:
            
    """
        
    # Calculation
    mae = np.mean(np.abs(p-o), axis=axis)
    
    return mae



def wia1981(o, p, axis=0):
    """
    Calculate the Willmott's Index of Agreement (Willmott, 1981).
    
    References:
        Willmott, C. J. (1981). On the Validation of Models. Physical 
        Geography, 2(2), 184–194. 
        https://doi.org/10.1080/02723646.1981.10642213

    Parameters:
        o : array_like
            Observed variate.

        p : array_like
            Predicted variate.
            
        axis : int, optional
            Axis to be averaged. Default: 0.

    Returns:
        d : float
            Willmott's Index of Agreement of the input variates.

    Raises:
            
    """
    
    # Preparation
    shp = np.ones(len(p.shape), dtype=int)
    shp[0] = p.shape[axis]
    om = np.mean(o, axis=axis)
    om = np.tile(om, shp)
    om = np.moveaxis(om, 0, axis)
    
    # Calculation
    a = np.sum((p-o)**2, axis=axis)
    b = np.sum((np.abs(p-om) + np.abs(o-om))**2, axis=axis)
    d = 1 - a/b
    
    return d



def wia1985(o, p, axis=0):
    """
    Calculate the Willmott's Modified Index of Agreement (Willmott et al, 1985).
    
    References:
        Willmott, C. J., Ackleson, S. G., Davis, R. E., Feddema, J. J., Klink, 
        K. M., Legates, D. R., et al. (1985). Statistics for the evaluation and
        comparison of models. Journal of Geophysical Research: Oceans, 90(C5), 
        8995–9005. https://doi.org/10.1029/JC090iC05p08995

    Parameters:
        o : array_like
            Observed variate.

        p : array_like
            Predicted variate.
            
        axis : int, optional
            Axis to be averaged. Default: 0.

    Returns:
        d1 : float
            Willmott's Modified Index of Agreement of the input variates.

    Raises:
            
    """
    
    # Preparation
    shp = np.ones(len(p.shape), dtype=int)
    shp[0] = p.shape[axis]
    om = np.mean(o, axis=axis)
    om = np.tile(om, shp)
    om = np.moveaxis(om, 0, axis)
    
    # Calculation
    a = np.sum(np.abs(p-o), axis=axis)
    b = np.sum(np.abs(p-om) + np.abs(o-om), axis=axis)
    d1 = 1 - a/b
    
    return d1



def wia2012(o, p, axis=0):
    """
    Calculate the Willmott's Refined Index of Agreement (Willmott et al., 2012)
    
    References:
        Willmott, C. J., Robeson, S. M., & Matsuura, K. (2012). A refined index
        of model performance. International Journal of Climatology, 32(13), 
        2088–2094. https://doi.org/10.1002/joc.2419

    Parameters:
        o : array_like
            Observed variate.

        p : array_like
            Predicted variate.
            
        axis : int, optional
            Axis to be averaged. Default: 0.

    Returns:
        dr : float
            Willmott's Refined Index of Agreement of the input variates.

    Raises:
            
    """
    
    # Preparation
    shp = np.ones(len(p.shape), dtype=int)
    shp[0] = p.shape[axis]
    om = np.mean(o, axis=axis)
    om = np.tile(om, shp)
    om = np.moveaxis(om, 0, axis)
    
    # Calculation
    a = np.sum(np.abs(p-o), axis=axis)
    b = 2*np.sum(np.abs(o-om), axis=axis)
    dr = (1 - a/b)*(a <= b) + (b/a - 1)*(a > b)
    
    return dr