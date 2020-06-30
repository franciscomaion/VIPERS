import numpy as np
import scipy.integrate

def matgrow(Omegam, OmegaDE, w0, w1, zcentral, gamma):
    a = 1./(1.+zcentral)
    w = w0 + (1.-a)*w1
    return( ( Omegam*a**(-3.)/( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )**(gamma) )

def H(H0, Omegam, OmegaDE, w0, w1, z):
    a = 1./(1.+z)
    w = w0 + (1.-a)*w1
    return( H0*np.sqrt( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )

def comoving(H0, Omegam, OmegaDE, w0, w1, z):
	c = 299792.458 #km/s
	z_temp = np.linspace(0, z, 1000)
	integrand = c/H(H0,Omegam, OmegaDE, w0, w1, z_temp)
	d_c = scipy.integrate.simps(integrand, z_temp)
	return(d_c)
