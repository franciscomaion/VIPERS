import numpy as np
from astropy.io import fits
from cosmo_funcs import comoving

# Cosmological Parameters
Omegak=0.0
w0= -0.999
w1= 0.0
Omegab=0.048206
Omegac=0.2589
Omegam = Omegab+Omegac
H0=67.7
n_SA=0.96
ln10e10ASA=3.085
z_re=9.9999



# Import VIPERS W1 field data
print("Importing data file...\n")
with fits.open('../../W1_combined.fits') as hdul_w1:
    data_w1 = hdul_w1[1].data

print("Making appropriate cuts\n")
print("z_flag>=2, tsr*ssr!=0, z in first slice \n")
data_w1_z1 = data_w1[np.where( (data_w1['zflg']>=2) & (data_w1['tsr']*data_w1['ssr']!=0) & (data_w1['zspec_1']>0.55) & (data_w1['zspec_1']<0.7)  )]

# Vectorize distance calculator
comov = np.vectorize(comoving)
z_temp = data_w1_z1['zspec_1']
theta_temp = (90-data_w1_z1['delta'])*np.pi/180 # This converts the angle to the usual theta of Sph Coord
alpha_temp = data_w1_z1['alpha']*np.pi/180

cat = np.zeros((4,len(data_w1_z1['alpha'])))
print("Turning (RA,DEC,z) to (x,y,z)\n")
cat[0] = data_w1_z1['M_I']
cat[1] = comov(H0,Omegam,1-Omegam, -1,0.0,z_temp)*np.sin(theta_temp)*np.cos(alpha_temp)
cat[2] = comov(H0,Omegam,1-Omegam, -1,0.0,z_temp)*np.sin(theta_temp)*np.sin(alpha_temp)
cat[3] = comov(H0,Omegam,1-Omegam, -1,0.0,z_temp)*np.cos(theta_temp)
tsr = data_w1_z1['tsr']
ssr = data_w1_z1['ssr']

# Find the mean position of the catalogue
xmean = cat[1].mean()
ymean = cat[2].mean()
zmean = cat[3].mean()
print("The mean position of the catalogue is ", xmean, ymean, zmean, "\n")
rho = np.sqrt(xmean**2+ymean**2)

# Angles of this vector wrt original coord. syst.
alpha_mean = np.arctan2(ymean,xmean)
theta_mean = np.arctan(zmean/rho)

vec_mean = np.asarray([xmean,ymean,zmean])

print("alpha_mean =",alpha_mean*180/np.pi, ", theta_mean =" ,theta_mean*180/np.pi, "\n")

#Define the rotation matrices
R_y = np.asarray( [[np.cos(np.pi/2-theta_mean),0,-np.sin(np.pi/2-theta_mean)],[0,1,0],[np.sin(np.pi/2-theta_mean),0,np.cos(np.pi/2-theta_mean)] ] )
R_z = np.asarray( [[np.cos(alpha_mean),np.sin(alpha_mean),0],[-np.sin(alpha_mean),np.cos(alpha_mean),0],[0,0,1]] )

r_mean = np.dot(R_z,vec_mean)
print("The first rotation produced ", r_mean, "\n")

rr_mean = np.dot(R_y,r_mean)
print("The second, and final one produced", rr_mean, "\n")

xmean, ymean, zmean = rr_mean

# Rotate the whole data catalog
print("Rotating data catalogue\n")
cat_temp = np.einsum('ij,jk',R_z,cat[1:,:])
cat_temp = np.einsum('ij,jk',R_y,cat_temp)

print("Saving data catalogue at cat_temp.txt")
np.savetxt('cat_temp.txt',cat_temp)

print("Deleting arrays")
del data_w1, cat_temp, cat

#DONE WITH THE DATA. LET'S DO THE MOCKS
print("Let's do the mocks\n")

n_mocks=153
for i in range(n_mocks):
	print("Importing mock number ", i, "\n")
	data_temp = np.loadtxt("../../../ANTONIO_MOCKS/W1mocks/mock_"+str(i)+".txt")
	