import numpy as np
from astropy.io import fits
from cosmo_funcs import comoving
import h5py
import scipy.interpolate

# We have to begin with the data, since we must 
# determine what is the angle by which we should
# rotate the mocks.

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

# Interpolate distances to speed up calculation
comov = np.vectorize(comoving)
z_interp = np.linspace(0.55,0.7,1000)
d_interp = comov(H0,Omegam,1-Omegam,-1,0.0,z_interp)
comov = scipy.interpolate.interp1d(z_interp,d_interp)

z_temp = data_w1_z1['zspec_1']
theta_temp = (90-data_w1_z1['delta'])*np.pi/180 # This converts the angle to the usual theta of Sph Coord
alpha_temp = data_w1_z1['alpha']*np.pi/180

cat = np.zeros((3,len(data_w1_z1['alpha'])))
print("Turning (RA,DEC,z) to (x,y,z)\n")
cat[0] = comov(z_temp)*np.sin(theta_temp)*np.cos(alpha_temp)
cat[1] = comov(z_temp)*np.sin(theta_temp)*np.sin(alpha_temp)
cat[2] = comov(z_temp)*np.cos(theta_temp)

# Find the mean position of the catalogue
xmean = cat[0].mean()
ymean = cat[1].mean()
zmean = cat[2].mean()
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
cat_temp = np.einsum('ij,jk',R_z,cat)
cat_temp = np.einsum('ij,jk',R_y,cat_temp)

# Coordinates of the origin
cell_size = 4 #Mpc/h

xmin = cat_temp[0,:].min()
ymin = cat_temp[1,:].min()
zmin = cat_temp[2,:].min()

n_x_orig = xmin/cell_size
n_y_orig = ymin/cell_size
n_z_orig = zmin/cell_size

# Reincorporate magnitude to cat_temp
cat_temp = np.vstack((cat_temp,data_w1_z1['M_I']))

# Bin the objects in spatial grid
xmax = cat_temp[0,:].max()
ymax = cat_temp[1,:].max()
zmax = cat_temp[2,:].max()

x_edges = np.arange(xmin,xmax+cell_size,cell_size)
y_edges = np.arange(ymin,ymax+cell_size,cell_size)
z_edges = np.arange(zmin,zmax+cell_size,cell_size)
M_I_edges = np.asarray([data_w1_z1['M_I'].min(),-20.5,data_w1_z1['M_I'].max()])

bin_edges = np.asarray([x_edges,y_edges,z_edges,M_I_edges])

n_x = len(x_edges)
n_y = len(y_edges)
n_z = len(z_edges)

del cat, cat_temp
# Now let's load the mocks to build
# the selection function

alpha_mocks = np.asarray([])
theta_mocks = np.asarray([])
z_mocks = np.asarray([])
M_I_mocks = np.asarray([])

for i in range(153):
	print("-------------------------------------------------------------------")
	print("Importing mock number ", i, "\n")
	mock_temp = np.loadtxt("../../../ANTONIO_MOCKS/W1mocks/mock_"+str(i+1)+".txt",skiprows=1)

	# The ten columns in these new mocks are

	#1) id
	#2) RA [deg]
	#3) DEC [deg]
	#4) redshift
	#5) B-band abs. mag.
	#6) I-band abs. mag.
	#7) i-band app. mag.
	#8) Galaxy Type: 1 => Red; 2 => Blue
	#9) 0 => Cen.; 1 => Sat.
	#10) id_halo

	# We have to cut these mocks to be in z_1 redshift slice
	print("Prior to z cut, we have ", len(mock_temp), " objects \n")
	mock_temp = mock_temp[np.where( (mock_temp[:,3]>0.55) & (mock_temp[:,3]<0.7) ) ]
	print("After z cut we have ", len(mock_temp), " objects \n")

	alpha_temp = mock_temp[:,1]*np.pi/180
	theta_temp = (90 - mock_temp[:,2])*np.pi/180

	alpha_mocks = np.append(alpha_mocks,alpha_temp)
	theta_mocks = np.append(theta_mocks,theta_temp)
	z_mocks = np.append(z_mocks,mock_temp[:,3])
	M_I_mocks = np.append(M_I_mocks,mock_temp[:,5])

cat = np.zeros((3,len(alpha_mocks)))
print("Turning (RA,DEC,z) to (x,y,z)")
cat[0] = comov(z_mocks)*np.sin(theta_mocks)*np.cos(alpha_mocks)
cat[1] = comov(z_mocks)*np.sin(theta_mocks)*np.sin(alpha_mocks)
cat[2] = comov(z_mocks)*np.cos(theta_mocks)

print("Rotating data catalogue\n")
cat_temp = np.einsum('ij,jk',R_z,cat)
cat_temp = np.einsum('ij,jk',R_y,cat_temp)

# Reincorporate magnitude to cat_temp
cat_temp = np.vstack((cat_temp,M_I_mocks))

print("Binning the full amount of mock galaxies")
grid_cat = np.histogramdd(cat_temp.T, bin_edges )[0]/153

print("Saving them to selfunc_22_95_120.hdf5")
out_name = "selfunc_"+str(n_x)+"_"+str(n_y)+"_"+str(n_z)
out = h5py.File("../../cats/"+out_name+".hdf5",'w')
out.create_dataset(out_name,data=grid_cat)
out.close()
