import numpy as np
from astropy.io import fits
from cosmo_funcs import comoving
import h5py
import scipy.interpolate

# Cosmological Parameters
Omegak=0.0
w0= -0.999
w1= 0.0
Omegab=0.048206
Omegac=0.2589
Omegam = Omegab+Omegac
H0=70.0
n_SA=0.96
ln10e10ASA=3.085
z_re=9.9999

# Import VIPERS W1 field data
print("Importing data file...\n")
with fits.open('../../W1_combined.fits') as hdul_w1:
    data_w1 = hdul_w1[1].data

print("Making appropriate cuts\n")
print("z_flag>=2, tsr*ssr>0, z in first slice \n")
data_w1_z1 = data_w1[np.where( (data_w1['zflg']>=2) & (data_w1['tsr']*data_w1['ssr']>0) & (data_w1['zspec_1']>=0.55) & (data_w1['zspec_1']<=0.7)  )]

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

r_mean = np.einsum('ij,j',R_z,vec_mean)
print("The first rotation produced ", r_mean, "\n")

rr_mean = np.einsum('ij,j',R_y,r_mean)
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

# Reincorporate B-band magnitude to cat_temp
cat_temp = np.vstack((cat_temp,data_w1_z1['M_B']))

# Bin the objects in spatial grid
xmax = cat_temp[0,:].max()
ymax = cat_temp[1,:].max()
zmax = cat_temp[2,:].max()

x_edges = np.arange(xmin,xmax+cell_size,cell_size)
y_edges = np.arange(ymin,ymax+cell_size,cell_size)
z_edges = np.arange(zmin,zmax+cell_size,cell_size)
M_B_edges = np.asarray([data_w1_z1['M_B'].min(),-20.0,data_w1_z1['M_B'].max()])

bin_edges = np.asarray([x_edges,y_edges,z_edges,M_B_edges])

n_x = len(x_edges)-1
n_y = len(y_edges)-1
n_z = len(z_edges)-1

grid_cat = np.histogramdd(cat_temp.T, bin_edges, weights=1/(data_w1_z1['tsr']*data_w1_z1['ssr']))[0]
grid_cat = np.transpose(grid_cat,(3,0,1,2))

# Save grid in hdf5 format
out = h5py.File('../../cats/z1/VIPERS_w1_z1_L0_DATA.hdf5','w')
out.create_dataset('VIPERS_w1_z1_L0_DATA',data=grid_cat)
out.close()

print("Deleting arrays")
del data_w1, cat_temp, cat

#DONE WITH THE DATA. LET'S DO THE MOCKS
print("Let's do the mocks\n")

n_mocks=153
for i in range(n_mocks):
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
	mock_temp = mock_temp[np.where( (mock_temp[:,3]>=0.55) & (mock_temp[:,3]<=0.7) ) ]
	print("After z cut we have ", len(mock_temp), " objects \n")

	alpha_temp = mock_temp[:,1]*np.pi/180
	theta_temp = (90 - mock_temp[:,2])*np.pi/180
	z_temp = mock_temp[:,3]

	cat = np.zeros((3,len(mock_temp)))
	print("Turning (RA,DEC,z) to (x,y,z)\n")
	cat[0] = comov(z_temp)*np.sin(theta_temp)*np.cos(alpha_temp)
	cat[1] = comov(z_temp)*np.sin(theta_temp)*np.sin(alpha_temp)
	cat[2] = comov(z_temp)*np.cos(theta_temp)

	print("Rotating data catalogue\n")
	cat_temp = np.einsum('ij,jk',R_z,cat)
	cat_temp = np.einsum('ij,jk',R_y,cat_temp)

	# We do not calculate new n_i_orig, since we'll use the data ones. This will make all the catalogues consistent
	# In a similar way, we'll use the same xmin and xmax. This guarantees that same bins are precisely
	# the same regions of space.

	# Reincorporate magnitude to cat_temp
	# We must sum this factor onto the mocks magnitude to correct for inconsistencies
	# in the definition of H0 between data and mocks
	cat_temp = np.vstack((cat_temp,mock_temp[:,4]+5*np.log10(0.7)))

	grid_cat = np.histogramdd(cat_temp.T, bin_edges )[0]
	grid_cat = np.transpose(grid_cat,(3,0,1,2))

	out_name = "Box_"+str(i).zfill(3)+"_"+str(n_x)+"_"+str(n_y)+"_"+str(n_z)+"_L0"

	out = h5py.File("../../cats/z1/"+out_name+".hdf5",'w')
	out.create_dataset(out_name,data=grid_cat)
	out.close()
