import numpy as np
import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds
from scipy.ndimage import gaussian_filter

def setup_dataset_grid(ds):
    
    # create x_left and z_left coords
    if 'xy' in ds:
        ds_full = generate_grid_ds(ds.reset_index('xy').swap_dims({'xy':'x_m'}), {'Z':'z','X':'x_m'})
    else:
        ds_full = generate_grid_ds(ds, {'Z':'z','X':'x_m'})
        
    # spacing in horiz and vert
    ds_full['dx'] = ds_full.x_m.diff('x_m').mean()
    ds_full['dz'] = ds_full.z.diff('z').mean()
    ds_full['dx_left'] = ds_full.x_m_left.diff('x_m_left').mean()
    ds_full['dz_left'] = ds_full.z_left.diff('z_left').mean()
    metrics = {
        ('X'): ['dx','dx_left'], # X distances
        ('Z'): ['dz','dz_left'] # Z distances
        }
    # generate grid
    grid = Grid(ds_full, metrics=metrics, periodic=False)
    
    return ds_full, grid

def calc_coriolis(var_w_lat):

    # Coriolis parameter
    earth_rot = 7.2921*10**-5 # rotation rate of the earth rad/s
    lat_mean = np.mean(var_w_lat.lat) # latitude degrees
    var_w_lat['fo'] = 2*earth_rot*np.sin(lat_mean*np.pi/180) # coriolis parameter for given latitude s^-1

    return var_w_lat

def calc_buoyancy(with_sigma0_x_z,grid):
    g = 9.81
    #rho_o = 10.1325+1000 # needs to be pot density
    rho_o = with_sigma0_x_z.sigma_0.mean()+1000 # needs to be pot density
    with_sigma0_x_z['b'] = -g*(with_sigma0_x_z.sigma_0+1000)/rho_o #
    return with_sigma0_x_z

def calc_M2(with_sigma0_x_z,grid):
    with_sigma0_x_z = calc_buoyancy(with_sigma0_x_z,grid)
    with_sigma0_x_z['db_dx'] = ((-g/rho_o)*grid.diff((with_sigma0_x_z.sigma_0+1000),'X', boundary='fill',fill_value=np.nan)
                                /grid.diff(with_sigma0_x_z.x_m,'X', boundary='fill',fill_value=np.nan))
    with_sigma0_x_z['db_dx'] = grid.interp(with_sigma0_x_z['db_dx'],'X', boundary='fill',fill_value=np.nan) # realign

def calc_N2_M2(with_sigma0_x_z,grid):
    import numpy as np
    import xarray as xr
    import gsw

    # Horizontal and Vertical Buoyancy Gradient 
    g = 9.81
    #rho_o = 10.1325+1000 # needs to be pot density
    rho_o = with_sigma0_x_z.sigma_0.mean()+1000 # needs to be pot density
    with_sigma0_x_z['b'] = -g*(with_sigma0_x_z.sigma_0+1000)/rho_o #

    #b_left = grid.interp(with_sigma0_x_z.b,'X', boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #with_sigma0_x_z['db_dx'] = grid.derivative(b_left,'X', boundary='fill',fill_value=np.nan)
#
    ## need to take sqrt to get N, this is N^2
    #b_top = grid.interp(with_sigma0_x_z.b,'Z',boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #with_sigma0_x_z['db_dz'] = grid.derivative(b_top,'Z', boundary='fill',fill_value=np.nan)

    #b_left = grid.interp((with_sigma0_x_z.sigma_0+1000),'X', boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #with_sigma0_x_z['db_dx'] = (-g/rho_o)*grid.derivative(b_left,'X', boundary='fill',fill_value=np.nan)
    #print('db_dx:', with_sigma0_x_z.x_m, grid.diff(with_sigma0_x_z.x_m,'X', boundary='fill',fill_value=np.nan))
    with_sigma0_x_z['db_dx'] = ((-g/rho_o)*grid.diff((with_sigma0_x_z.sigma_0+1000),'X', boundary='fill',fill_value=np.nan)
                                /grid.diff(with_sigma0_x_z.x_m,'X', boundary='fill',fill_value=np.nan))
    with_sigma0_x_z['db_dx'] = grid.interp(with_sigma0_x_z['db_dx'],'X', boundary='fill',fill_value=np.nan) # realign
    with_sigma0_x_z = calc_M2(with_sigma0_x_z,grid)
    
    # need to take sqrt to get N, this is N^2
    #b_top = grid.interp(with_sigma0_x_z.sigma_0+1000,'Z',boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #with_sigma0_x_z['db_dz'] = (-g/rho_o)*grid.derivative(b_top,'Z', boundary='fill',fill_value=np.nan)
    with_sigma0_x_z['db_dz'] = ((-g/rho_o)*grid.diff((with_sigma0_x_z.sigma_0+1000),'Z', boundary='fill',fill_value=np.nan)
                                /grid.diff(with_sigma0_x_z.z,'Z', boundary='fill',fill_value=np.nan))
    with_sigma0_x_z['db_dz'] = grid.interp(with_sigma0_x_z['db_dz'],'Z', boundary='fill',fill_value=np.nan) # realign

    #db_dz = gsw.Nsquared(with_sigma0_x_z.SA.values, with_sigma0_x_z.CT.values, 
    #                                        with_sigma0_x_z.p.values, lat=with_sigma0_x_z.lat.mean().values, axis=1)
        
    return with_sigma0_x_z

def calc_dv_dx(var_w_v_x_fo,grid,vel_name='across'):
    # assuming m/s and m
    var_w_v_x_fo['dv_dx'] = (grid.diff(var_w_v_x_fo[vel_name],'X', boundary='fill',fill_value=np.nan)
                                /grid.diff(var_w_v_x_fo.x_m,'X', boundary='fill',fill_value=np.nan))
    var_w_v_x_fo['dv_dx'] = grid.interp(var_w_v_x_fo['dv_dx'],'X', boundary='fill',fill_value=np.nan) # realign
    return var_w_v_x_fo
def calc_dv_dz(var_w_v_x_fo,grid,vel_name='across'):
    # assuming m/s and m
    var_w_v_x_fo['dv_dz'] = (grid.diff(var_w_v_x_fo[vel_name],'Z', boundary='fill',fill_value=np.nan)
                                /grid.diff(var_w_v_x_fo.z,'Z', boundary='fill',fill_value=np.nan))
    var_w_v_x_fo['dv_dz'] = grid.interp(var_w_v_x_fo['dv_dz'],'Z', boundary='fill',fill_value=np.nan) # realign
    return var_w_v_x_fo

def calc_vertical_vorticity(var_w_v_x_fo,grid):
    # assuming m/s and m
    
    var_w_v_x_fo = calc_coriolis(var_w_v_x_fo)
        
    #dv_dx  = var_w_v_x_fo['vel_name'].diff(x_name) / var_w_v_x_fo['dist_name'].diff(x_name) 
    #v_left = grid.interp(var_w_v_x_fo.across,'X', boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #var_w_v_x_fo['dv_dx'] = grid.derivative(v_left,'X', boundary='fill',fill_value=np.nan)
    var_w_v_x_fo = calc_dv_dx(var_w_v_x_fo,grid,vel_name='across')
    #var_w_v_x_fo['dv_dx'] = (grid.diff(var_w_v_x_fo.across,'X', boundary='fill',fill_value=np.nan)
    #                            /grid.diff(var_w_v_x_fo.x_m,'X', boundary='fill',fill_value=np.nan))
    #var_w_v_x_fo['dv_dx'] = grid.interp(var_w_v_x_fo['dv_dx'],'X', boundary='fill',fill_value=np.nan) # realign

    #dv_dz  = var_w_v_x_fo['vel_name'].diff('z') / var_w_v_x_fo.z.diff('z') 
    #v_top = grid.interp(var_w_v_x_fo.across,'Z', boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #var_w_v_x_fo['dv_dz'] = grid.derivative(v_top,'Z', boundary='fill',fill_value=np.nan)
    var_w_v_x_fo = calc_dv_dz(var_w_v_x_fo,grid,vel_name='across')
    #var_w_v_x_fo['dv_dz'] = (grid.diff(var_w_v_x_fo.across,'Z', boundary='fill',fill_value=np.nan)
    #                            /grid.diff(var_w_v_x_fo.z,'Z', boundary='fill',fill_value=np.nan))
    #var_w_v_x_fo['dv_dz'] = grid.interp(var_w_v_x_fo['dv_dz'],'Z', boundary='fill',fill_value=np.nan) # realign
            
    #% ====================
    #%  Vertical vorticity 
    #%  ====================
    
    # from Adams et al. 2017, Equation 15
    # dependent on cross-front and vertical gradients in alongfront velocity and buoyancy
    # here, our across-track velocity = along-front velocity and cross-front gradient = x or distance
    if 'db_dz' in var_w_v_x_fo and 'db_dx' in var_w_v_x_fo:
        var_w_v_x_fo['Ertel_Potential_Vorticity'] = ((var_w_v_x_fo.fo - var_w_v_x_fo.dv_dx)*var_w_v_x_fo.db_dz + 
                                                     var_w_v_x_fo.dv_dz*var_w_v_x_fo.db_dx)
    
    #% vertical vort = dv/dx - du/dy
    #% estimate for now, across-track vel(in m/s) / along-track distance (m)
    #du_dy = -dv_dx 
    var_w_v_x_fo['relative_vorticity'] =  - var_w_v_x_fo.dv_dx  # du_dy - dv_dx  
    var_w_v_x_fo['absolute_vorticity'] = var_w_v_x_fo.fo + var_w_v_x_fo.relative_vorticity # + because negative is in relative vorticity already

    #% ang_Rib only applicable if barotropic, centrifugal/inertial
    #% instability are excluded. These occur if fo*absolute_vorticity < 0;
    #% Checking criteria: from Thomas et al. 2013
    var_w_v_x_fo['ang_Rib_Check'] = var_w_v_x_fo['absolute_vorticity']*var_w_v_x_fo.fo
    
    return var_w_v_x_fo

def calc_Ri_Balanced(var_w_v_x_fo_vort_M2,grid):
    import numpy as np
    import xarray as xr
    
    if 'db_dz' not in var_w_v_x_fo_vort_M2:
        var_w_v_x_fo_vort_M2 = calc_N2_M2(var_w_v_x_fo_vort_M2, grid)
    if 'fo' not in var_w_v_x_fo_vort_M2:
        var_w_v_x_fo_vort_M2 = calc_coriolis(var_w_v_x_fo_vort_M2)
    if 'dv_dz' not in var_w_v_x_fo_vort_M2:
        var_w_v_x_fo_vort_M2 = calc_vertical_vorticity(var_w_v_x_fo_vort_M2,grid)
        
    #% --- Balanced Richardson number
    #% Ri = N^2.*abs(du./dz)^-2 = N^2.*f^2./M^4 assuming geostrophic shear
    #% replaces the full shear
    var_w_v_x_fo_vort_M2['Rib'] = var_w_v_x_fo_vort_M2.db_dz * var_w_v_x_fo_vort_M2.fo**2 / var_w_v_x_fo_vort_M2.db_dx**2
    # shifted in vertical and horizontal by diff()
    
    #% Gradient Richardson Number, Ri = N^2/(du/dz)^2 = N^2/M^2
    #% From Ch 14 Intro to GFD: physical meaning to the Richardson number: 
    #% It is essentially a ratio between potential and kinetic energies, with the numerator being the potential-energy barrier that 
    #% mixing must overcome if it is to occur and the denominator being the kinetic energy that the shear flow can supply when smoothed away. 
    #% From Fer et al. 2010 (Faroe) N is based on sorted density profiles,
    #% to approximate "the background stratification against which the
    #% turbulence works"
    # Ri = N^2/(du/dz)^2
    var_w_v_x_fo_vort_M2['Rig'] = var_w_v_x_fo_vort_M2.db_dz/var_w_v_x_fo_vort_M2.dv_dz.T**2 
    
    #% --- Balanced Ri angle 
    #% - from Thompson et al 2016
    #% gravitational instability (-pi/2 < ang_Rib < -pi/4), 
    #% a mixed gravitational and symmetric instability (-pi/4 < ang_Rib < 0), 
    #% symmetric instability (0 < ang_Rib < ang_crit).
    #% ang_Rib = tan^-1(-Rib^-1)
    var_w_v_x_fo_vort_M2['ang_Rib'] = np.arctan(-var_w_v_x_fo_vort_M2.Rib**-1) #  in radians
    var_w_v_x_fo_vort_M2['ang2_Rib'] = np.arctan2(-var_w_v_x_fo_vort_M2.db_dx**2,var_w_v_x_fo_vort_M2.db_dz * var_w_v_x_fo_vort_M2.fo**2) #  in radians
    # shifted in vertical and horizontal by diff()

    #% ang_Rib only applicable if barotropic, centrifugal/inertial
    #% instability are excluded. These occur if fo*absolute_vorticity < 0;
    #% Checking criteria: from Thomas et al. 2013
    #%     imagesc(fo.*absolute_vorticity_mid); colorbar; colormap(cmap3); caxis([-1e-8 1e-8])
    #% always > 0;
    
    return var_w_v_x_fo_vort_M2

#def calc_geostrophic_velocity(var_w_sa_ct_p_lon_lat):
#    import gsw
#
#    # need geostrophic stream function first
#    temp = var_w_sa_ct_p_lon_lat.sortby('z',ascending=False) + 0
#
#    temp['geo_height'] = xr.full_like(temp.SA,fill_value=np.nan) # make empty dataarray
#    temp.geo_height.values = gsw.geo_strf_dyn_height(temp.SA,temp.CT,temp.Pressure,p_ref=10.1325,axis=1) 
#    temp['geo_velocity'] = xr.full_like(temp.geo_height,fill_value=np.nan) # make empty dataarray
#    # smoothing geo_height first as it is noisy and makes and even noisier velocity
#    temp['geo_height'].values = gaussian_filter(temp.geo_height.values, sigma=(5,3))
#    temp.geo_velocity[1:,:] = np.transpose(gsw.geostrophic_velocity(gaussian_filter(temp.geo_height.values, sigma=(5,3)).T,
#                                                                    temp.lon,temp.lat)[0])
#
#    temp = temp.sortby('z',ascending=True)
#    
#    var_w_sa_ct_p_lon_lat['geo_height'] = temp['geo_height']
#    var_w_sa_ct_p_lon_lat['geo_velocity'] = xr.DataArray(temp['geo_velocity'].values, coords=[("x_m_left", temp.x_m_left), ("z", temp.z)])
#    # gsw.geostrophic_velocity shifts along x, now shifting it back
#    # leave it so that in calc_Rossby_Geostrophic interp is not needed
#    # var_w_sa_ct_p_lon_lat['geo_velocity'] = grid.interp(var_w_sa_ct_p_lon_lat['geo_velocity'],'X') 
#
#    return var_w_sa_ct_p_lon_lat

def calc_geostrophic_velocity(var_w_sa_ct_p_lon_lat,grid):
    import gsw
    from scipy.ndimage import gaussian_filter

    # need geostrophic stream function first
    # this way cumsum of dvgeo/dz assumes velocity = 0 at large depths, not surface
    temp = var_w_sa_ct_p_lon_lat.sortby('z',ascending=True) + 0

    #temp['geo_height'] = xr.full_like(temp.SA,fill_value=np.nan) # make empty dataarray
    #temp.geo_height.values = gsw.geo_strf_dyn_height(temp.SA,temp.CT,temp.Pressure,p_ref=10.1325,axis=1) 
    #temp['geo_velocity'] = xr.full_like(temp.geo_height,fill_value=np.nan) # make empty dataarray
    # smoothing geo_height first as it is noisy and makes and even noisier velocity
    #temp['geo_height'].values = gaussian_filter(temp.geo_height.values, sigma=(5,3))
    #temp.geo_velocity[1:,:] = np.transpose(gsw.geostrophic_velocity(gaussian_filter(temp.geo_height.values, sigma=(5,3)).T,
    #                                                                temp.lon,temp.lat)[0])
    
    # Now calculating geostrophic velocity from thermal wind equation as gsw.geostrophic_velocity may not work the way it
    # is setup here. 

    # needs to be pot density
    # -f*dv_dz = g*drho_dx http://www.earth.ox.ac.uk/~helenj/planetearthHT12/Lecture3.pdf
    sigma_0_left = grid.interp(temp.sigma_0,'X', boundary='fill',fill_value=np.nan) # interp first so aligned in the end
    #temp['dsigma_0_dx'] = grid.derivative(sigma_0_left+1000,'X', boundary='fill',fill_value=np.nan)
    temp['dsigma_0_dx'] = (grid.diff((temp.sigma_0+1000),'X', boundary='fill',fill_value=np.nan)
                                /grid.diff(temp.x_m,'X', boundary='fill',fill_value=np.nan))
    temp['dsigma_0_dx'] = grid.interp(temp['dsigma_0_dx'],'X', boundary='fill',fill_value=np.nan) # realign
    
    #temp['geo_velocity'] = xr.full_like(temp.SA,fill_value=np.nan) # make empty dataarray
    rho_o = sigma_0_left.mean()+1000
    dvgeo_dz = -(9.81/temp.fo)*temp['dsigma_0_dx']/rho_o
    # apply 2d smoothing, with larger weight on horizontal to remove noise
    # need to remove colums of nan, otherwise they propogate and half the transect is nan
    #dvgeo_dz = gaussian_filter(dvgeo_dz.dropna('x_m'), sigma=(3,2))
    # sum from bottom up, so that 
    #temp['geo_velocity'][1:-1,:] = np.cumsum(dvgeo_dz*temp.dz.values,1)
    temp['geo_velocity'] = np.cumsum(dvgeo_dz*temp.dz.values,1)

    temp = temp.sortby('z',ascending=False)

    #var_w_sa_ct_p_lon_lat['geo_height'] = temp['geo_height']
    #var_w_sa_ct_p_lon_lat['geo_velocity'] = xr.DataArray(temp['geo_velocity'].values, coords=[("x_m_left", temp.x_m_left), ("z", temp.z)])
    var_w_sa_ct_p_lon_lat['geo_velocity'] = temp['geo_velocity']
    # gsw.geostrophic_velocity shifts along x, now shifting it back
    # leave it so that in calc_Rossby_Geostrophic interp is not needed
    var_w_sa_ct_p_lon_lat['geo_velocity'] = grid.interp(var_w_sa_ct_p_lon_lat['geo_velocity'],'X', boundary='fill',fill_value=np.nan) 

    return var_w_sa_ct_p_lon_lat

def calc_Rossby_Geostrophic(var_w_geoVel_z_f, grid):
    import numpy as np
    import xarray as xr
    import gsw

    if 'fo' not in var_w_geoVel_z_f:
        var_w_geoVel_z_f = calc_coriolis(var_w_geoVel_z_f)
    if 'geo_velocity' not in var_w_geoVel_z_f:
        calc_geostrophic_velocity(var_w_geoVel_z_f,grid)
    # from Adams et al. 2017

    #var_w_geoVel_z_f['Ro_Geo'] = -(grid.derivative(var_w_geoVel_z_f.geo_velocity,'X', boundary='fill',fill_value=np.nan) )/var_w_geoVel_z_f.fo
    var_w_geoVel_z_f['Ro_Geo'] = -(grid.diff(var_w_geoVel_z_f.geo_velocity,'X', boundary='fill',fill_value=np.nan)
                                /grid.diff(var_w_geoVel_z_f.x_m_left,'X', boundary='fill',fill_value=np.nan))/var_w_geoVel_z_f.fo
    return var_w_geoVel_z_f 


def SI_GI_Check(var_w_Rib,grid):

    # from Thomas et al 2013: "A variety of instabilities can develop when the Ertel potential 
    # vorticity (PV), q, takes the opposite sign of the Coriolis parameter (Hoskins, 1974)"
    
    if 'Rib' not in var_w_Rib:
        var_w_Rib = calc_Ri_Balanced(var_w_Rib,grid) # shifted in vertical and horizontal by diff()
    if 'Ertel_Potential_Vorticity' not in var_w_Rib:
        var_w_Rib = calc_vertical_vorticity(var_w_Rib,grid)  # shifted in vertical and horizontal by diff()
    
    # Geostrophic 
    var_w_Rib = calc_geostrophic_velocity(var_w_Rib,grid)
    var_w_Rib = calc_Rossby_Geostrophic(var_w_Rib,grid) # shifted in horizontal by diff()
        
    # from Adams et al. based on Thomas et al 2013
    #var_w_Rib['StableGravMixSymInert'] = xr.full_like(var_w_Rib.Rib,fill_value=np.nan) # make empty dataarray
    var_w_Rib['StableGravMixSymInert'] = (0*(var_w_Rib.Rib > 1/var_w_Rib.Ro_Geo)  + 
                                          1*(var_w_Rib.Rib<=-1) + 
                                          2*(np.logical_and(var_w_Rib.Rib>-1,var_w_Rib.Rib<=0)) + 
                                          3*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, 
                                                                          np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1)), 
                                                           np.logical_and(var_w_Rib.Ro_Geo>0, 
                                                                          np.logical_and(var_w_Rib.Rib>0, 
                                                                                         var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo)))) + 
                                          4*(np.logical_and(var_w_Rib.Ro_Geo<0,
                                                            np.logical_and(var_w_Rib.Rib>1, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo)) ) )
    
    var_w_Rib['StableGravMixSymInert'] = var_w_Rib['StableGravMixSymInert'].where(~np.isnan(var_w_Rib.Rib))

    return var_w_Rib