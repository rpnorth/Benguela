import numpy as np
import xarray as xr
from sklearn import preprocessing


def GPR_to_xr(X,y,x1,x2,new_grid_size,kernel,n_restarts_optimizer,var_name):
    # running GaussianProcessRegressor with data in xarray format as input and output
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from itertools import product
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=n_restarts_optimizer)
    # fit the model with training data
    gp.fit(X=X_train, y=Y_train)
    # check how well model does with training and testing data
    pred_train = gp.predict(X_train)
    pred_test = gp.predict(X_test)
    xr_train_test = xr.Dataset({('train'+var_name): (['training_data'],  pred_train),
                        ('test'+var_name): (['testing_data'],  pred_test)},
                        coords={'training_data':(Y_train),'testing_data':(Y_test)})
    
    # now use model on grid
    x1x2 = np.array(list(product(x1, x2)))
    y_pred, y_std = gp.predict(x1x2, return_std=True)
    # R^2 coefficient of determination;
    # Suppose R2 = 0.49. This implies that 49% of the variability of the dependent variable 
    # has been accounted for, and the remaining 51% of the variability is still unaccounted for
    R2 = gp.score(X,y)
    X0p, X1p = x1x2[:,0].reshape(new_grid_size,new_grid_size), x1x2[:,1].reshape(new_grid_size,new_grid_size)
    Zp = np.reshape(y_pred,(new_grid_size,new_grid_size))
    Zstd = np.reshape(y_std,(new_grid_size,new_grid_size))
    xr_gpr = xr.Dataset({var_name: (['x_m','y_m'],  Zp),
                        (var_name+'_std'): (['x_m','y_m'],Zstd),
                        (var_name+'_R**2'): R2},coords={'x_m':(x1),'y_m':(x2)})
    return xr_gpr, xr_train_test

def setup_and_run_GPR(tia_all,var_names,new_grid_size=200):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    kernel = 1.0 * RBF([5e-2,5e-2], (1e-2, 1e2)) # lat, lon
    kernel = 1.0 * RBF([5e0,5e0], (1e-1, 1e2)) # lat, lon
    # 2D input to RBF means anisotropic RBF kernel: assigns different length-scales to the two dimensions.
    #kernel = 1.0 * RBF([5e3,5e3], (1e2, 1e4)) # tested for x,y in m; input is length scale and length scale bounds
    # !! work on kernel https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html

    #new_grid_size = 200 # new grid size
    tia_gpr_allz = [] # empty list
    hv_plot_all = []
    xr_train_test = []
    for z_sel in tia_all.z:
        var_in = tia_all.sel(z=z_sel,method='nearest').dropna('time')
        X = np.array((var_in.lon.values,var_in.lat.values)).T
        # Input space
        x1 = np.linspace(X[:,0].min(), X[:,0].max(),new_grid_size) #p
        x2 = np.linspace(X[:,1].min(), X[:,1].max(),new_grid_size) #q
        x = (np.array([x1, x2])).T

        tia_gpr = []
        for vname in var_names:
            y = (var_in[vname].values)
            #y = preprocessing.scale(y) # standardization
            gpr_out, xr_tt = GPR_to_xr(X,y,x1,x2,new_grid_size,kernel,0,vname)
            tia_gpr.append(gpr_out)
            xr_train_test.append(xr_tt)
        tia_gpr = xr.merge(tia_gpr,compat='override')
        # creat list of datasets
        tia_gpr_allz.append(tia_gpr)

    # turn list of datasets into one dataset
    tia_gpr_allz = xr.concat(tia_gpr_allz,dim='z')
    tia_gpr_allz['z'] = tia_all.z.values # assign depth values
    xr_train_test = xr.merge(xr_train_test,compat='override')
    # Notes:
    # n_restarts_optimizer = 15 made it worse
    # magnitude seems to be related to x,y scale as needed to increase for x,y_m from lat/lon: 
    # kernel = 1.0 * RBF([5e3,5e3], (1e2, 1e4)) # x,y in m
    return tia_gpr_allz, xr_train_test

def calc_KE(sadcp_transects):

    # remove mean
    sadcp_transects['u_prime']=sadcp_transects.u-sadcp_transects.u.mean(dim='xy')
    sadcp_transects['v_prime']=sadcp_transects.v-sadcp_transects.v.mean(dim='xy')
    sadcp_transects['across_prime']=sadcp_transects.across-sadcp_transects.across.mean(dim='xy')
    sadcp_transects['along_prime']=sadcp_transects.along-sadcp_transects.along.mean(dim='xy')
    
    # calculate Kinetic Energy 
    sadcp_transects['full_ke'] = 0.5*((sadcp_transects.u)**2+(sadcp_transects.v)**2) # m**2/s**2
    sadcp_transects['eddy_ke'] = 0.5*((sadcp_transects.u_prime)**2+(sadcp_transects.v_prime)**2) # m**2/s**2
    sadcp_transects['full_fromAcAl_ke'] = 0.5*((sadcp_transects.across)**2+(sadcp_transects.along)**2) # m**2/s**2
    sadcp_transects['full_fromAcAl_ke_x2'] = ((sadcp_transects.across)**2+(sadcp_transects.along)**2) # m**2/s**2

    ## is this the velocity variance? as in Rocha et al. 2016?
    #! what does "spec_est2" do in https://github.com/cesar-rocha/dp_spectra/blob/master/synthetic/aux_func.py ?
    #    ! seems to be what Rocha describes on p 603
    #    ! following https://github.com/cesar-rocha/dp_spectra/blob/master/synthetic/stochastic_spectra.ipynb 
    #    ! also see use of spec.Spectrum in https://github.com/pyspec/pyspec/blob/master/examples/example_1d_spec.ipynb 
    #! spec.Spectrum is basically what I did, but more in depth; use as simple, but double check results
        #! results are similar, without Welch windowing, ie D=N
    # now with across, along and using spec.Spectra
    # why is divergent component so small - read papers on interpreting these plots
#     sadcp_transects['u_full_ke'] = ((sadcp_transects.u)) # after taking fft units become m**2/s**2; divide by 2 done in plotting
#     sadcp_transects['v_full_ke'] = ((sadcp_transects.v)) # 
#     sadcp_transects['u_eddy_ke'] = ((sadcp_transects.u_prime)) #
#     sadcp_transects['v_eddy_ke'] = ((sadcp_transects.v_prime)) # 

    # rotate to get along-across track velocities
    #sadcp_transects = sw.rotate_vel2across_along(sadcp_transects)
#     sadcp_transects['across_ke'] = ((sadcp_transects.across)) # after taking fft units become m**2/s**2; divide by 2 done in plotting
#     sadcp_transects['along_ke'] = ((sadcp_transects.along)) # 
#     sadcp_transects['across_x2'] = ((sadcp_transects.across))*2 # 
#     sadcp_transects['along_x2'] = ((sadcp_transects.along))*2 # 
#     sadcp_transects['across_eddy_ke'] = ((sadcp_transects.across_prime)) # 
#     sadcp_transects['along_eddy_ke'] = ((sadcp_transects.along_prime)) # 
    sadcp_transects['eddy_ke_track'] = 0.5*((sadcp_transects.across_prime)**2+(sadcp_transects.along_prime)**2) # *2

    return sadcp_transects

#def calc_M2(sigma_in,distance_in,depth_in):
#    import numpy as np
#    # Horizontal Buoyancy Gradient 
#    g = 9.81
#    rho_o = 1025-1000
#    b = -g*sigma_in/rho_o # not sure why .T is necessary, but keeps dims consistent
#    db_dx = abs(-np.diff(b,axis=1))/np.diff(distance_in,axis=1)
#    db_dz = abs(-np.diff(b,axis=0))/np.diff(depth_in,axis=0)
#    x4_dbdx = distance_in[:,0:-1] + np.diff(distance_in)/2 
#    z4_dbdx = depth_in[:,0:-1] #ctd_data.z.isel(xy=ind_CTD_section[ri])[:,0:-1]
#    x4_dbdz = distance_in[0:-1,:] 
#    z4_dbdz = depth_in[0:-1,:] + np.diff(depth_in,axis=0)/2
#
#    return b,db_dx,db_dz,x4_dbdx,z4_dbdx,x4_dbdz,z4_dbdz

def calc_N2_M2(with_sigma0_x_z):
    import gsw

    # Horizontal and Vertical Buoyancy Gradient 
    g = 9.81
    rho_o = 10.1325+1000 # needs to be pot density
    with_sigma0_x_z['b'] = -g*(with_sigma0_x_z.sigma_0+1000)/rho_o #
    if 'xy' in with_sigma0_x_z:
        with_sigma0_x_z['db_dx'] = with_sigma0_x_z.b.diff('xy')/with_sigma0_x_z.x_m.diff('xy')
    else:
        with_sigma0_x_z['db_dx'] = with_sigma0_x_z.b.diff('time')/with_sigma0_x_z.distance.diff('time')
    # need to take sqrt to get N, this is N^2
    with_sigma0_x_z['db_dz'] = with_sigma0_x_z.b.diff('z')/with_sigma0_x_z.z.diff('z') # negative sign is in b calc
    # using gsw.Nsquared doesn't work as well with xarray
    # with_sigma0_x_z['N2'] = gsw.Nsquared(with_sigma0_x_z.SA, with_sigma0_x_z.CT, with_sigma0_x_z.p, axis=1)
    #x4_dbdx = distance_in[:,0:-1] + np.diff(distance_in)/2 
    #z4_dbdx = depth_in[:,0:-1] #ctd_data.z.isel(xy=ind_CTD_section[ri])[:,0:-1]
    #x4_dbdz = distance_in[0:-1,:] 
    #z4_dbdz = depth_in[0:-1,:] + np.diff(depth_in,axis=0)/2
    
    # Vertical velocity Shear
    if 'u' in with_sigma0_x_z:
        #with_sigma0_x_z['dV_dz'] = np.sqrt( (with_sigma0_x_z.u.diff('z')/with_sigma0_x_z.z.diff('z'))**2 + (with_sigma0_x_z.v.diff('z')/with_sigma0_x_z.z.diff('z'))**2 )
        with_sigma0_x_z['dV_dz'] = np.sqrt(with_sigma0_x_z.u.diff('z')**2 + with_sigma0_x_z.v.diff('z')**2)/with_sigma0_x_z.z.diff('z')
    
    # mixed layer depth
    # 0.01 kg/m**3 density difference from the shallowest values

    return with_sigma0_x_z

def calc_coriolis(var_w_lat):
    import numpy as np
    import xarray as xr

    # Coriolis parameter
    earth_rot = 7.2921*10**-5 # rotation rate of the earth rad/s
    lat_mean = np.mean(var_w_lat.lat) # latitude degrees
    var_w_lat['fo'] = 2*earth_rot*np.sin(lat_mean*np.pi/180) # coriolis parameter for given latitude s^-1

    return var_w_lat

def calc_vertical_vorticity(var_w_v_x_fo):
    import numpy as np
    import xarray as xr
    # assuming m/s and m
    
    var_w_v_x_fo = calc_coriolis(var_w_v_x_fo)

    if 'across_track_vel' in var_w_v_x_fo:
        if 'xy' in var_w_v_x_fo:
            dv_dx  = var_w_v_x_fo.across_track_vel.diff('xy') / var_w_v_x_fo.distance.diff('xy') 
        else:
            dv_dx  = var_w_v_x_fo.across_track_vel.diff('time') / var_w_v_x_fo.distance.diff('time') 
        dv_dz  = var_w_v_x_fo.across_track_vel.diff('z') / var_w_v_x_fo.z.diff('z') 
    else:
        if 'xy' in var_w_v_x_fo:
            dv_dx  = var_w_v_x_fo.across.diff('xy') / var_w_v_x_fo.x_m.diff('xy') 
        else:
            dv_dx  = var_w_v_x_fo.across.diff('time') / var_w_v_x_fo.x_m.diff('time') 
        dv_dz  = var_w_v_x_fo.across.diff('z') / var_w_v_x_fo.z.diff('z') 
        
    var_w_v_x_fo['dv_dz'] = dv_dz
    var_w_v_x_fo['dv_dx'] = dv_dx
    #% ====================
    #%  Vertical vorticity 
    #%  ====================
    
    # from Adams et al. 2017, Equation 15
    # dependent on cross-front and vertical gradients in alongfront velocity and buoyancy
    # here, our across-track velocity = along-front velocity and cross-front gradient = x or distance
    if 'db_dz' in var_w_v_x_fo and 'db_dx' in var_w_v_x_fo:
        var_w_v_x_fo['Ertel_Potential_Vorticity'] = (var_w_v_x_fo.fo - dv_dx)*var_w_v_x_fo.db_dz + dv_dz*var_w_v_x_fo.db_dx
    
    #% vertical vort = dv/dx - du/dy
    #% estimate for now, across-track vel(in m/s) / along-track distance (m)
    #du_dy = -dv_dx 
    var_w_v_x_fo['relative_vorticity'] =  - dv_dx  # du_dy - dv_dx  
    var_w_v_x_fo['absolute_vorticity'] = var_w_v_x_fo.fo + var_w_v_x_fo.relative_vorticity # + because negative is in relative vorticity already

    #% ang_Rib only applicable if barotropic, centrifugal/inertial
    #% instability are excluded. These occur if fo*absolute_vorticity < 0;
    #% Checking criteria: from Thomas et al. 2013
    var_w_v_x_fo['ang_Rib_Check'] = var_w_v_x_fo['absolute_vorticity']*var_w_v_x_fo.fo
    
    #dist_new = (along_track_distance + along_track_distance(:,1:end-1))./2;

    #% ====================
    #%  Divergence 
    #%  ====================
    #% divergence = dv/dy + du/dx
    #% estimate for now, along-track vel(in m/s) / along-track distance (m)
    #% dv_dy  = diff(along_track_vel./100,1,2) ./ diff(along_track_distance,1,2);
    #% du_dx = 0; % assuming it's small
    #% if nargin > 3 & ~isempty(across_track_distance)
    #%     du_dx = diff(across_track_vel./100) ./ diff(across_track_distance);
    #% end
    #% divergence = dv_dy + du_dx;
    #% === From George's manuscript:
    #% div = (1/r) ?(r Vr)/?r = Vr /r + ?(Vr)/?r 
    #% get same spacing
    #shift_along_track =  (along_track_vel(:,2:end) + along_track_vel(:,1:end-1))./2;
    #divergence = (shift_along_track./100)./dist_new + diff(along_track_vel./100,1,2) ./ diff(along_track_distance,1,2);

    return var_w_v_x_fo

def calc_Ri_Balanced(var_w_v_x_fo_vort_M2):
    import numpy as np
    import xarray as xr
    
    if 'db_dz' not in var_w_v_x_fo_vort_M2:
        var_w_v_x_fo_vort_M2 = calc_N2_M2(var_w_v_x_fo_vort_M2)
    if 'fo' not in var_w_v_x_fo_vort_M2:
        var_w_v_x_fo_vort_M2 = calc_coriolis(var_w_v_x_fo_vort_M2)

    #% --- Balanced Richardson number
    #% Ri = N^2.*abs(du./dz)^-2 = N^2.*f^2./M^4 assuming geostrophic shear
    #% replaces the full shear
    #% BUT, for eddy need to account for curvature of the front. SEE ABOVE:
    #% du/dz = du/dz = (fo + 2.*u_azimuthal./r)^-1.*db/dz
    #%     Rib = fo.^2.*db_dz_centered./db_dy_mid.^2;
    #%     Rib = db_dz_centered.*(fo + 2.*max_cAcross./r_of_max_cAcross).^2./db_dy_mid.^2;
    #var_w_v_x_fo_vort_M2['Rib'] = var_w_v_x_fo_vort_M2.db_dz*(var_w_v_x_fo_vort_M2.fo + 
    #                                                          2*var_w_v_x_fo_vort_M2.across_track_vel/var_w_v_x_fo_vort_M2.distance)**2/var_w_v_x_fo_vort_M2.db_dx**2
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
    var_w_v_x_fo_vort_M2['Rig'] = var_w_v_x_fo_vort_M2.db_dz/var_w_v_x_fo_vort_M2.dV_dz.T**2 
    
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

def calc_geostrophic_velocity(var_w_sa_ct_p_lon_lat):
    import numpy as np
    import xarray as xr
    import gsw

    # need geostrophic stream function first
    temp = var_w_sa_ct_p_lon_lat.sortby('z',ascending=False) + 0
    
    temp['geo_height'] = xr.full_like(temp.SA,
                                      gsw.geo_strf_dyn_height(temp.SA,temp.CT,temp.Pressure,p_ref=10.1325,axis=1)) 
    temp['geo_velocity'] = xr.full_like(temp.geo_height,fill_value=np.nan) # make empty dataarray
    temp.geo_velocity[0:-1,:] = np.transpose(gsw.geostrophic_velocity(temp.geo_height.T.values,temp.lon,temp.lat)[0])
    temp = temp.sortby('z',ascending=True)
    
    var_w_sa_ct_p_lon_lat['geo_height'] = temp['geo_height']
    var_w_sa_ct_p_lon_lat['geo_velocity'] = temp['geo_velocity']

    return var_w_sa_ct_p_lon_lat

def calc_Rossby_Geostrophic(var_w_geoVel_z_f):
    import numpy as np
    import xarray as xr
    import gsw

    if 'fo' not in var_w_geoVel_z_f:
        var_w_geoVel_z_f = calc_coriolis(var_w_geoVel_z_f)
    # from Adams et al. 2017
    if 'xy' in var_w_geoVel_z_f:
        var_w_geoVel_z_f['Ro_Geo'] = -(var_w_geoVel_z_f.geo_velocity.diff('xy') / var_w_geoVel_z_f.x_m.diff('xy') )/var_w_geoVel_z_f.fo
    else:
        var_w_geoVel_z_f['Ro_Geo'] = -(var_w_geoVel_z_f.geo_velocity.diff('time') / var_w_geoVel_z_f.distance.diff('time') )/var_w_geoVel_z_f.fo

    return var_w_geoVel_z_f # shifted in time/x by diff()

def SI_GI_Check(var_w_Rib):
    import numpy as np
    import xarray as xr

    # from Thomas et al 2013: "A variety of instabilities can develop when the Ertel potential 
    # vorticity (PV), q, takes the opposite sign of the Coriolis parameter (Hoskins, 1974)"
    
    if 'Rib' not in var_w_Rib:
        var_w_Rib = calc_Ri_Balanced(var_w_Rib) # shifted in vertical and horizontal by diff()
    if 'Ertel_Potential_Vorticity' not in var_w_Rib:
        var_w_Rib = calc_vertical_vorticity(var_w_Rib)  # shifted in vertical and horizontal by diff()
    
    # Geostrophic 
    var_w_Rib = calc_geostrophic_velocity(var_w_Rib)
    var_w_Rib = calc_Rossby_Geostrophic(var_w_Rib) # shifted in horizontal by diff()
        
    # from Adams et al. based on Thomas et al 2013
    # var_w_Rib['Instability_GravMixSymInertStab'] = xr.full_like(var_w_Rib.Rib,fill_value=np.nan) # make empty dataarray
    var_w_Rib['Instability_StableGravMixSymInertStab'] = 0*1.*(var_w_Rib.Rib > 1/var_w_Rib.Ro_Geo)  + \
                                            1*1.*(var_w_Rib.Rib<=-1) + \
                                            2*1.*(np.logical_and(var_w_Rib.Rib>-1,var_w_Rib.Rib<=0)) + \
                                            3*1.*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1)), \
                                                                    np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo))) ) + \
                                            4*1.*(np.logical_and(var_w_Rib.Ro_Geo<0, np.logical_and(var_w_Rib.Rib>1, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo)) )
    
#     # !!! I'm not sure why this doesn't work. Results don't match method above, and don't make sense; e.g. SI should occur near sloping isopycnals, see Thomas et al. 2016 Figure 7
#     var_w_Rib['ang_crit'] = np.arctan(-var_w_Rib.Ro_Geo) # presumably in radians
#     var_w_Rib['Instability_StableGravMixSymInertStab_fromThomas'] = xr.full_like(var_w_Rib.Rib,fill_value=np.nan) # make empty dataarray
#     var_w_Rib['Instability_StableGravMixSymInertStab_fromThomas'] =  1*1.*(np.rad2deg(var_w_Rib.ang_Rib) <= -45) + \
#                                             2*1.*(np.logical_and(np.rad2deg(var_w_Rib.ang_Rib) >= -45, np.rad2deg(var_w_Rib.ang_Rib) < 0 )) + \
#                                             3*1.*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(np.rad2deg(var_w_Rib.ang_Rib)> 0, np.rad2deg(var_w_Rib.ang_Rib)<= 45)), \
#                                                               np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(np.rad2deg(var_w_Rib.ang_Rib)> 0, np.rad2deg(var_w_Rib.ang_Rib)<=var_w_Rib.ang_crit))) ) + \
#                                             4*1.*(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(np.rad2deg(var_w_Rib.ang_Rib)> 45, np.rad2deg(var_w_Rib.ang_Rib)<= -np.rad2deg(var_w_Rib.ang_crit))) ) +\
#                                             0*1.*(np.rad2deg(var_w_Rib.ang_Rib) > -np.rad2deg(var_w_Rib.ang_crit) )

#     #    % --- Checking for GI and SI
#     #%     % Radians: From Thompson et al. 2016, but does not match Thomas et al. 2013
#     # from Thompson et al. 2016 ang_crit = arctan(-1-vertical_relative_vorticity_from_along_track_and_across_track_velocity_estimated_from_glider/fo)
#     # var_w_Rib.Ro_Geo is -geostrophic_velocity_estimate / fo; so it already has -ve in front of it, thus a + is used here
    
#     # from Adams referencing Thompson: Ro_Geo = rel_vort_from_geo/fo ~= -(d_alongfront_vel_geo/d_along_track_distance)/fo
#     # from Thompson: ang_crit = arctan(-1-rel_vort_from_geo/fo)
#     # therefore ang_crit = arctan(-1-Ro_Geo)
    
# #     var_w_Rib['ang_crit'] = np.arctan(-1 - var_w_Rib.Ro_Geo) # presumably in radians
# #     var_w_Rib['Instability_StableGravMixSymInertStab_fromThompson'] =  0*1.*(var_w_Rib.ang_Rib > var_w_Rib.ang_crit)  + \
# #                                             1*1.*(np.logical_and(var_w_Rib.ang_Rib >= -np.pi/2, var_w_Rib.ang_Rib < -np.pi/4)) + \
# #                                             2*1.*(np.logical_and(var_w_Rib.ang_Rib >= -np.pi/4, var_w_Rib.ang_Rib < 0)) + \
# #                                             3*1.*(np.logical_and(var_w_Rib.ang_Rib >= 0, var_w_Rib.ang_Rib < var_w_Rib.ang_crit))
 
#     var_w_Rib['Instability_StableGravMixSymInertStab_fromThompson'] = xr.full_like(var_w_Rib.Rib,fill_value=np.nan) # make empty dataarray
#     var_w_Rib['Instability_StableGravMixSymInertStab_fromThompson'] =  1*1.*(var_w_Rib.ang_Rib < -np.pi/1.3333) + \
#                                             2*1.*(np.logical_and(var_w_Rib.ang_Rib >= -np.pi/1.3333, var_w_Rib.ang_Rib < -np.pi/2)) + \
#                                             3*1.*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/2, var_w_Rib.ang_Rib<=-np.pi/4)), \
#                                                                     np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/2, var_w_Rib.ang_Rib<=var_w_Rib.ang_crit))) ) + \
#                                             4*1.*(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/4, var_w_Rib.ang_Rib<=var_w_Rib.ang_crit)) ) +\
#                                             0*1.*(var_w_Rib.ang_Rib > var_w_Rib.ang_crit) 
# #     var_w_Rib['ang_crit'] = np.arctan(-var_w_Rib.Ro_Geo) # presumably in radians
# #     var_w_Rib['Instability_StableGravMixSymInertStab_fromThomas'] = xr.full_like(var_w_Rib.Rib,fill_value=np.nan) # make empty dataarray
# #     var_w_Rib['Instability_StableGravMixSymInertStab_fromThomas'] =  1*1.*(var_w_Rib.ang_Rib < -np.pi/1.3333) + \
# #                                             2*1.*(np.logical_and(var_w_Rib.ang_Rib >= -np.pi/1.3333, var_w_Rib.ang_Rib < -np.pi/2)) + \
# #                                             3*1.*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/2, var_w_Rib.ang_Rib<=-np.pi/4)), \
# #                                                                     np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/2, var_w_Rib.ang_Rib<=var_w_Rib.ang_crit))) ) + \
# #                                             4*1.*(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.ang_Rib>-np.pi/4, var_w_Rib.ang_Rib<=var_w_Rib.ang_crit)) ) +\
# #                                             0*1.*(var_w_Rib.ang_Rib > var_w_Rib.ang_crit) 
    

#     #GI = var_w_Rib.ang_Rib > -pi()/2 & ang_Rib < -pi()/4;
#     #GI_SI = ang_Rib > -pi()/4 & ang_Rib < 0;
#     #SI = ang_Rib > 0 & ang_Rib < ang_crit;
#     #    % Degrees: from Thomas et al. 2013
#     #GI = angd_Rib > -180 & angd_Rib < -135;
#     #GI_SI = angd_Rib > -135 & angd_Rib < -90;
#     #SI = angd_Rib > -90 & angd_Rib < angd_crit;#
#     #    
#     #    GI_SI_Mix = GI;
#     #    GI_SI_Mix = GI_SI_Mix + GI_SI.*2;
#     #    GI_SI_Mix = GI_SI_Mix + SI.*3;

    return var_w_Rib