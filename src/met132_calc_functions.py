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
    import numpy as np
    import xarray as xr
    import gsw

    # Horizontal and Vertical Buoyancy Gradient 
    g = 9.81
    rho_o = 10.1325+1000 # needs to be pot density
    with_sigma0_x_z['b'] = -g*(with_sigma0_x_z.sigma_0+1000)/rho_o # not sure why .T is necessary, but keeps dims consistent
    with_sigma0_x_z['db_dx'] = with_sigma0_x_z.b.diff('time')/with_sigma0_x_z.distance.diff('time')
    with_sigma0_x_z['db_dz'] = with_sigma0_x_z.b.diff('z')/with_sigma0_x_z.z.diff('z') # negative sign is in b calc
    # using gsw.Nsquared doesn't work as well with xarray
    # with_sigma0_x_z['N2'] = gsw.Nsquared(with_sigma0_x_z.SA, with_sigma0_x_z.CT, with_sigma0_x_z.p, axis=1)
    #x4_dbdx = distance_in[:,0:-1] + np.diff(distance_in)/2 
    #z4_dbdx = depth_in[:,0:-1] #ctd_data.z.isel(xy=ind_CTD_section[ri])[:,0:-1]
    #x4_dbdz = distance_in[0:-1,:] 
    #z4_dbdz = depth_in[0:-1,:] + np.diff(depth_in,axis=0)/2
    
    # Vertical velocity Shear
    with_sigma0_x_z['dV_dz'] = np.sqrt( (with_sigma0_x_z.u.diff('z')/with_sigma0_x_z.z.diff('z'))**2 + (with_sigma0_x_z.v.diff('z')/with_sigma0_x_z.z.diff('z'))**2 )
    
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
    
    dv_dx  = var_w_v_x_fo.across_track_vel.diff('time') / var_w_v_x_fo.distance.diff('time') 
    dv_dz  = var_w_v_x_fo.across_track_vel.diff('z') / var_w_v_x_fo.z.diff('z') 

    #% ====================
    #%  Vertical vorticity 
    #%  ====================
    
    # from Adams et al. 2017, Equation 15
    # dependent on cross-front and vertical gradients in alongfront velocity and buoyancy
    # here, our across-track velocity = along-front velocity and cross-front gradient = x or distance
    var_w_v_x_fo['Ertel_Potential_Vorticity'] = (var_w_v_x_fo.fo - dv_dx)*var_w_v_x_fo.db_dz + dv_dz*var_w_v_x_fo.db_dx
    
    #% vertical vort = dv/dx - du/dy
    #% estimate for now, across-track vel(in m/s) / along-track distance (m)
    du_dy = -dv_dx 
    var_w_v_x_fo['relative_vorticity'] = du_dy - dv_dx  
    var_w_v_x_fo['absolute_vorticity'] = var_w_v_x_fo.fo + var_w_v_x_fo.relative_vorticity # + because negative is in relative vorticity already

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

    #% Gradient Richardson Number, Ri = N^2/(du/dz)^2 = N^2/M^2
    #% From Ch 14 Intro to GFD: physical meaning to the Richardson number: 
    #% It is essentially a ratio between potential and kinetic energies, with the numerator being the potential-energy barrier that 
    #% mixing must overcome if it is to occur and the denominator being the kinetic energy that the shear flow can supply when smoothed away. 
    #% From Fer et al. 2010 (Faroe) N is based on sorted density profiles,
    #% to approximate "the background stratification agains which the
    #% turbulence works"
    var_w_v_x_fo_vort_M2['Rig'] = var_w_v_x_fo_vort_M2.db_dz/var_w_v_x_fo_vort_M2.dV_dz**2 
    
    #% --- Balanced Ri angle 
    #% gravitational instability (-pi/2 < ang_Rib < -pi/4), 
    #% a mixed gravitational and symmetric instability (-pi/4 < ang_Rib < 0), 
    #% symmetric instability (0 < ang_Rib < ang_crit).
    #% ang_Rib = tan^-1(-Rib^-1)
    var_w_v_x_fo_vort_M2['ang_Rib'] = np.arctan(-var_w_v_x_fo_vort_M2.Rib**-1)

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
    temp = var_w_sa_ct_p_lon_lat.sortby('z',ascending=False)
    temp['geo_height'] = xr.full_like(temp.SA,fill_value=np.nan) # make empty dataarray
    temp.geo_height.values = gsw.geo_strf_dyn_height(temp.SA,temp.CT,temp.Pressure,p_ref=10.1325,axis=1) 
    temp['geo_velocity'] = xr.full_like(temp.geo_height,fill_value=np.nan) # make empty dataarray
    temp.geo_velocity[0:-1,:] = np.transpose(gsw.geostrophic_velocity(temp.geo_height.T.values,temp.lon,temp.lat)[0])
    var_w_sa_ct_p_lon_lat = temp.sortby('z',ascending=True)

    return var_w_sa_ct_p_lon_lat

def calc_Rossby_Geostrophic(var_w_geoVel_z_f):
    import numpy as np
    import xarray as xr
    import gsw
    
    # from Adams et al. 2017
    var_w_geoVel_z_f['Ro_Geo'] = -(var_w_geoVel_z_f.geo_velocity.diff('time') / var_w_geoVel_z_f.distance.diff('time') )/var_w_geoVel_z_f.fo

    return var_w_geoVel_z_f

def SI_GI_Check(var_w_Rib):
    import numpy as np
    import xarray as xr

    # from Thomas et al 2013: "A variety of instabilities can develop when the Ertel potential 
    # vorticity (PV), q, takes the opposite sign of the Coriolis parameter (Hoskins, 1974)"
    
    # var_w_lat_v_x_fo_vort_M2 = calc_Ri_Balanced(var_w_lat_v_x_fo_vort_M2)
    
    # Geostrophic 
    var_w_Rib = calc_geostrophic_velocity(var_w_Rib)
    var_w_Rib = calc_Rossby_Geostrophic(var_w_Rib)
    
    # from Adams et al. based on Thomas et al 2013
    var_w_Rib['Instability_GravMixSymInertStab'] = xr.full_like(var_w_Rib.Rib,fill_value=0) # make empty dataarray
    # just consider Gravitational, mixed and symmetric for now
    var_w_Rib['Instability_GravMixSymInertStab'] = 1*1.*(var_w_Rib.Rib<=-1) + \
                                            2*1.*(np.logical_and(var_w_Rib.Rib>-1,var_w_Rib.Rib<=0)) + \
                                            3*1.*(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1)), \
                                                                    np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo))) )
    #var_w_Rib['Instability_GravMixSymInertStab'] = 1*1.*var_w_Rib.Rib.where(var_w_Rib.Rib<=-1, other = 0) + \
    #                                        2*1.*var_w_Rib.Rib.where(np.logical_and(var_w_Rib.Rib>-1,var_w_Rib.Rib<=0), other = 0) + \
    #                                        3*1.*var_w_Rib.Rib.where(np.logical_or(np.logical_and(var_w_Rib.Ro_Geo<=0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1)), \
    #                                                                np.logical_and(var_w_Rib.Ro_Geo>0, np.logical_and(var_w_Rib.Rib>0, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo))) , other = 0) + \
    ##                                        4*1.*var_w_Rib.Rib.where(np.logical_and(var_w_Rib.Ro_Geo<0, np.logical_and(var_w_Rib.Rib>1, var_w_Rib.Rib<=1/var_w_Rib.Ro_Geo)) , other = 0) + \
    #                                        5*1.*var_w_Rib.Rib.where(var_w_Rib.Rib>1/var_w_Rib.Ro_Geo, other = 0) #+ \
    #                                        #100*np.isnan(var_w_Rib.Rib) # for some reason np.nan*() makes everything nan
    ##var_w_Rib['Instability_GravMixSymInertStab'] =  var_w_Rib.Instability_GravMixSymInertStab.where(var_w_Rib.Instability_GravMixSymInertStab<99) # don't plot where Rib was nan
    
    #    % --- Critical angle for SI
    #    % ang_crit = tan^-1(-1- rel_vort./fo) from Thompson et al. 2016
    #    % ang_crit = tan^-1(abs_vort./fo) from Thomas et al. 2013
    #    angd_crit = atand(-1 - relative_vorticity_mid./fo);
    #    ang_crit = atan(-1 - relative_vorticity_mid./fo);
    #%     ang_crit2 = atan(- absolute_vorticity_mid./fo); % produce basically the same result
    #%     ang_crit = atan(-1 - maxnan(maxnan(relative_vorticity_mid))./fo);
    #    
    #    % ang_crit must be < -45
    #    angd_crit(angd_crit>-45) = -45;
    #    ang_crit(ang_crit>-pi/4) = -pi/4;###
    #
    #    % --- Checking for GI and SI
    #%     % Radians: From Thompson et al. 2016, but does not match Thomas et al. 2013
    #%     GI = ang_Rib > -pi()/2 & ang_Rib < -pi()/4;
    #%     GI_SI = ang_Rib > -pi()/4 & ang_Rib < 0;
    #%     SI = ang_Rib > 0 & ang_Rib < ang_crit;
    #    % Degrees: from Thomas et al. 2013
    #    GI = angd_Rib > -180 & angd_Rib < -135;
    #    GI_SI = angd_Rib > -135 & angd_Rib < -90;
    #    SI = angd_Rib > -90 & angd_Rib < angd_crit;#
    #    
    #    GI_SI_Mix = GI;
    #    GI_SI_Mix = GI_SI_Mix + GI_SI.*2;
    #    GI_SI_Mix = GI_SI_Mix + SI.*3;

    return var_w_Rib