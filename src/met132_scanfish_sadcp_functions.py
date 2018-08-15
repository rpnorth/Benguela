def grid_scanfish_wrapper(cast_as_ds,dx=500,dz=10,d_factor=500):
    import glob
    import os
    import numpy as np
    import xarray as xr
    import pandas as pd
    import gsw

    # use scipy.interpolate.Rbf to interpolate to grid; use d_factor to put more weight on values on x-axis, and not z-axis
    #d_factor = 500
    #dx = 300 # m
    #dz = 5 # m
    xi,yi,ti,CT_grid,cast_as_ds_corrected =      gridding_scanfish_data(cast_as_ds,varname='CT',     dx=dx/d_factor,dz=dz,d_factor=d_factor)
    xi,yi,ti,SA_grid,cast_as_ds_corrected =      gridding_scanfish_data(cast_as_ds,varname='SA',     dx=dx/d_factor,dz=dz,d_factor=d_factor)
    # xi,yi,sigma_0_grid,cast_as_ds_corrected = gridding_scanfish_data(cast_as_ds,varname='sigma_0',dx=dx/d_factor,dz=dz,d_factor=d_factor)
    p_ref     = 10.1325; # reference pressure # following scanfish calc
    Pot_dens  = gsw.rho(SA_grid,CT_grid,p_ref)
    sigma_0_grid   = Pot_dens - 1000;

    # convert to Dataset
    gridded_as_ds = xr.Dataset({'CT': (['z', 'time'], CT_grid ),'SA': (['z', 'time'], SA_grid ),'sigma_0': (['z', 'time'], sigma_0_grid )},
                               coords={'distance':(['z', 'time'], xi ),'z': (['z'], yi[:,0] ),'time':(['time'], ti[0,:] )})
    
    return gridded_as_ds, cast_as_ds_corrected

def load_process_scanfish_data(pathData):
    import glob
    import os
    import numpy as np
    import xarray as xr
    import pandas as pd
    import gsw

    # load scanfish
    #pathData = r'/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/Scanfish/'                     # use your path
    data_files = glob.glob(os.path.join(pathData, "Scanfish_complete_cast*.nc"))     # advisable to use os.path.join as this makes concatenation OS independent
    #from scanfish_functions import gridding_scanfish_data

    scanfish_data, scanfish_gridded= [], []
    for ti,f in zip(range(len(data_files[0:3:2])),data_files[0:3:2]): # skipping second transect as it doesn't cross filament
        cast_as_ds = xr.open_dataset(f,decode_times=False)
        # Convert time axis to date form:
        cast_as_ds.time.values = pd.to_datetime(cast_as_ds.time.values-719529, unit='D') #[datetime.utcfromtimestamp(i) for i in cast_as_ds.time.values]

        # rough check for outliers; problem seems to be with near-zero values
        cast_as_ds['CT'] = cast_as_ds.CT.where(cast_as_ds.CT>1)
        cast_as_ds['SA'] = cast_as_ds.SA.where(cast_as_ds.SA>1)  
        
        # seems easier if time is a dim/coord
        cast_as_ds = cast_as_ds.swap_dims({'x': 'time'})

        # use scipy.interpolate.Rbf to interpolate to grid; use d_factor to put more weight on values on x-axis, and not z-axis
        #d_factor = 500
        #dx = 500 # m
        #dz = 10 # m
        xi,yi,ti,CT_grid,cast_as_ds_corrected =      gridding_scanfish_data(cast_as_ds,varname='CT',     skip = -1) # just getting cast_as_ds_corrected
        # xi,yi,ti,CT_grid,cast_as_ds_corrected =      gridding_scanfish_data(cast_as_ds,varname='CT',     dx=dx/d_factor,dz=dz,d_factor=d_factor)
        #xi,yi,ti,SA_grid,cast_as_ds_corrected =      gridding_scanfish_data(cast_as_ds,varname='SA',     dx=dx/d_factor,dz=dz,d_factor=d_factor)
        ## xi,yi,sigma_0_grid,cast_as_ds_corrected = gridding_scanfish_data(cast_as_ds,varname='sigma_0',dx=dx/d_factor,dz=dz,d_factor=d_factor)
        #p_ref     = 10.1325; # reference pressure # following scanfish calc
        #Pot_dens  = gsw.rho(SA_grid,CT_grid,p_ref);
        #sigma_0_grid   = Pot_dens - 1000;

        # convert to Dataset
        #gridded_as_ds = xr.Dataset({'CT': (['z', 'x'], CT_grid ),'SA': (['z', 'x'], SA_grid ),'sigma_0': (['z', 'x'], sigma_0_grid )},
        #                            coords={'distance':(['z', 'x'], xi ),'depth': (['z', 'x'], yi ),'time':(['z', 'x'], ti )})

        # create a list of transects; shortcut instead of concat datasets
        #scanfish_gridded.append(gridded_as_ds) #
        scanfish_data.append(cast_as_ds_corrected) #
        
    return scanfish_data # scanfish_gridded,

    
def gridding_scanfish_data(scanfish_data,varname='CT',dt='950400000ns',dx=None,dz=None,d_factor=500,skip=10):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr
    import gsw
    from scipy.interpolate import Rbf

    # 2. Fix lat/lon which don't seem to sample at the same frequency as sensors
    ind_goodlons_lats = np.append(True,np.logical_or(np.diff(scanfish_data.lon.values) != 0,np.diff(scanfish_data.lat.values) != 0)) # add true to include 1st value
    ind_goodlons_lats[-1] = True # set last to true to avoid nan

    # create a ds to interpolate through bad lat/lon values
    # set duplicate coords to nan, drop nan, then interpolate back to original timestamps
    # scanfish_data.time.diff(dim='time').mean() = 1006462390 = 1 second = 1e9 ns rounded
    # 950400000ns seems to be the only value that works ????
    # only want to set nan to lat, lon. So first setup temp datset:
    #temp_data = scanfish_data.where(ind_goodlons_lats).dropna(dim='time').resample(time=dt).interpolate('linear')
    # just need to interpolate across the nan's created using where:
    temp_data = scanfish_data.where(ind_goodlons_lats).interpolate_na(dim='time') 

    # create ds that will have correct lat,lon, and all the rest of the data as original
    scanfish_correct = scanfish_data
    # but use corrected lat,lon
    scanfish_correct['lon'] = temp_data.lon # use "fixed" lat/lon values
    scanfish_correct['lat'] = temp_data.lat
    # and check for nan in variable of interest
    scanfish_correct[varname] = scanfish_correct[varname].interpolate_na(dim='time')

    scanfish_correct['distance'] = np.cumsum(np.append(np.array(0),gsw.distance(scanfish_correct.lon.values,scanfish_correct.lat.values,p=0)))
    
    #Scaling:
    #    if different X coordinates measure different things, Euclidean distance
    #    can be way off.  For example, if X0 is in the range 0 to 1
    #    but X1 0 to 1000, the X1 distances will swamp X0;
    #    rescale the data, i.e. make X0.std() ~= X1.std() .
    # https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
    #scanfish_correct['distance_scaled'] = scanfish_correct.depth.mean() + 
    #                                        (scanfish_correct.distance - scanfish_correct.distance.mean())*scanfish_correct.depth.std()/scanfish_correct.distance.std()
    #d_factor = 500
    scanfish_correct['distance_scaled'] = scanfish_correct.distance/d_factor #
    
    # setup new grid to be interpolated onto
    if dx is None: dx = 300/d_factor# 100*abs(scanfish_correct.distance_scaled.diff(dim='distance').mean().round())
    if dz is None: dz = 5 #10*abs(scanfish_correct.depth.diff(dim='time').mean().round(4))
    x_grid = np.arange(scanfish_correct.distance_scaled.min(),scanfish_correct.distance_scaled.max(),dx)
    #dx_grid = np.arange(scanfish_correct.distance.min(),scanfish_correct.distance.max(),dx)/1000
    #dz_grid = np.arange(scanfish_correct.depth.min(),scanfish_correct.depth.max(),dz)
    z_grid = np.arange(-150,-5,dz)
    XI, YI = np.meshgrid(x_grid, z_grid)

    if skip > 0: # option to skip this part
        # use RBF; epsilon seems to be the only option, higher values give strange output
        rbf = Rbf(scanfish_correct.distance_scaled[::skip].values, scanfish_correct.depth[::skip].values, scanfish_correct[varname][::skip].values, epsilon=1)
        CT_grid = rbf(XI, YI)

        # determine equivalent timestamp
        # based on https://stackoverflow.com/questions/31212141/how-to-resample-a-df-with-datetime-index-to-exactly-n-equally-sized-periods
        ntimesteps = len(scanfish_correct.time)
        time_vec = pd.DataFrame(np.ones(ntimesteps), scanfish_correct.time.values)
        first = time_vec.index.min()
        last = time_vec.index.max() + pd.Timedelta('1s')
        n = len(x_grid) #
        secs = int((last-first).total_seconds()/n)
        periodsize = '{:d}S'.format(secs)
        t_grid = pd.date_range(start=first, periods = n, freq=periodsize)
        TI, YI2 = np.meshgrid(t_grid, z_grid)
    else:
        XI,YI,TI,CT_grid = 0,0,0,0

    # plot the result
    #T_range = np.array((15,18)) #sst_range
    #plt.subplot(2, 1, 1)
    #plt.pcolor(XI*d_factor, YI, CT_grid, cmap=plt.cm.jet,vmin=T_range[0],vmax=T_range[1])
    #plt.scatter(scanfish_correct.distance, scanfish_correct.depth, 15, scanfish_correct.CT, cmap=plt.cm.jet,vmin=T_range[0],vmax=T_range[1])
    #plt.colorbar()
    #plt.scatter(scanfish_correct.distance, scanfish_correct.depth, 15)
    
    return XI*d_factor, YI, TI, CT_grid, scanfish_correct

def load_combine_sadcp_scanfish_data(fileLADCP, pathScan, grid_dx = 750, grid_dz = 10):
    import os
    import glob
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    from . import met132_calc_functions as cf
    #from scanfish_sadcp_functions import load_process_scanfish_data
    
    # load SADCP data
    sadcp = xr.open_dataset(fileLADCP)
    sadcp['distance'] =  xr.DataArray(np.append(np.array(0),gsw.distance(sadcp.lon, sadcp.lat,p=0)),dims='time')

    # load Scanfish data
    scanfish_data = load_process_scanfish_data(pathScan)

    # create transects 
    ind_SADCP_section, ind_scanfish_section = list((1,1)), list((1,1))

    ind_SADCP_section[0] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(sadcp.lon.values > 11., sadcp.lon.values < 12.), 
                                                                                    sadcp.lat.values < -26.25), 
                                                                     sadcp.lat.values > -27), 
                                                      sadcp.time.values <= sadcp.time.sel(time=np.datetime64('2016-11-18T18:37:07'), method='nearest').values),
                                      sadcp.time.values >= sadcp.time.sel(time=np.datetime64('2016-11-18T08:20:00'), method='nearest').values )

    ind_SADCP_section[1] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(sadcp.lon.values > 12., sadcp.lon.values < 13.5),
                                                                                        sadcp.lat.values < -26), 
                                                                         sadcp.time.values <= sadcp.time.sel(time=np.datetime64('2016-11-20T04:30:00'), method='nearest').values ),
                                                          sadcp.time.values >= sadcp.time.sel(time=np.datetime64('2016-11-19T13:30:00'), method='nearest').values ) 

    ind_scanfish_section[0] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(scanfish_data[0].lon.values > 11., scanfish_data[0].lon.values < 12.),
                                                                                       scanfish_data[0].lat.values < -26.25),
                                                                        scanfish_data[0].lat.values > -27),
                                                         scanfish_data[0].time.values <= sadcp.time.sel(time=np.datetime64('2016-11-18T18:37:07'), method='nearest').values),
                                          scanfish_data[0].time.values >= sadcp.time.sel(time=np.datetime64('2016-11-18T08:20:00'), method='nearest').values )

    ind_scanfish_section[1] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(scanfish_data[1].lon.values > 12., scanfish_data[1].lon.values < 13.5),
                                                                                                             scanfish_data[1].lat.values < -26),
                                                                                              scanfish_data[1].lat.values > -27), 
                                                            scanfish_data[1].time.values <= sadcp.time.sel(time=np.datetime64('2016-11-20T04:30:00'),method='nearest').values),
                                             scanfish_data[1].time.values >= sadcp.time.sel(time=np.datetime64('2016-11-19T13:30:00'), method='nearest').values)

    scanfish_gridded_section, scanfish_data_section, scan_sadcp= list((1,1)),list((1,1)),list((1,1))
    for ri in range(len(ind_scanfish_section)):
        
        # create gridded scanfish data for selected sections
        scanfish_gridded_section[ri], scanfish_data_section[ri] = grid_scanfish_wrapper(scanfish_data[ri].drop('distance').drop('distance_scaled').isel(time=ind_scanfish_section[ri]),
                                                                            dx = grid_dx, dz = grid_dz)

        # combine SADCP and Scanfish into one dataset
        sadcp_test = sadcp.isel(time=ind_SADCP_section[ri])
        sadcp_test['z'] = -1*sadcp_test.depth[0,:]
        sadcp_test = sadcp_test.swap_dims({'depth_cell': 'z'})
        sadcp_test = sadcp_test.resample(time='300000000000ns').mean('time')

        scan_test = scanfish_gridded_section[ri]
        scan_test = scan_test.resample(time='300000000000ns').mean('time')
        # get same z coords too, before merging
        sadcp_test = sadcp_test.interp(z=scan_test.z)

        # merge along time axis
        scan_sadcp[ri] = xr.merge([sadcp_test,scan_test])
        # Pressure may be needed
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(Pressure=scan_sadcp[ri].z) # to get right dims
        scan_sadcp[ri].Pressure.values = gsw.p_from_z(scan_sadcp[ri].z,scan_sadcp[ri].lat.mean())
        
        # calculate Vorticity, M**2, N**2, and Ri_Balanced
        scan_sadcp[ri]['across_track_vel'] = scan_sadcp[ri].u #(scan_sadcp[ri].u**2+scan_sadcp[ri].v**2)**0.5
        scan_sadcp[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(scan_sadcp[ri].lon.dropna(dim='time').values,  
                                                                                               scan_sadcp[ri].lat.dropna(dim='time').values,p=0))))
        
        # for plotting better if there is a coord option
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(x_km=scan_sadcp[ri].distance/1000)
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(x_m=scan_sadcp[ri].distance)
        #dx = scan_sadcp[ri].x_km.diff('time').mean().values
        #scan_sadcp[ri] = scan_sadcp[ri].assign_coords(x_km_shift=scan_sadcp[ri].x_km) # for contour plot on pcolormesh
        #dz = scan_sadcp[ri].z.diff('z').mean().values
        #scan_sadcp[ri] = scan_sadcp[ri].assign_coords(z_shift=scan_sadcp[ri].z + dz/2) # for contour plot on pcolormesh

        scan_sadcp[ri] = cf.calc_N2_M2(scan_sadcp[ri])
        scan_sadcp[ri] = cf.calc_vertical_vorticity(scan_sadcp[ri])
        scan_sadcp[ri] = cf.calc_Ri_Balanced(scan_sadcp[ri])
        scan_sadcp[ri] = cf.SI_GI_Check(scan_sadcp[ri])


        #Rib_range = [0,0.25]
        #scan_sadcp[ri].Rib.T.plot(vmin=Rib_range[0],vmax=Rib_range[1])
        #vort_range = [-0.0005,0.0005]
        #scan_sadcp[ri].absolute_vorticity.T.plot(vmin=vort_range[0],vmax=vort_range[1],cmap=plt.cm.coolwarm)

    return scan_sadcp, scanfish_data, sadcp