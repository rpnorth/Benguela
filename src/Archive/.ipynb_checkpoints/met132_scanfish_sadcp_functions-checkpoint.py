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

def load_process_scanfish_data_2ds_format(pathData):
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

def load_combine_sadcp_scanfish_data(fileSADCP, pathScan, grid_dx = 750, grid_dz = 10):
    import os
    import glob
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    import Denmark_Strait.src.spectra_and_wavelet_functions as sw
    #from . import met132_calc_functions as cf
    from collections import OrderedDict
    #from scanfish_sadcp_functions import load_process_scanfish_data
    
    # load SADCP data
    sadcp = xr.open_dataset(fileSADCP)
    #sadcp['distance'] =  xr.DataArray(np.append(np.array(0),gsw.distance(sadcp.lon, sadcp.lat,p=0)),dims='time')
    sadcp['x_m'] =  xr.DataArray(np.cumsum(np.append(np.array(0),gsw.distance(sadcp.lon, sadcp.lat,p=0))),dims='time')
    # rotate to get along-across track velocities
    sadcp = sw.rotate_vel2across_along(sadcp)
    # get z instead of depth
    sadcp['z'] = -1*sadcp.depth[0,:]
    sadcp = sadcp.swap_dims({'depth_cell': 'z'})#.drop('depth_cell')
    # fill all nan using linear interpolation along time axis; don't need for Scanfish as it will be gridded
    sadcp = sadcp.interpolate_na(dim='time')
    
    # load Scanfish data
    scanfish_data = load_process_scanfish_data_2ds_format(pathScan)
    
    def sadcp_select(sadcp,ind_SADCP_section):
        # This method ensures uniform dx, but loses time information; see sadcp_scanfish_combine for the other option
        sadcp_out = sadcp.isel(time=ind_SADCP_section)
        # to preserve some time information, as "time" variable is lost with interpolation
        sadcp_out = sadcp_out.assign_coords(time_secs=sadcp_out.time.astype(int))
        # make x_m the dimesion
        sadcp_out = sadcp_out.swap_dims({'time': 'x_m'}) 
        # need distance variable starting at 0m to get interpolation right
        sadcp_out.x_m.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out.lon.dropna(dim='x_m').values,  
                                                                                                   sadcp_out.lat.dropna(dim='x_m').values,p=0))))
        # interp to even spacing
        if sadcp_out.x_m.diff(dim='x_m').max() > 700:
            dx_sadcp = 1000
        else:
            dx_sadcp = 500 
        print('Setting dx to: ' , dx_sadcp , ' m')
        sadcp_out = sadcp_out.interp(x_m=np.arange(0,sadcp_out.x_m.max(),dx_sadcp))
        
        # rebuild time
        sadcp_out = sadcp_out.assign_coords(time=sadcp_out.time_secs.astype('datetime64'))
        
        sadcp_out = sadcp_out.rename({'x_m': 'xy'})
        sadcp_out.coords['x_m'] = sadcp_out.xy
        sadcp_out = sadcp_out.assign_coords(x_km=sadcp_out.x_m/1000)
        sadcp_out = sadcp_out.set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
        return sadcp_out
    
    # function to combine selected section
    def sadcp_scanfish_combine(sadcp,scanfish_data,ind_SADCP_section,ind_scanfish_section,grid_dx,grid_dz):
        
        # create gridded scanfish data for selected sections
        # !!! This is somewhat unreliable, as lon/lat aren't consistent (missing data) but are necessary for distance calc !!!
        scanfish_gridded_section, scanfish_data_section = grid_scanfish_wrapper(scanfish_data.drop('distance').drop('distance_scaled').isel(time=ind_scanfish_section),
                                                                            dx = grid_dx, dz = grid_dz)

        # combine SADCP and Scanfish into one dataset; get similar variable and time and z variables
        sadcp_test = sadcp.isel(time=ind_SADCP_section)
        #sadcp_test = sadcp_test.resample(time='300000000000ns').mean('time')
        #sadcp_test = sadcp_test.interp(time=scanfish_gridded_section.time)
        sadcp_test = sadcp_test.interp_like(scanfish_gridded_section)
        print(sadcp_test,scanfish_gridded_section)
        
        #scan_test = scanfish_gridded_section
        #scan_test = scan_test.resample(time='300000000000ns').mean('time')
        # get same z coords too, before merging
        sadcp_test = sadcp_test.interp(z=scanfish_gridded_section.z)

        # merge along time axis
        scan_sadcp = xr.merge([sadcp_test,scanfish_gridded_section])
        # Pressure may be needed
        scan_sadcp = scan_sadcp.assign_coords(Pressure=scan_sadcp.z) # to get right dims
        scan_sadcp.Pressure.values = gsw.p_from_z(scan_sadcp.z,scan_sadcp.lat.mean())

        # get distance starting at 0, from lat lon positions
        scan_sadcp['x_m_full'] = scan_sadcp['x_m']
        scan_sadcp['x_m'].values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(scan_sadcp.lon.dropna(dim='time').values,  
                                                                                               scan_sadcp.lat.dropna(dim='time').values,p=0))))

        # now interpolate to uniform x-spacing instead of time, as this is more useful in this analysis
        # now get to correct x-spacing, because result from scanfish gridding is not reliable
        new_time_grid = pd.to_datetime(np.interp(np.arange(0,scan_sadcp.x_m.max(),grid_dx),scan_sadcp.x_m.values,pd.to_datetime(scan_sadcp.time.values).astype(int))) 
        scan_sadcp = scan_sadcp.interp(time=new_time_grid).drop('distance')
        # will have changed slightly, so need to redo
        scan_sadcp['x_m'].values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(scan_sadcp.lon.dropna(dim='time').values,  
                                                                                               scan_sadcp.lat.dropna(dim='time').values,p=0))))
        
        # to create multi-index without losing time information; rename to associate all variables with xy instead of time
        scan_sadcp = scan_sadcp.swap_dims({'time': 'x_m'}) 
        scan_sadcp = scan_sadcp.rename({'x_m': 'xy'})
        scan_sadcp.coords['x_m'] = scan_sadcp.xy
        scan_sadcp = scan_sadcp.assign_coords(x_km=scan_sadcp.x_m/1000)
        scan_sadcp = scan_sadcp.assign_coords(time_secs=scan_sadcp.time.astype(int))
        scan_sadcp = scan_sadcp.set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
        return scan_sadcp
    
    ind_SADCP_section, ind_SADCP_section2, ind_scanfish_section = list((1,1)), list((1,1)), list((1,1))

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
        
    scan_sadcp_out = list((1,1))
    scan_sadcp_out[0] = sadcp_scanfish_combine(sadcp,scanfish_data[0],ind_SADCP_section[0],ind_scanfish_section[0],grid_dx,grid_dz)
    scan_sadcp_out[1] = sadcp_scanfish_combine(sadcp,scanfish_data[1],ind_SADCP_section[1],ind_scanfish_section[1],grid_dx,grid_dz)
    
    sadcp_outOD = OrderedDict() # Preallocate output dictionary
    sadcp_outOD['ScanTransect1'] = sadcp_select(sadcp,ind_SADCP_section[0])
    sadcp_outOD['ScanTransect2'] = sadcp_select(sadcp,ind_SADCP_section[1])
    sadcp_outOD['FullScanTransect1'] = sadcp_select(sadcp,np.logical_and(sadcp.time.values >= np.datetime64('2016-11-17T18:22:07'),sadcp.time.values <= np.datetime64('2016-11-18T21:02:00')))
    sadcp_outOD['AcrossUpwelling1'] = sadcp_select(sadcp,np.logical_and(sadcp.time.values >= np.datetime64('2016-11-25T07:10:00'),sadcp.time.values <= np.datetime64('2016-11-25T13:00:00')))
    sadcp_outOD['WestwardFromUpwellTransect1'] = sadcp_select(sadcp,np.logical_and(sadcp.time.values >= np.datetime64('2016-11-27T05:02:00'),sadcp.time.values <= np.datetime64('2016-11-27T17:32:00')))
    sadcp_outOD['AwayFromFilament1'] = sadcp_select(sadcp,np.logical_and(sadcp.time.values >= np.datetime64('2016-11-29T17:52:00'),sadcp.time.values <= np.datetime64('2016-11-30T02:30:00')))
    sadcp_outOD['ToCapetown1'] = sadcp_select(sadcp,np.logical_and(sadcp.time.values >= np.datetime64('2016-12-08T04:00:00'),sadcp.time.values <= np.datetime64('2016-12-09T10:00:00')))

    return scan_sadcp_out,scanfish_data, sadcp_outOD, sadcp




def xxOLD_load_combine_sadcp_scanfish_data(fileSADCP, pathScan, grid_dx = 750, grid_dz = 10):
    import os
    import glob
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    from . import met132_calc_functions as cf
    from collections import OrderedDict
    #from scanfish_sadcp_functions import load_process_scanfish_data
    
    # load SADCP data
    sadcp = xr.open_dataset(fileSADCP)
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

    scanfish_gridded_section, scanfish_data_section, scan_sadcp, sadcp_out = list((1,1)),list((1,1)),list((1,1)),list((1,1))
    sadcp_outOD = OrderedDict() # Preallocate output dictionary
    for ri in range(len(ind_scanfish_section)):
        
        # create gridded scanfish data for selected sections
        scanfish_gridded_section[ri], scanfish_data_section[ri] = grid_scanfish_wrapper(scanfish_data[ri].drop('distance').drop('distance_scaled').isel(time=ind_scanfish_section[ri]),
                                                                            dx = grid_dx, dz = grid_dz)

        # combine SADCP and Scanfish into one dataset
        sadcp_test = sadcp.isel(time=ind_SADCP_section[ri])
        sadcp_test['z'] = -1*sadcp_test.depth[0,:]
        sadcp_test = sadcp_test.swap_dims({'depth_cell': 'z'})
        sadcp_out[ri] = sadcp_test # for output
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
        
        # get distance from lat lon positions
        # a bit backwards, but because of distance_scaled etc., waiting until new distance calc to swap time and distance variables
        scan_sadcp[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(scan_sadcp[ri].lon.dropna(dim='time').values,  
                                                                                               scan_sadcp[ri].lat.dropna(dim='time').values,p=0))))

        # a more universal variable name as a coordinate
        #scan_sadcp[ri] = scan_sadcp[ri].swap_dims({'time': 'distance'}).rename({'distance': 'x_m'})
        scan_sadcp[ri] = scan_sadcp[ri].rename({'distance': 'x_m'})
        scan_sadcp[ri] = scan_sadcp[ri].set_coords('x_m')
        
        # add lat lon as coords
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(lon=scan_sadcp[ri].lon)
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(lat=scan_sadcp[ri].lat)
        
        #dx = scan_sadcp[ri].x_km.diff('time').mean().values
        #scan_sadcp[ri] = scan_sadcp[ri].assign_coords(x_km_shift=scan_sadcp[ri].x_km) # for contour plot on pcolormesh
        #dz = scan_sadcp[ri].z.diff('z').mean().values
        #scan_sadcp[ri] = scan_sadcp[ri].assign_coords(z_shift=scan_sadcp[ri].z + dz/2) # for contour plot on pcolormesh

        # calculate Vorticity, M**2, N**2, and Ri_Balanced
        scan_sadcp[ri]['across_track_vel'] = scan_sadcp[ri].u #(scan_sadcp[ri].u**2+scan_sadcp[ri].v**2)**0.5
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(distance=scan_sadcp[ri].x_m) # temporary for these calcs
        scan_sadcp[ri] = cf.calc_N2_M2(scan_sadcp[ri])
        scan_sadcp[ri] = cf.calc_vertical_vorticity(scan_sadcp[ri])
        scan_sadcp[ri] = cf.calc_Ri_Balanced(scan_sadcp[ri])
        scan_sadcp[ri] = cf.SI_GI_Check(scan_sadcp[ri])
        scan_sadcp[ri] = scan_sadcp[ri].drop('distance') # sticking with x_m

        # now interpolate to uniform x-spacing instead of time, as this is more useful in this analysis
        scan_sadcp[ri] = scan_sadcp[ri].swap_dims({'time': 'x_m'}) 
        # do time separately, otherwise it is lost with scan_sadcp[ri].interp()
        new_time = pd.to_datetime(np.interp(np.arange(0,grid_dx*scan_sadcp[ri].x_m.size,grid_dx),
                                            scan_sadcp[ri].x_m.values,pd.to_datetime(scan_sadcp[ri].time.values).astype(int))) 
        scan_sadcp[ri] = scan_sadcp[ri].interp(x_m=np.arange(0,grid_dx*scan_sadcp[ri].x_m.size,grid_dx))
        # put time back in
        scan_sadcp[ri].coords['time'] = scan_sadcp[ri].x_m
        scan_sadcp[ri]['time'].values = new_time.values
        # and create multi-dimension 
        # to create multi-index without losing time information; rename to associate all variables with xy instead of time
        scan_sadcp[ri] = scan_sadcp[ri].rename({'x_m': 'xy'})
        scan_sadcp[ri].coords['x_m'] = scan_sadcp[ri].xy
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(x_km=scan_sadcp[ri].x_m/1000)
        scan_sadcp[ri] = scan_sadcp[ri].assign_coords(time_secs=scan_sadcp[ri].time.astype(int))
        scan_sadcp[ri] = scan_sadcp[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
        # a high resolution version of SADCP data
        sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
        sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
        sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
        sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
        sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
        #print(sadcp_out[ri].x_m.diff(dim='x_m').mean(),sadcp_out[ri].x_m.diff(dim='x_m').max(),sadcp_out[ri].x_m.diff(dim='x_m').min())
        if ri == 0:
            if sadcp_out[ri].x_m.diff(dim='x_m').max() > 700:
                dx_sadcp = 1000
            else:
                dx_sadcp = 500 #sadcp_out[ri].x_m.diff(dim='x_m').max().round(-2).values #300 # for the two transects, dx is currently < 300, so 300 seems acceptable
            print('Setting dx to: ' , dx_sadcp , ' m')
        new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                            sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
        sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
        sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
        sadcp_out[ri]['time'].values = new_time2.values
        sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
        sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
        sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
        sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
        sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])

        sadcp_outOD['ScanTransect{}'.format(ri+1)]  = sadcp_out[ri] # quickest way to change output to Ordered Dict
        

    # more transects, sadcp only
    ri = 0    
    sadcp_out[ri] = sadcp.sel(time=slice(np.datetime64('2016-11-17T18:22:07'),np.datetime64('2016-11-18T21:02:00')))
    sadcp_out[ri]['z'] = -1*sadcp_out[ri].depth[0,:]
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'depth_cell': 'z'})
         
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
    sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
    sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
    new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                         sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
    sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
    sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
    sadcp_out[ri]['time'].values = new_time2.values
    sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
    sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
    sadcp_outOD['FullScanTransect{}'.format(ri+1)] = sadcp_out[ri]
    
    # transect near upwelling, not straight line
    sadcp_out[ri] = sadcp.sel(time=slice(np.datetime64('2016-11-25T07:10:00'),np.datetime64('2016-11-25T13:00:00')))
    sadcp_out[ri]['z'] = -1*sadcp_out[ri].depth[0,:]
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'depth_cell': 'z'})
        
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
    sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
    sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
    new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                         sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
    sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
    sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
    sadcp_out[ri]['time'].values = new_time2.values
    sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
    sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
    sadcp_outOD['AcrossUpwelling{}'.format(ri+1)] = sadcp_out[ri]

    # transect leaving upwelling, heading west across filament
    sadcp_out[ri] = sadcp.sel(time=slice(np.datetime64('2016-11-27T05:02:00'),np.datetime64('2016-11-27T17:32:00')))
    sadcp_out[ri]['z'] = -1*sadcp_out[ri].depth[0,:]
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'depth_cell': 'z'})
       
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
    sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
    sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
    new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                         sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
    sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
    sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
    sadcp_out[ri]['time'].values = new_time2.values
    sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
    sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
    sadcp_outOD['WestwardFromUpwellTransect{}'.format(ri+1)] = sadcp_out[ri]

    # long route during storm, not straight line
    #sadcp_out[ri] = sadcp.sel(time=slice(np.datetime64('2016-11-29T17:52:00'),np.datetime64('2016-11-30T05:00:00')))
    #ind_section = np.logical_and(np.logical_and(np.logical_and(sadcp.lon.values > 10.6, sadcp.lon.values < 11.4),
    #                                            sadcp.time.values >= sadcp.time.sel(time=np.datetime64('2016-11-29T17:52:00'), method='nearest').values),
    #                             sadcp.time.values <= sadcp.time.sel(time=np.datetime64('2016-11-30T09:00:00'), method='nearest').values )
    ind_section = np.logical_and(np.logical_and(np.logical_and(sadcp.lon.values > 10.6, sadcp.lon.values < 11.4),
                                                sadcp.time.values >= np.datetime64('2016-11-29T17:52:00'),
                                 sadcp.time.values <= np.datetime64('2016-11-30T09:00:00') ))
    sadcp_out[ri] = sadcp.isel(time=ind_section)
    print(sadcp_out[ri].time.values,sadcp.time.values)
    sadcp_out[ri]['z'] = -1*sadcp_out[ri].depth[0,:]
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'depth_cell': 'z'})
        
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
    sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
    sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
    new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                         sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
    sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
    #new_time1 = sadcp_out[ri].time
    sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
    sadcp_out[ri]['time'].values = new_time2.values
    sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
    sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
    sadcp_outOD['AwayFromFilament{}'.format(ri+1)] = sadcp_out[ri]
    
    # long route during storm, not straight line
    sadcp_out[ri] = sadcp.sel(time=slice(np.datetime64('2016-12-08T04:00:00'),np.datetime64('2016-12-08T10:00:00')))
    sadcp_out[ri]['z'] = -1*sadcp_out[ri].depth[0,:]
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'depth_cell': 'z'})
        
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lon=sadcp_out[ri].lon)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(lat=sadcp_out[ri].lat)
    sadcp_out[ri].distance.values = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(sadcp_out[ri].lon.dropna(dim='time').values,  
                                                                                               sadcp_out[ri].lat.dropna(dim='time').values,p=0))))
    sadcp_out[ri] = sadcp_out[ri].rename({'distance': 'x_m'})
    sadcp_out[ri] = sadcp_out[ri].swap_dims({'time': 'x_m'}) 
    new_time2 = pd.to_datetime(np.interp(np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp),
                                         sadcp_out[ri].x_m.values,pd.to_datetime(sadcp_out[ri].time.values).astype(int))) 
    sadcp_out[ri] = sadcp_out[ri].interp(x_m=np.arange(0,dx_sadcp*sadcp_out[ri].x_m.size,dx_sadcp))
    sadcp_out[ri].coords['time'] = sadcp_out[ri].x_m
    sadcp_out[ri]['time'].values = new_time2.values
    sadcp_out[ri] = sadcp_out[ri].rename({'x_m': 'xy'})
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_m=sadcp_out[ri].xy)
    sadcp_out[ri] = sadcp_out[ri].assign_coords(time_secs=sadcp_out[ri].time.astype(int))
    sadcp_out[ri] = sadcp_out[ri].assign_coords(x_km=sadcp_out[ri].x_m/1000)
    sadcp_out[ri] = sadcp_out[ri].set_index(xy=['x_m','x_km','lat','lon','time','time_secs'])
        
    sadcp_outOD['ToCapetown{}'.format(ri+1)] = sadcp_out[ri]    
        
    return scan_sadcp, scanfish_data, sadcp_outOD, sadcp