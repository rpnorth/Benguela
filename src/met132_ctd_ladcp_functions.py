def load_ladcp_data(all_files):
    import os
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    
    for ti,f in zip(range(len(all_files)),all_files):
        ladcp_cast = load_ladcp_csv(f)
        name = os.path.basename(f).split('.')[0]
        st_name = np.array(int(name[-3:]))# (int(name[-3:]))

        # create dataset and combine lat,lon into one dimension with stack
        prof_as_ds = xr.Dataset({'u': (['lon', 'lat', 'z'],  ladcp_cast.u.values[np.newaxis,np.newaxis,:]),
                                 'v': (['lon', 'lat', 'z'], ladcp_cast.v.values[np.newaxis,np.newaxis,:])},
                                coords={'lon': (ladcp_cast.longitude[...,np.newaxis]),
                                        'lat': (ladcp_cast.latitude[...,np.newaxis]),
                                        'station': (st_name[...,np.newaxis]),
                                        'time': (np.array(ladcp_cast.datetime)[...,np.newaxis]),
                                        'z': -ladcp_cast.z.values}).stack(xy = ('lon','lat','station','time'))
        if ti == 0:
            ladcp_data = prof_as_ds
        else:
            ladcp_data = xr.concat([ladcp_data,prof_as_ds],dim=('xy'))

    return ladcp_data

def load_ctd_data(data_files):
    import os
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    
    for ti,f in zip(range(len(data_files)),data_files):
        ctd_cast = load_ctd_csv(f)
        name = os.path.basename(f).split('.')[0]

        # convert to CT,SA,z
        z = gsw.z_from_p(ctd_cast.Pressure,ctd_cast.lat)
        p = ctd_cast.Pressure
        if p.max() < 50:
            continue
        #print(p.max(),p.min(),z.max(),z.min())
        SA = gsw.SA_from_SP(ctd_cast.Salinity,ctd_cast.Pressure,ctd_cast.lon,ctd_cast.lat)
        CT = gsw.CT_from_t(SA,ctd_cast.Temperature,ctd_cast.Pressure)
        RHO = gsw.rho(SA,CT,ctd_cast.Pressure)
        p_ref     = 10.1325; # reference pressure # following scanfish calc
        Pot_dens  = gsw.rho(SA,CT,p_ref); # potential density
        sigma_0   = Pot_dens - 1000; # potential density anomaly

        st_name = np.array(int(name[-3:]))# (int(name[-3:]))

        # 'time','Pressure','Temperature','O2','Salinity'
        # !!! NOTE: time is off by 1 days therefore subtracting 1 when converting to datetime format
        # create dataset and combine lat,lon into one dimension with stack
        prof_as_ds = xr.Dataset({'CT': (['lon', 'lat', 'z'],  CT[np.newaxis,np.newaxis,:]),
                                'SA': (['lon', 'lat', 'z'], SA[np.newaxis,np.newaxis,:]),
                                'RHO': (['lon', 'lat', 'z'], RHO[np.newaxis,np.newaxis,:]),
                                'sigma_0': (['lon', 'lat', 'z'], sigma_0[np.newaxis,np.newaxis,:]),
                                'Temperature': (['lon', 'lat', 'z'],  ctd_cast.Temperature.values[np.newaxis,np.newaxis,:]),
                                'Salinity': (['lon', 'lat', 'z'], ctd_cast.Salinity.values[np.newaxis,np.newaxis,:]),
                                'O2': (['lon', 'lat', 'z'], ctd_cast.O2.values[np.newaxis,np.newaxis,:]),
                                'Pressure': (['lon', 'lat', 'z'], ctd_cast.Pressure.values[np.newaxis,np.newaxis,:])},
                                coords={'lon': (np.array(ctd_cast.longitude)[...,np.newaxis]),
                                        'lat': (np.array(ctd_cast.latitude)[...,np.newaxis]),
                                        'station': (st_name[...,np.newaxis]),
                                        'time': np.array(pd.to_datetime(ctd_cast.time.values[0]-1, unit='D', 
                                                                        origin=pd.Timestamp('01-01-2016')))[...,np.newaxis],
                                        'z': z}).stack(xy = ('lon','lat','station','time'))

        z_1m = np.arange(np.ceil(z.max()),np.floor(z.min()),-1.0) # seem to need exactly the same z to make concat work
        prof_as_ds = prof_as_ds.interp(z=z_1m,method='linear', kwargs={'fill_value':np.nan})

        if ti == 0:
            ctd_data = prof_as_ds
        else:
            ctd_data = xr.concat([ctd_data,prof_as_ds],dim=('xy'))

    return ctd_data

import pandas as pd
class cast_data_format(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, name=None,
                 longitude=None, latitude=None, datetime=None,
                 config=None, dtype=None, copy=False):
        super(cast_data_format, self).__init__(data=data, index=index,
                                  columns=columns, dtype=dtype,
                                  copy=copy)
        self.longitude = longitude
        self.latitude = latitude
        self.datetime = datetime
        self.config = config
        self.name = name

    def __reduce__(self):
        return self.__class__, (
            DataFrame(self),  # NOTE Using that type(data)==DataFrame and the
                              # the rest of the arguments of DataFrame.__init__
                              # to defaults, the constructors acts as a
                              # copy constructor.
            None,
            None,
            self.longitude,
            self.latitude,
            self.datetime,
            self.config,
            self.name,
            None,
            False,
        )

def load_ladcp_csv(filename_in,skiprows=10,headerlines=6):
    # modified from ctd module
    from io import StringIO
    import os
    import pandas as pd
    import numpy as np

    # first get info from file header
    read_header = pd.read_csv(filename_in,sep='\s+',iterator=True,header=None)
    header_info = read_header.get_chunk(headerlines)
    file = header_info.values[0,2]
    date = header_info.values[1,2]
    time = header_info.values[2,2]
    lat = header_info.values[3,2]
    lon = header_info.values[4,2]
    # convert string to float
    lon_lat = np.array([lon,lat]).astype(np.float)
    #from datetime import datetime
    #date_time = datetime.strptime(date+' '+time, '%Y/%m/%d %H:%M:%S')
    #date_time = pd.DatetimeIndex(pd.Series((date+' '+time)))
    date_time = pd.to_datetime((date+' '+time))

    cfile = open(filename_in, 'rb')
    f_text = cfile.read().decode(encoding='utf-8', errors='replace')
    cfile.close()
    f_text2 =  StringIO(f_text)

    name = os.path.basename(filename_in).split('.')[0]

    cast = pd.read_csv(filename_in, sep='\s+', skiprows=skiprows, header=None, names=['z','u','v','err'])
    
    # now convert to desired panda format
    return cast_data_format(cast, longitude=lon_lat[0], latitude=lon_lat[1], datetime=date_time, name=name)

def load_ctd_csv(data_filename,skiprows=0,headerlines=45):
    # modified from ctd module
    from io import StringIO
    import os
    import pandas as pd
    import numpy as np
    # name 0 = latitude: Latitude [deg]
    # name 1 = longitude: Longitude [deg]
    # name 2 = timeJ: Julian Days
    # name 3 = prDM: Pressure, Digiquartz [db]
    # name 4 = t090C: Temperature [ITS-90, deg C]
    # name 5 = sbeox0Mg/L: Oxygen, SBE 43 [mg/l]
    # name 6 = sal00: Salinity, Practical [PSU]

    cfile = open(data_filename, 'rb')
    f_text = cfile.read().decode(encoding='utf-8', errors='replace')
    cfile.close()
    f_text2 =  StringIO(f_text)

    name = os.path.basename(data_filename).split('.')[0]

    cast = pd.read_csv(data_filename, sep='\s+', skiprows=skiprows, header=None, names=['lat','lon','time','Pressure','Temperature','O2','Salinity'])

    # now convert to desired panda format
    return cast_data_format(cast, longitude=cast.lon.mean(), latitude=cast.lat.mean(), datetime=cast.time.mean(), name=name)

def load_combine_ladcp_ctd_data(pathLADCP, pathCTD):
    import os
    import glob
    import gsw
    import xarray as xr
    import pandas as pd
    import numpy as np
    from . import met132_calc_functions as cf

    # load CTD data
    pathCTD = r'/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/CTD_calibrated/Down_Casts/1db_mean/data/'                     # use your path
    data_files = glob.glob(os.path.join(pathCTD, "*.asc"))     # advisable to use os.path.join as this makes concatenation OS independent
    ctd_data = load_ctd_data(data_files)
    ctd_data = ctd_data.sortby('time')

    # load LADCP data
    pathLADCP = r'/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/LADCP/profiles/'                     # use your path
    all_files = glob.glob(os.path.join(pathLADCP, "*.lad"))     # advisable to use os.path.join as this makes concatenation OS independent
    ladcp_data = load_ladcp_data(all_files)
    ladcp_data = ladcp_data.sortby('time')

    # create transects 
    ind_LADCP_section, ind_CTD_section = list((1,1,1,1,1,1,1,1,1)), list((1,1,1,1,1,1,1,1,1))

    ind_LADCP_section[0] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-20T10:00:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-11-21T07:00:00'))
    ind_LADCP_section[1] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-21T14:38:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-11-22T15:08:00'))
#    ind_LADCP_section[1] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(ladcp_data.lon.values > 11.5, ladcp_data.lon.values < 12.5), 
#                                                                                    ladcp_data.lat.values < -26.25),
#                                                      ladcp_data.time.values <= np.datetime64('2016-11-22T15:08:00')),
#                                       ladcp_data.time.values >= np.datetime64('2016-11-21T14:38:00')) 
    # LADCP_CTD_Transect3
    ind_LADCP_section[2] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-24T10:33:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-11-24T23:30:00'))
    
    ind_LADCP_section[3] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-26T08:30:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-11-27T00:01:00'))
#    ind_LADCP_section[3] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(ladcp_data.lon.values > 12.95, ladcp_data.lon.values < 13.25), 
#                                      ladcp_data.lat.values < -25.9),
#                                      ladcp_data.lat.values > -26.25),
#                                      ladcp_data.time.values >= np.datetime64('2016-11-26T08:30:00')),
#                                      ladcp_data.time.values <= np.datetime64('2016-11-27T00:01:00') )
    # LADCP_CTD_Transect5
    ind_LADCP_section[4] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-27T18:22:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-11-28T06:50:00'))
    # no data available for Transect 6
    #ind_LADCP_section[5] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-28T07:30:00'),ladcp_data.time.values <= np.datetime64('2016-11-28T19:00:00'))
    ind_LADCP_section[5] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-11-30T21:05:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-12-01T19:35:00')) 
    ind_true = np.where(ind_LADCP_section[5])[0]
    ind_LADCP_section[5][ind_true[6]] = False # CTD is missing for this station
    ind_LADCP_section[6] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-12-02T21:57:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-12-03T07:35:00'))
    #ind_LADCP_section[6] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(ladcp_data.lon.values > 12.5, ladcp_data.lon.values < 13.1), 
    #                                                    ladcp_data.lat.values < -26.15),
    #                                                    ladcp_data.lat.values > -26.5),
    #                                                    ladcp_data.time.values >= np.datetime64('2016-12-02T21:57:00')),
    #                                                    ladcp_data.time.values <= np.datetime64('2016-12-03T07:35:00')) 
    # LADCP_CTD_Transect9
    ind_LADCP_section[7] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-12-05T13:35:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-12-06T10:20:00')) 
    ind_LADCP_section[8] = np.logical_and(ladcp_data.time.values >= np.datetime64('2016-12-07T10:07:00'),
                                          ladcp_data.time.values <= np.datetime64('2016-12-08T02:35:00')) 

    
    # tried to get all transects, named in "0.MET132..."
    ind_CTD_section[0] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-20T10:00:00'),
                                        ctd_data.time.values <= np.datetime64('2016-11-21T07:00:00'))
                                          
    ind_CTD_section[1] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-21T14:38:00'),
                                        ctd_data.time.values <= np.datetime64('2016-11-22T15:08:00'))
    #ind_CTD_section[1] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(ctd_data.lon.values > 11.5, ctd_data.lon.values < 12.5), 
    #                                                                                ctd_data.lat.values < -26.25),
    #                                                  ctd_data.time.values <= np.datetime64('2016-11-22T15:08:00')),
    #                                   ctd_data.time.values >= np.datetime64('2016-11-21T14:38:00')) 
    # LADCP_CTD_Transect3
    ind_CTD_section[2] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-24T10:33:00'),
                                        ctd_data.time.values <= np.datetime64('2016-11-24T23:30:00'))
    ind_true = np.where(ind_CTD_section[2])[0]
    ind_CTD_section[2][ind_true[5]] = False # LADCP is missing for this station
    ind_CTD_section[3] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-26T08:30:00'),
                                        ctd_data.time.values <= np.datetime64('2016-11-27T00:01:00'))
    #ind_CTD_section[3] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(ctd_data.lon.values > 12.95, ctd_data.lon.values < 13.25), 
    #                                                   ctd_data.lat.values < -25.9),
    #                                                   ctd_data.lat.values > -26.25),
    #                                                   ctd_data.time.values > np.datetime64('2016-11-26T08:30:00')),
    #                                                   ctd_data.time.values < np.datetime64('2016-11-27T00:01:00')) 
    # LADCP_CTD_Transect5
    ind_CTD_section[4] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-27T17:43:00'),
                                        ctd_data.time.values <= np.datetime64('2016-11-28T06:45:00'))
    #ind_CTD_section[5] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-28T07:30:00'),ctd_data.time.values <= np.datetime64('2016-11-28T19:00:00'))
    ind_CTD_section[5] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-11-30T21:05:00'),
                                        ctd_data.time.values <= np.datetime64('2016-12-01T19:35:00')) 
    
    ind_CTD_section[6] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-12-02T21:57:00'),
                                        ctd_data.time.values <= np.datetime64('2016-12-03T07:35:00'))
    #ind_CTD_section[6] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(ctd_data.lon.values > 12.5, ctd_data.lon.values < 13.1), 
    #                                                   ctd_data.lat.values < -26.15),
    #                                                   ctd_data.lat.values > -26.5),
    #                                                   ctd_data.time.values > np.datetime64('2016-12-02T21:57:00')),
    #                                                   ctd_data.time.values < np.datetime64('2016-12-03T07:35:00')) 
    # LADCP_CTD_Transect9
    ind_CTD_section[7] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-12-05T13:35:00'),
                                        ctd_data.time.values <= np.datetime64('2016-12-06T10:20:00')) 
    ind_CTD_section[8] = np.logical_and(ctd_data.time.values >= np.datetime64('2016-12-07T10:07:00'),
                                        ctd_data.time.values <= np.datetime64('2016-12-08T02:35:00')) 

    
    #print(ind_CTD_section[3],ind_LADCP_section[3])
    #print('full:',ladcp_data.time.values,ctd_data.time.values)
    # combine LADCP and CTD into one dataset
    ctd_ladcp = list((1,1,1,1,1,1,1,1,1))
    for ri in range(len(ind_CTD_section)):
        ctd_test = ctd_data.isel(xy=ind_CTD_section[ri])
        ladcp_test = ladcp_data.isel(xy=ind_LADCP_section[ri])
        # no temporal interpolation, because casts are so random

        # get same z coords too, before merging; using ladcp which has bigger spacing
        ctd_test = ctd_test.interp(z=ladcp_test.z)
        
        # time/position of each cast may differ between ladcp and ctd, but referring to the same cast; so set to consistent times/positions
        ctd_test = ctd_test.reset_index('xy') # need to separate out 'time'
        ladcp_temp = ladcp_test.reset_index('xy') # only way I found to get ctd_test to accept ladcp times
        ctd_test['time'] = ladcp_temp.xy.time 
        ctd_test['lon'] = ladcp_temp.xy.lon
        ctd_test['lat'] = ladcp_temp.xy.lat
        ctd_test['station'] = ladcp_temp.xy.station
        ctd_test = ctd_test.set_index(xy=('lon','lat','station','time')) # put back to multi-index
        
        # merge along all matching coords
        ctd_ladcp[ri] = xr.merge([ctd_test,ladcp_test])

        # For consistency with scan_sadcp, make average Pressure dim
        ctd_ladcp[ri]['Pressure_array'] = ctd_ladcp[ri].Pressure
        #ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(Pressure=ctd_ladcp[ri].z) # to get right dims
        ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(Pressure=ctd_ladcp[ri].Pressure_array.mean(dim='xy')) # to get right dims
        #ctd_ladcp[ri]['Pressure'] = ctd_ladcp[ri].Pressure_array.mean(dim='xy')
        #ctd_ladcp[ri] = ctd_ladcp[ri].assigan(Pressure)ctd_ladcp[ri].Pressure_array.mean(dim='xy')
        # and re-order dimensions
        ctd_ladcp[ri] = ctd_ladcp[ri].transpose('xy','z')
        
        # calculate Vorticity, M**2, N**2, and Ri_Balanced
        ctd_ladcp[ri]['across_track_vel'] = ctd_ladcp[ri].u #(ctd_ladcp[ri].u**2+ctd_ladcp[ri].v**2)**0.5
        # need to create DataArray to get coords right
        new_dx_m = gsw.distance(ctd_ladcp[ri].lon.dropna(dim='xy').values,
                                ctd_ladcp[ri].lat.dropna(dim='xy').values,p=0)
        dist = np.cumsum((np.append(np.array(0),new_dx_m)))
        ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(distance=dist)

        # for plotting better if there is a coord option
        ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(x_km=ctd_ladcp[ri].distance/1000)
        ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(x_m=ctd_ladcp[ri].distance)
        # and add as to multi-dimension (nned to reset in order to set it seems)
        ctd_ladcp[ri] = ctd_ladcp[ri].reset_index('xy').set_index(xy=['x_m','x_km','lon','lat','station','time'])
        #dx = ctd_ladcp[ri].x_km.diff('xy').mean().values  !!! Taken care of by xarray
        #ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(x_km_shift=ctd_ladcp[ri].x_km) # for contour plot on pcolormesh
        #dz = ctd_ladcp[ri].z.diff('z').mean().values
        #ctd_ladcp[ri] = ctd_ladcp[ri].assign_coords(z_shift=ctd_ladcp[ri].z + dz/2) # for contour plot on pcolormesh
        
        ctd_ladcp[ri] = cf.calc_N2_M2(ctd_ladcp[ri])
        ctd_ladcp[ri] = cf.calc_vertical_vorticity(ctd_ladcp[ri])
        ctd_ladcp[ri] = cf.calc_Ri_Balanced(ctd_ladcp[ri])
        ctd_ladcp[ri] = cf.SI_GI_Check(ctd_ladcp[ri])
     
    return ctd_ladcp, ctd_data, ladcp_data#, ctd_test, ladcp_test
