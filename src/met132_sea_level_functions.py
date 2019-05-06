import xarray as xr
import numpy as np

def coriolis(lat):  
    """Compute the Coriolis parameter for the given latitude:
    ``f = 2*omega*sin(lat)``, where omega is the angular velocity 
    of the Earth.
    
    Parameters
    ----------
    lat : array
      Latitude [degrees].
      
    from: http://www.meteo.mcgill.ca/~huardda/amelie/geowind.py copied and modified by rpn 23.5.2019
    Output: coriolis parameter in s^-1
    """
    deg2rad = np.pi/180
    omega   = 7.2921159e-05  # angular velocity of the Earth [rad/s]
    #return 2*omega*np.sin(lat/360.*2*np.pi)
    return 2*omega*np.sin(lat*deg2rad) 

def import_gos_sla_adt_data(year='2016',month='12',days=['01','02','03','04','05']):
    """ import Absolute geostrophic velocity calculated from sea level height
    source: https://cds.climate.copernicus.eu
    NOTE: doesn't work well with only 1 day, try at least 2 days
    """
    import cdsapi
    c = cdsapi.Client()
    c.retrieve('satellite-sea-level-global',
        {'variable':'all', 'year':year, 'month':month, 
         'day':days,'format':'tgz'},
        '/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/Remote_Sensing/201612_dataset-satellite-sea-level-global.tar.gz')
    #c.retrieve('satellite-sea-level-global',
    #    {'variable':'all', 'year':'2016', 'month':'11', 
    #     'day':['17','18','19','20','21','22','23','24','25','27','28','29','30'],'format':'tgz'},
    #    '/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/Remote_Sensing/201611_dataset-satellite-sea-level-global.tar.gz')

def dll_dist(dlon, dlat, lon, lat):
    """Converts lat/lon differentials into distances in meters

    PARAMETERS
    ----------
    dlon : xarray.DataArray longitude differentials
    dlat : xarray.DataArray latitude differentials
    lon  : xarray.DataArray longitude values
    lat  : xarray.DataArray latitude values

    RETURNS
    -------
    dx  : xarray.DataArray distance inferred from dlon
    dy  : xarray.DataArray distance inferred from dlat
    """

    distance_1deg_equator = 111000.0
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * distance_1deg_equator
    dy = ((lon * 0) + 1) * dlat * distance_1deg_equator
    return dx, dy

def load_gos_data(gos_filenames):
    #import xgcm
    from xgcm import Grid
    from xgcm.autogenerate import generate_grid_ds
    # ====== load in all .nc files and combine into one xarray dataset
    gos_map = xr.open_mfdataset(gos_filenames) 
    gos_map = gos_map.rename({'latitude': 'lat'}).rename({'longitude': 'lon'})
    gos_select = gos_map #.sel(time='2016-11-19',lon=slice(10,16),lat=slice(-28,-24))
    #gos_map.ugos
    #dx = gos_map.lon.diff('lon')
    #gos_map['rel_vort'] = gos_map.vgos.diff('lon')/gos_map.lon.diff('lon')

    #gos_select = gos_map #gos_map.sel(time='2016-11-19',lon=slice(10,16),lat=slice(-28,-24))
    # create grid for interpolation, differencing
    #grid = xgcm.Grid(gos_select)
    # for Satellite data:
    # https://xgcm.readthedocs.io/en/latest/autogenerate_examples.html
    ds_full = generate_grid_ds(gos_select, {'X':'lon', 'Y':'lat'})
    ds_full.vgos

    grid = Grid(ds_full, periodic=['X'])

    # compute the difference (in degrees) along the longitude and latitude for both the cell center and the cell face
    # need to specify the boundary_discontinutity in order to avoid the introduction of artefacts at the boundary
    dlong = grid.diff(ds_full.lon, 'X', boundary_discontinuity=360)
    dlonc = grid.diff(ds_full.lon_left, 'X', boundary_discontinuity=360)
    #dlonc_wo_discontinuity = grid.diff(ds_full.lon_left, 'X')
    dlatg = grid.diff(ds_full.lat, 'Y', boundary='fill', fill_value=np.nan)
    dlatc = grid.diff(ds_full.lat_left, 'Y', boundary='fill', fill_value=np.nan)

    # converted into approximate cartesian distances on a globe.
    ds_full.coords['dxg'], ds_full.coords['dyg'] = dll_dist(dlong, dlatg, ds_full.lon, ds_full.lat)
    ds_full.coords['dxc'], ds_full.coords['dyc'] = dll_dist(dlonc, dlatc, ds_full.lon, ds_full.lat)

    # Relative vorticity: ζ = ∂ v/∂ x – ∂ u/∂ y
    ds_full['dv_dx'] = grid.diff(ds_full.vgos, 'X') / ds_full.dxg
    ds_full['du_dy'] = grid.diff(ds_full.ugos, 'Y', boundary='fill', fill_value=np.nan)/ ds_full.dyg
    dv_dx = grid.interp(ds_full['dv_dx'],'Y', boundary='fill', fill_value=np.nan   ) # get dv_dx and du_dy on same grid
    du_dy = grid.interp(ds_full['du_dy'],'X', boundary='fill', fill_value=np.nan   )
    ds_full['Rel_Vort'] = dv_dx-du_dy

    # Vorticity Rossby Number = ζ / f
    ds_full['Ro'] = ds_full.Rel_Vort/coriolis(ds_full.Rel_Vort.lat_left)

    return ds_full


#def import_PO.DAAC Drive_sst_data(sst_filenames):
#    # code source:  #https://github.com/nasa/podaacpy/blob/master/examples/Using%20podaacpy%20to%20interact%20with%20PO.DAAC%20Drive.#ipynb
#    # data example: https://podaac-#tools.jpl.nasa.gov/drive/files/OceanTemperature/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/2016/011
    
    
    
    
##################
# Imports        #
##################
## import the podaac package
#import podaac.podaac as podaac
## import the podaac_utils package
#import podaac.podaac_utils as utils
## import the mcc package
#import podaac.mcc as mcc
#from podaac import drive as drive
#######################
# Class instantiation #
#######################
## then create an instance of the Podaac class
#p = podaac.Podaac()
## then create an instance of the PodaacUtils class
#u = utils.PodaacUtils()
## then create an instance of the MCC class
#m = mcc.MCC()
#d = drive.Drive('podaac.ini',None,None)

#result = p.granule_search(dataset_id='PODAAC-GHGMR-4FJ04',
#                          start_time='2016-11-17T00:00:01Z',
#                          end_time='2016-11-17T11:59:59Z',
#                          bbox='-81,28,-67,40')

#searchStr = 'totalResults'
#numResultsStr = [ str(i) for i in result.strip().split() if searchStr in i ]
#print(numResultsStr)
##Here's the actual granule names
#print(u.mine_granules_from_granule_search(granule_search_response=str(result)))
##Now we simply need to reproduce the Drive URL's for the above granules.
#granules = d.mine_drive_urls_from_granule_search(granule_search_response=(str(result)))
#print(granules)
##retrieve these granules from PO.DAAC Drive.
##Note that the download_granules function actually decompresses
##and removes the compressed archive files locally for us.
#folder_name =  '/Users/North/Drive/Work/UniH_Work/DataAnalysis/Data/MET_132/Remote_Sensing/'
#d.download_granules(granule_collection=granules, path='.')










#print(p.dataset_variables(dataset_id='PODAAC-GHGMR-4FJ04'))#

#print(p.granule_metadata(dataset_id='PODAAC-GHGMR-4FJ04'), granule_name='20160111090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')


#from IPython.display import Image
#from IPython.core.display import HTML 
#result = p.granule_preview(dataset_id='PODAAC-GHGMR-4FJ04')



#from podaac import l2ss as l2ss
#l = l2ss.L2SS()
#granule_id = '20161117090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
#query = {
#        "email": "ryan.peter.north@uni-hamburg.de",
#        "query":
#        [
#            {
#                "urs_username" : "ryanpeternorth",
#                "urs_password" : "lB8d@fmUhvkEDhIWrRn",
#                "webdav_url" : "https://podaac-tools.jpl.nasa.gov/drive/files",
#                "compact": "true",
#                "datasetId": "PODAAC-GHGMR-4FJ04",
#                "bbox": "8,-30,20,-17",
#                "variables": ['lat', 'lon', 'time', 'sea_surface_temperature', 'sst_dtime', 'rejection_flag'],
#                "granuleIds": [granule_id]
#            }
#        ]
#    }
#l.granule_download(query_string=query)


