import os
import gc
import re
import json
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, mapping
from xml.etree import ElementTree as ET
from icelakes.utilities import get_size


##########################################################################################
def shp2geojson(shapefile, output_directory = 'geojsons/'):
    """
    Convert a shapefile to a geojson polygon file that can be used to 
    subset data from NSIDC. This already simplifies large polygons
    to reduce file size.

    Parameters
    ----------
    shapefile : string
        the path to the shapefile to convert
    output_directory : string
        the directory in which to write the geojson file

    Returns
    -------
    nothing

    Examples
    --------
    >>> shp2geojson_nsidc(my_shapefile.shp, output_directory = 'geojsons/')
    """    
    
    outfilename = shapefile.replace('.shp', '.geojson')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if outfilename[outfilename.rfind('/'):] != -1:
        outfilename = output_directory + outfilename[outfilename.rfind('/')+1:]
    else:
        outfilename = output_directory + outfilename

    gdf = gpd.read_file(shapefile)
    gdf.to_file(outfilename, driver='GeoJSON')
    print('Wrote file: %s' % outfilename)
    return outfilename
    
##########################################################################################    
def make_granule_list(geojson, start_date, end_date, icesheet, meltseason, list_out_name, geojson_dir_local='geojsons/', geojson_dir_remote=None, return_df=True, version=None):
    """
    Query for available granules over a region of interest and a start
    and end date. This will write a csv file with one column being the 
    available granule producer IDs and the other one the path to the 
    geojson file that needs to be used to subset them. 

    Parameters
    ----------
    geojson : string
        the filename of the geojson file
    start_date : string
        the start date in 'YYYY-MM-DD' format
    end_date : string
        the end date in 'YYYY-MM-DD' format
    list_out_name : string
        the path+filename of the csv file to be written out
    geojson_dir_local : string
        the path to the directory in which the geojson file is stored locally
    geojson_dir_remote : string
        the path to the directory in which the geojson file is stashed remotely
        if None (default) it will be the same as the local path

    Returns
    -------
    nothing, writes csv file to path given by list_out_name

    Examples
    --------
    >>> make_granule_list(my_geojson.geojson, '2021-05-01', '2021-09-15', 'auto', 
                          geojson_dir_local='geojsons/', geojson_dir_remote=None)
    """    

    short_name = 'ATL03'
    start_time = '00:00:00'
    end_time = '23:59:59'
    temporal = start_date + 'T' + start_time + 'Z' + ',' + end_date + 'T' + end_time + 'Z'

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

    # Get json response from CMR collection metadata
    params = {'short_name': short_name}
    response = requests.get(cmr_collections_url, params=params)
    results = json.loads(response.content)

    # Find all instances of 'version_id' in metadata and print most recent version number
    if not version:
        versions = [el['version_id'] for el in results['feed']['entry']]
        latest_version = max(versions)
    else:
        latest_version = version
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{latest_version}.xml'

    # read in geojson file
    gdf = gpd.read_file(geojson_dir_local + geojson)
    # poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    # polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    polygon = ','.join([str(c) for xy in zip(*gdf.exterior.loc[0].coords.xy) for c in xy])
    search_params = {'short_name': short_name, 'version': latest_version, 'temporal': temporal, 'page_size': 100,
                     'page_num': 1,'polygon': polygon}

    # query for granules 
    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            break # Out of results, so break out of loop

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    granule_list, idx_unique = np.unique(np.array([g['producer_granule_id'] for g in granules]), return_index=True)
    granules = [g for i,g in enumerate(granules) if i in idx_unique]
    size_mb = [float(result["granule_size"]) for result in granules]
    
    print('Found %i %s version %s granules over %s between %s and %s.' % (len(granule_list), short_name, latest_version, 
                                                                          geojson, start_date, end_date))
    description = [icesheet + '_' + meltseason + '_' + geojson.replace('.geojson','')] * len(granule_list)
    if geojson_dir_remote is None:
        geojson_remote = geojson_dir_local + geojson
    else:
        geojson_remote = geojson_dir_remote + geojson

    thisdf = pd.DataFrame({'granule': granule_list, 
                           'geojson': geojson_remote, 
                           'description': description, 
                           'geojson_clip': geojson_remote.replace('simplified_', ''),
                           'size_mb': size_mb})
    if return_df:
        return thisdf
    else:
        if list_out_name == 'auto':
            list_out_name = 'granule_lists/' + geojson.replace('.geojson', ''), + '_' + start_date[:4] + '.csv'
        thisdf.to_csv(list_out_name, header=False, index=False)
        print('Wrote file: %s' % list_out_name)
    

##########################################################################################
# @profile
def download_granule(granule_id, gtxs, geojson, granule_output_path, uid, pwd, vars_sub='default', spatial_sub=False): 
    """
    Download a single ICESat-2 ATL03 granule based on its producer ID,
    subsets it to a given geojson file, and puts it into the specified
    output directory as a .h5 file. A NASA earthdata user id (uid), and
    the associated password are required. 
    (Can also provide a shapefile instead of geojson.)

    Parameters
    ----------
    granule_id : string
        the producer_granule_id for CMR search
    gtxs : string or list
        the ground tracks to request
        possible values:
            'gt1l' or 'gt1r' or 'gt2l', ... (single gtx)
            ['gt1l', 'gt3r', ...] (list of gtxs)
    geojson : string
        filepath to the geojson file used for spatial subsetting
    granule_output_path : string
        folder in which to save the subsetted granule
    uid : string
        earthdata user id
    pwd : string
        the associated password

    Returns
    -------
    nothing

    Examples
    --------
    >>> download_granule_nsidc(granule_id='ATL03_20210715182907_03381203_005_01.h5', 
                               geojson='geojsons/jakobshavn.geojson', 
                               gtxs='all'
                               granule_output_path='IS2data', 
                               uid='myuserid', 
                               pwd='mypasword')
    """
    print('--> parameters: granule_id = %s' % granule_id)
    print('                gtxs = %s' % gtxs)
    print('                geojson = %s' % geojson)
    print('                granule_output_path = %s' % granule_output_path)
    print('                vars_sub = %s' % vars_sub)
    print('                spatial_sub = %s\n' % spatial_sub)
    
    short_name = 'ATL03'
    version = granule_id[30:33]
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{version}.xml'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
    
    geojson_filepath = str(os.getcwd() + '/' + geojson)
    
    # set the variables for subsetting
    if vars_sub == 'default':
        vars_sub = ['/ancillary_data/atlas_sdp_gps_epoch',
                    '/ancillary_data/calibrations/dead_time/gtx',
                    '/orbit_info/rgt',
                    '/orbit_info/cycle_number',
                    '/orbit_info/sc_orient',
                    '/gtx/geolocation/segment_id',
                    '/gtx/geolocation/ph_index_beg',
                    '/gtx/geolocation/segment_dist_x',
                    '/gtx/geolocation/segment_length',
                    '/gtx/geolocation/segment_ph_cnt',
                    # '/gtx/geophys_corr/dem_h',
                    '/gtx/geophys_corr/geoid',
                    '/gtx/bckgrd_atlas/pce_mframe_cnt',
                    '/gtx/bckgrd_atlas/tlm_height_band1',
                    '/gtx/bckgrd_atlas/tlm_height_band2',
                    '/gtx/bckgrd_atlas/tlm_top_band1',
                    '/gtx/bckgrd_atlas/tlm_top_band2',
                    # '/gtx/bckgrd_atlas/bckgrd_counts',
                    # '/gtx/bckgrd_atlas/bckgrd_int_height',
                    # '/gtx/bckgrd_atlas/delta_time',
                    '/gtx/heights/lat_ph',
                    '/gtx/heights/lon_ph',
                    '/gtx/heights/h_ph',
                    '/gtx/heights/delta_time',
                    '/gtx/heights/dist_ph_along',
                    '/gtx/heights/quality_ph',
                    # '/gtx/heights/signal_conf_ph',
                    '/gtx/heights/pce_mframe_cnt',
                    '/gtx/heights/ph_id_pulse'
                    ]
        if int(version) > 5:
            vars_sub.append('/gtx/heights/weight_ph')
    beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    
    if gtxs == 'all':
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    elif type(gtxs) == str:
        var_list = [v.replace('/gtx','/'+gtxs.lower()) if '/gtx' in v else v for v in vars_sub]
    elif type(gtxs) == list:
        var_list = sum([[v.replace('/gtx','/'+bm.lower()) for bm in gtxs] if '/gtx' in v else [v] for v in vars_sub],[])
    else: # default to requesting all beams
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    
    # search for the given granule
    search_params = {
        'short_name': short_name,
        'page_size': 100,
        'page_num': 1,
        'producer_granule_id': granule_id}

    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            # Out of results, so break out of loop
            break

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1
        
    granule_list, idx_unique = np.unique(np.array([g['producer_granule_id'] for g in granules]), return_index=True)
    granules = [g for i,g in enumerate(granules) if i in idx_unique] # keeps double counting, not sure why
    print('\nDownloading ICESat-2 data. Found granules:')
    if len(granules) == 0:
        print('None')
        return 'none', 404
    for result in granules:
        print('  '+result['producer_granule_id'], f', {float(result["granule_size"]):.2f} MB',sep='')
        
    # Use geopandas to read in polygon file as GeoDataFrame object 
    # Note: a shapefile, KML, or almost any other vector-based spatial data format could be substituted here.
    gdf = gpd.read_file(geojson_filepath)

    # make sure the two regions that go over the date line are adjusted 
    # if ('West_Ep-F.geojson' in geojson_filepath) or ('East_E-Ep.geojson' in geojson_filepath): 
    #     lon180 = np.array(gdf.geometry.iloc[0].exterior.coords.xy[0])
    #     lon180[lon180 < 0] = lon180[lon180 < 0]  + 360
    #     gdf['geometry'] = Polygon(list(zip(lon180, gdf.geometry.iloc[0].exterior.coords.xy[1])))
    #     poly = orient(gdf.loc[0].geometry,sign=1.0)
    #     lon180 = np.array(poly.exterior.coords.xy[0])
    #     # lon180[lon180 >= 180] = lon180[lon180 >= 180] - 360
    #     gdf['geometry'] = Polygon(list(zip(lon180, gdf.geometry.iloc[0].exterior.coords.xy[1])))
    #     poly = gdf.loc[0].geometry
    
    # Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. 
    # The larger the tolerance value, the more simplified the polygon.
    # Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. 
    # The last point should match the first point to close the polygon.
    # poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    # else:
    poly = orient(gdf.loc[0].geometry,sign=1.0)

    geojson_data = gpd.GeoSeries(poly).to_json() # Convert to geojson
    geojson_data = geojson_data.replace(' ', '') #remove spaces for API call
    
    #Format dictionary to polygon coordinate pairs for CMR polygon filtering
    polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    
    print('\nInput geojson:', geojson)
    print('Simplified polygon coordinates based on geojson input:', polygon)
    
    # Create session to store cookie and pass credentials to capabilities url
    session = requests.session()
    s = session.get(capability_url)
    response = session.get(s.url,auth=(uid,pwd))

    try:
        root = ET.fromstring(response.content)
    except:
        try:
            cont = str(request._content)
            print('request status code:', request.status_code)
            the_code = cont[cont.find('<Code>')+6:cont.find('</Code>')]
            if len(the_code) < 1000:
                print(the_code)
            the_message = cont[cont.find('<Message>')+9:cont.find('</Message>')]
            if len(the_message) < 5000:
                print(the_message)
            print('')
            return 'none', response.status_code
        except:
            return 'none', response.status_code

    #collect lists with each service option
    subagent = [subset_agent.attrib for subset_agent in root.iter('SubsetAgent')]
    
    # this is for getting possible variable values from the granule search
    if len(subagent) > 0 :
        # variable subsetting
        variables = [SubsetVariable.attrib for SubsetVariable in root.iter('SubsetVariable')]  
        variables_raw = [variables[i]['value'] for i in range(len(variables))]
        variables_join = [''.join(('/',v)) if v.startswith('/') == False else v for v in variables_raw] 
        variable_vals = [v.replace(':', '/') for v in variables_join]
    
    # make sure to only request the variables that are available
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    if vars_sub == 'all':
        var_list_subsetting = ''
    else:
        var_list_subsetting = intersection(variable_vals,var_list)
    
    if len(subagent) < 1 :
        print('No services exist for', short_name, 'version', latest_version)
        agent = 'NO'
        coverage,Boundingshape,polygon = '','',''
    else:
        agent = ''
        subdict = subagent[0]
        if (subdict['spatialSubsettingShapefile'] == 'true') and spatial_sub:
            ######################################## Boundingshape = geojson_data
            Boundingshape = polygon
        else:
            Boundingshape, polygon = '',''
        coverage = ','.join(var_list_subsetting)
    if (vars_sub=='all') & (not spatial_sub):
        agent = 'NO'
        
    page_size = 100
    request_mode = 'stream'
    page_num = int(np.ceil(len(granules)/page_size))

    param_dict = {'short_name': short_name, 
                  'producer_granule_id': granule_id,
                  'version': version,  
                  'polygon': polygon,
                  'Boundingshape': Boundingshape,  
                  'Coverage': coverage, 
                  'page_size': page_size, 
                  'request_mode': request_mode, 
                  'agent': agent, 
                  'email': 'yes'}

    #Remove blank key-value-pairs
    param_dict = {k: v for k, v in param_dict.items() if v != ''}

    #Convert to string
    param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
    param_string = param_string.replace("'","")

    #Print API base URL + request parameters
    endpoint_list = [] 
    for i in range(page_num):
        page_val = i + 1
        API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
        endpoint_list.append(API_request)

    print('\nAPI request URL:')
    print(*endpoint_list, sep = "\n") 
    
    # Create an output folder if the folder does not already exist.
    path = str(os.getcwd() + '/' + granule_output_path)
    if not os.path.exists(path):
        os.mkdir(path)

    # Different access methods depending on request mode:
    for i in range(page_num):
        page_val = i + 1
        print('\nOrder: ', page_val)
        print('Requesting...')
        request = session.get(base_url, params=param_dict)
        print('HTTP response from order response URL: ', request.status_code)
        # try: 
        #     cont = str(request._content)
        #     print(cont[cont.find('<Code>')+6:cont.find('</Code>')],
        #       '(', cont[cont.find('<Message>')+9:cont.find('</Message>')], ')\n')
        # except:
        #     pass
        request.raise_for_status()
        d = request.headers['content-disposition']
        fname = re.findall('filename=(.+)', d)
        dirname = os.path.join(path,fname[0].strip('\"'))
        print('Downloading...')
        open(dirname, 'wb').write(request.content)
        print('Data request', page_val, 'is complete.')

    # Unzip outputs
    for z in os.listdir(path): 
        if z.endswith('.zip'): 
            zip_name = path + "/" + z 
            zip_ref = zipfile.ZipFile(zip_name) 
            zip_ref.extractall(path) 
            zip_ref.close() 
            os.remove(zip_name) 

    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))
            
    print('\nUnzipped files and cleaned up directory.')
    print('Output data saved in:', granule_output_path)
    
    filelist = [granule_output_path+'/'+f for f in os.listdir(granule_output_path) \
                if os.path.isfile(os.path.join(granule_output_path, f)) & (granule_id in f)]
    
    if len(filelist) == 0: 
        return 'none'
    else:
        filename = filelist[0]
    print('File to process: %s (%s)' % (filename, get_size(filename)))
    
    print(filename, 'status:', request.status_code)
    return filename, request.status_code


##########################################################################################
def print_granule_stats(photon_data, bckgrd_data, ancillary, outfile=None):
    """
    Print stats from a read-in granule.
    Mostly for checking that things are working / OSG testing. 

    Parameters
    ----------
    photon_data : dict of pandas dataframes
        the first output of read_atl03()
    bckgrd_data : dict of pandas dataframes
        the second output of read_atl03()
    ancillary : dict
        the third output of read_atl03()
    outfile : string
        file path and name for the output
        if outfile=None, results are printed to stdout

    Returns
    -------
    nothing
                                    
    Examples
    --------
    >>> print_granule_stats(photon_data, bckgrd_data, ancillary, outfile='stats.txt')
    """    

    if outfile is not None: 
        import sys
        original_stdout = sys.stdout
        f = open(outfile, "w")
        sys.stdout = f

    print('\n*********************************')
    print('** GRANULE INFO AND STATISTICS **')
    print('*********************************\n')
    print(ancillary['granule_id'])
    print('RGT:', ancillary['rgt'])
    print('cycle number:', ancillary['cycle_number'])
    print('spacecraft orientation:', ancillary['sc_orient'])
    print('beam configuation:')
    for k in ancillary['gtx_beam_dict'].keys():
        print(' ', k, ': beam', ancillary['gtx_beam_dict'][k], '(%s)'%ancillary['gtx_strength_dict'][k])
    for k in photon_data.keys():
        counts = photon_data[k].count()
        nanvals = photon_data[k].isna().sum()
        maxs = photon_data[k].max()
        mins = photon_data[k].min()
        print('\nPHOTON DATA SUMMARY FOR BEAM %s'%k.upper())
        print(pd.DataFrame({'count':counts, 'nans':nanvals, 'min':mins, 'max':maxs}))
        
    if outfile is not None:
        f.close()
        sys.stdout = original_stdout
        with open(outfile, 'r') as f:
            print(f.read())
    return


##########################################################################################
class edc:
    u = b"<paste your encrypted user id here>"
    p = b'<paste your encrypted password here>'

    # print setup info if values have not been changed
    if u == b"<paste your encrypted user id here>":
        print('\n WARNING: YOU NEED TO ENCRYPT YOUR NASA EARTHDATA CREDENTIALS TO DOWNLOAD ICESAT-2 DATA!\n')
        print('  - place your private/public keys in misc/test1 and misc/test2.')
        print("  - use icelakes/utilities/encedc(uid, pwd, misc/test1, misc/test2) to encrypt your credentials")
        print("  - paste the results in class edc in icelakes/nsidc")
        print("  - the script will call icelakes/utilities/decedc to decrypt credentials")
        print("  - when moving your code, keep your private key in a safe place (do not push to github, etc!)\n")