# author: Philipp Arndt, UC San Diego / Scripps Institution of Oceanography
# 
# intended for use on OSG OSPool, called in run_py.sh, which is called in a submit file 
# submit file is based on a granule list queried locally in make_granule_list.ipynb 
# see examples for submit files in: HTCondor_submit/ 
# see examples for granule lists in:  granule_lists/
# 
# run locally with: 
# $ conda activate icelakes-env
# $ python3 detect_lakes.py --granule <granule_producer_id> --polygon geojsons/<polygon_name.geojson>
# 
# a call that returns a bunch of lakes
# $ python3 detect_lakes.py --granule ATL03_20220714010847_03381603_006_02.h5 --polygon geojsons/simplified_GRE_2000_CW.geojson
# one with more than 100 lakes
# $ python3 detect_lakes.py --granule ATL03_20210729062325_05441205_006_01.h5 --polygon geojsons/simplified_GRE_2000_SW.geojson

import argparse
import os
import gc
import sys
import time
import pickle
import subprocess
import traceback
import numpy as np
import icelakes
from icelakes.utilities import encedc, decedc, get_size
from icelakes.nsidc import download_granule, edc
from icelakes.detection import read_atl03, detect_lakes, melt_lake

parser = argparse.ArgumentParser(description='Test script to print some stats for a given ICESat-2 ATL03 granule.')
parser.add_argument('--granule', type=str, default='ATL03_20220714010847_03381603_006_02.h5',
                    help='The producer_id of the input ATL03 granule')
parser.add_argument('--polygon', type=str, default='geojsons/simplified_GRE_2000_CW.geojson',
                    help='The file path of a geojson file for spatial subsetting') # geojsons/west_greenland.geojson
parser.add_argument('--is2_data_dir', type=str, default='IS2data',
                    help='The directory into which to download ICESat-2 granules')
parser.add_argument('--download_gtxs', type=str, default='all',
                    help='String value or list of gtx names to download, also accepts "all"')
parser.add_argument('--out_data_dir', type=str, default='detection_out_data',
                    help='The directory to which to write the output data')
parser.add_argument('--out_plot_dir', type=str, default='detection_out_plot',
                    help='The directory to which to write the output plots')
parser.add_argument('--out_stat_dir', type=str, default='detection_out_stat',
                    help='The directory to which to write the granule stats')
args = parser.parse_args()

print('\npython args:', args, '\n')

# shuffling files around for HTCondor
print('Shuffling files around for HTCondor...')
for thispath in (args.is2_data_dir, args.out_data_dir, args.out_plot_dir):
    if not os.path.exists(thispath): os.makedirs(thispath)

# download the specified ICESat-2 data from NSIDC
try_nr = 1
request_status_code = 0
while (request_status_code != 200) & (try_nr <= 50):
    try:
        print('Downloading granule from NSIDC. (try %i)' % try_nr)
        input_filename, request_status_code = download_granule(
            args.granule, 
            args.download_gtxs, 
            args.polygon, 
            args.is2_data_dir, 
            decedc(edc().u), 
            decedc(edc().p), 
            spatial_sub=True
        )
        if request_status_code != 200:
            print('  --> Request unsuccessful (%i), trying again in a minute...\n' % request_status_code)
            time.sleep(60)
            try_nr += 1
        
    except:
        print('  --> Request unsuccessful (error raised in code), trying again in a minute...\n')
        traceback.print_exc()
        time.sleep(60)
        try_nr += 1

# perform a bunch of checks to make sure everything went alright with the nsidc api
print('Request status code:', request_status_code, request_status_code==200)
if request_status_code == 200:
    print('NSIDC API request was successful!')
if request_status_code != 200:
    print('NSIDC API request failed. (Request status code: %i)' % request_status_code)
    sys.exit(127)
if request_status_code==200:
    with open('success.txt', 'w') as f: print('we got some sweet data', file=f)
    if input_filename == 'none': 
        print('no granule found. nothing more to do here.') 
        sys.exit(69)
if os.path.exists(input_filename):
    # if os.path.getsize(input_filename) < 31457280:# 30 MB
    if os.path.getsize(input_filename) < 1000000: # 2 MB
        print('granule seems to be empty. nothing more to do here.') 
        sys.exit(69)

gtx_list, ancillary = read_atl03(input_filename, gtxs_to_read='none')

# detect melt lakes
lake_list = []
granule_stats = [0,0,0,0]

for gtx in gtx_list:
    try:
        lakes_found, gtx_stats = detect_lakes(input_filename, gtx, args.polygon, verbose=False)
        for i in range(len(granule_stats)): granule_stats[i] += gtx_stats[i]
        lake_list += lakes_found
        del lakes_found, gtx_stats
        gc.collect()
    except:
        print('Something went wrong for %s' % gtx)
        traceback.print_exc()

try:
    if granule_stats[0] > 0:
        with open('success.txt', 'w') as f: print('we got some useable data from NSIDC!!', file=f)
        print('Sucessfully got some useable data from NSIDC!!')
except:
    traceback.print_exc()
    
# print stats for granule
try:
    print('\nGRANULE STATS (length total, length lakes, photons total, photons lakes):%.3f,%.3f,%i,%i\n' % tuple(granule_stats))
except:
    traceback.print_exc()

try:
    max_lake_length = 20000 # meters (there are no lakes >20km and it's usually where something went wrong over the ocean)
    lake_list[:] = [lake for lake in lake_list if (lake.photon_data.xatc.max()-lake.photon_data.xatc.min()) <= max_lake_length]
except:
    traceback.print_exc()

# for each lake call the surrf algorithm for depth determination
# if it fails, just skip the lake, but print trackeback for the logs 
print('---> determining depth for each lake')
for i, lake in enumerate(lake_list):
    try: 
        lake.surrf()
        print('   --> %3i/%3i, %s | %8.3fN, %8.3fE: %6.2fm deep / quality: %8.2f' % (i+1, len(lake_list), lake.gtx, lake.lat, 
                                                                                 lake.lon, lake.max_depth, lake.lake_quality))
    except:
        print('Error for lake %i (detection quality = %.5f) ... skipping:' % (i+1, lake.detection_quality))
        traceback.print_exc()
        lake.lake_quality = 0.0

# remove zero quality lakes
# lake_list[:] = [lake for lake in lake_list if lake.lake_quality > 0]

# for each lake 
for i, lake in enumerate(lake_list):
    try:
        lake.lake_id = '%s_%s_%s_%04i' % (lake.polygon_name, lake.granule_id[:-3], lake.gtx, i)
        filename_base = 'lake_%05i-%05i_%s_%s_%s' % (np.clip(1000-lake.lake_quality,0,None)*10, 
                                                     np.clip(1.0-lake.detection_quality,0,1)*1e4,
                                                     lake.ice_sheet, lake.melt_season, 
                                                     lake.lake_id)
        figname = args.out_plot_dir + '/%s.jpg' % filename_base
        h5name = args.out_data_dir + '/%s.h5' % filename_base
        
        # plot each lake and save to image
        try:
            fig = lake.plot_lake(closefig=True)
            if fig is not None: fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
        except:
            print('Could not make figure for lake <%s>' % lake.lake_id)
            traceback.print_exc()
        
        # export each lake to h5 and pickle
        try:
            datafile = lake.write_to_hdf5(h5name)
            print('Wrote data file: %s, %s' % (datafile, get_size(datafile)))
        except:
            print('Could not write hdf5 file <%s>' % lake.lake_id)
            traceback.print_exc()

        # only keep files where it was possible to both write the figure and the data file
        if os.path.isfile(figname) and (not os.path.isfile(h5name)):
            os.remove(figname)
        if os.path.isfile(h5name) and (not os.path.isfile(figname)):
            os.remove(h5name)
            
    except:
        traceback.print_exc()

try:
    statsfname = args.out_stat_dir + '/stats_%s_%s.csv' % (args.polygon[args.polygon.rfind('/')+1:].replace('.geojson',''),
                                                           args.granule.replace('.h5',''))
    stats = [args.polygon[args.polygon.rfind('/')+1:].replace('simplified_', ''), args.granule]
    stats += granule_stats
    with open(statsfname, 'w') as f: print('%s,%s,%.3f,%.3f,%i,%i' % tuple(stats), file=f)
except:
    print("could not write stats file")
    traceback.print_exc()
    
# clean up the input data
os.remove(input_filename)

print('\n-------------------------------------------------')
print(  '----------->   Python script done!   <-----------')
print(  '-------------------------------------------------\n')