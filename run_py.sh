#!/bin/bash

# move input files
mkdir icelakes
mkdir misc
mkdir geojsons
mkdir detection_out_data
mkdir detection_out_plot
mkdir detection_out_stat
mv __init__.py icelakes/
mv utilities.py icelakes/
mv nsidc.py icelakes/
mv detection.py icelakes/
mv test1 misc/
mv test2 misc/
mv *.geojson geojsons/
# # rm success.txt

# just to not send API requests to NSDIC at once
sleep $((RANDOM % 3))

# Run the Python script 
echo "Executing python script:"
echo "python3 detect_lakes.py --granule $1 --polygon $2"
python3 detect_lakes.py --granule $1 --polygon $2

if [ -f "success.txt" ]; then
    echo "Success!!!!!"
    exit 69
else
    echo "No success....."
    # echo "No success for $1 $2" > $_CONDOR_WRAPPER_ERROR_FILE
    exit 127
fi