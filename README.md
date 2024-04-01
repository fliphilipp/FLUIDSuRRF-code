# icelakes-methods

**A repository for:** *"A framework for Automatic Supraglacial Lake Detection and Depth Retrieval From ICESat-2 Photon Data on The Ice Sheets, Using Distributed High-Throughput Computing*

This documents the methodology for a fully automated and scalable algorithm for lake detection and depth determination from ICESat-2 data, and provide a framework for its large-scale implementation using distributed high-throughput computing. We also report resulting depth estimates over Central West Greenland and the Amery Ice Shelf catchment during two melt seasons each, where our method reveals a total of 1248 lakes up to 25 meters deep.

## Some results for the Central West Greenland study region
for a high-melt (2019) and low-melt (2020) boreal summer
![result map for two melt seasons in Central West Greenland](plots/results_map_greenland_cw.jpg)

## Some results for the Amery Catchment (Antarctica) study region
for a high-melt (2018-19) and low-melt (2020-21) austral summer
![result map for two melt seasons in Central West Greenland](plots/results_map_amery.jpg)

## Workflow: 
- Get a user account on the [OSG Open Science Pool](https://osg-htc.org/services/open_science_pool) (free for US-based researchers and their collaborators), or another high-throughput computing platform running HTCondor, that fulfills computing needs
- If the [singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html#) container image `icelake-container_v1.sif` is not available, follow singularity install and container build instructions to build the container image from `singularity_container/icelake-container_v1.def` (need administrator privileges, so this step needs to be done locally). Alternatively, define your own container that fulfills the requirements.
- Copy the singularity container image into stash, for OSG this would be something like `osdf:///ospool/<your_access_point>/data/<your_username>/containers/icelake-container_v1.sif`
- To be able to access ICESat-2 data, [create a NASA Earthdata user account](https://urs.earthdata.nasa.gov/). In this code, the class `icelakes.nsidc.edc` is used to read in credentials, but there are plenty of options to store credentials, they just need to be accessible to the jobs running on the OSG Pool. This can be changed directly in the main python script `detect_lakes.py`, where credentials are passed to `icelakes.nsidc.download_granule()`.
- Based on the drainage basin regions (stored in `geojsons/`), make granule lists for each batch of data using the notebook `make_granule_list.ipynb` (choose combination of dates / melt seasons and basins). Basin definition from source ([Antarctica](https://nsidc.org/data/nsidc-0709/versions/2)/[Greenland](https://datadryad.org/stash/dataset/doi:10.7280/D1WT11)) is detailed in `make_basins.ipynb`.
- Create a Condor submit file for job submission, similar to the ones in `HTCondor_submit/`. Requesting 16 GB of both disk and memory will be sufficient for the vast majority of jobs.
  - Make sure each batch does not exceed 20K jobs, and you don’t submit more than 100K jobs at once.
- Before submitting the first cluster *make sure you have saved any of the output files that you want to keep from previous runs*, and then clear the directories for output file, outs, logs, errs with `rm detection_out_data/* detection_out_stat/* detection_out_plot/* logs/* errs/* outs/*`
- Submit the job with `condor_submit HTCondor_submit/<your_submit_file>.submit`. [Monitor and troubleshoot jobs](https://portal.osg-htc.org/documentation/htc_workloads/submitting_workloads/monitor_review_jobs/) with condor_q and related commands.
- Troubleshoot jobs that go on hold with condor_q -held and the log files, common reasons seem to be:
  - *The job (shadow) restarted too many times*: This can happen mostly when the NSIDC download requests fail (for whichever reason) and return a bad status. If it looks like this is the reason, try re-submitting those held jobs in a new cluster.
  - *Input file transfer failed*: This seemed to be a problem with the job node’s access to the container image on stash/osdf. Re-submit these jobs in a new cluster, if the problem persists use a new version of the container / force osdf to write a new file.
  - *memory usage exceeded request_memory*: This should only be a small fraction of jobs. Re-submit these few jobs in a new cluster with higher memory/disk requests. (Unfortunately there is no good way to predict in advance how much memory a job will take up because it depends on the final subsetted size of the granuled delivered by the NSIDC API. However, 32 GB should do for all granules that I’ve seen so far.)
- Keep track of the jobs that failed (and associated granule/geojson combinations) and get them all done eventually
  - When no more jobs are idle/running, run `bash getheld.sh -c <cluster_id> -n <descriptive_cluster_name>` (example: `bash getheld.sh -c 114218 -n antarctica-18-21-run1`)
    - Make sure all jobs are accounted for. If not, change script to allow for any other hold reasons.
  - Copy the file output to `hold_lists/` with a `final_` prefix (just to make sure to not overwrite them accidentally). You can now remove the cluster from the pool with `condor_rm <cluster_id>`
  - Use `resubmit_make_list.ipynb` to create a new granule list with the held jobs
    - Keep track of the few jobs that ran out of memory, to run later with a higher `request_memory` requirement
    - Other jobs (usually re-started to often because of NSIDC API issues, or sometimes input file transfer failure) can be re-submitted to the pool, using a new submit file with the list generated in `resubmit_make_list.ipynb`
  - If all fails, maybe run a handful of jobs locally and troubleshoot??? / Or skip if something seems awfully off with the data…
- Transfer back output data to a local computer with `scp`. (ideally zip the entire directory and transfer that, if you don't need the log/err/out files anymore this is sped up substantially by removing those first, also helps to concatenate all the granule stats before) 
- There might be some weird false positives, mostly from sensor saturation artifacts over a few regions where the basins overlap with the ocean - browse through the quicklook images and manually remove files that are clearly not lakes.


## Useful commands for OSG:

Login to access node with SSH Keys set up
([Generate SSH Keys and Activate Your OSG Login](https://support.opensciencegrid.org/support/solutions/articles/12000027675)):
```
ssh <username>@<osg-login-node>
```
Example:
```
ssh fliphilipp@login05.osgconnect.net
```

Submit a file to HTCondor:
```
condor_submit <my_submit-file.submit>
```

Watch the progress of the queue after submitting jobs:
```
watch condor_q
```

See which jobs are on hold and why:
```
condor_q -hold
```

Release and re-queue jobs on hold:
```
condor_release <cluster ID>/<job ID>/<username>
```

Remove jobs on hold:
```
condor_rm <cluster ID>/<job ID>/<username>
```

Example to grep log files for memory/disk usage:
```
grep -A 3 'Partitionable Resources' <log_directory>/*.log
```

Put a container in /public stash:
```
scp <mycontainer>.sif fliphilipp@login05.osgconnect.net:/public/fliphilipp/containers/
```

Explore a container on access node:
```
singularity shell --home $PWD:/srv --pwd /srv --scratch /var/tmp --scratch /tmp --contain --ipc --pid /public/fliphilipp/containers/<mycontainer>.sif
```

## To get Singularity working with root privileges:

Get some required packages: 
```
$ sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup
```
    
Remove any previous intallation of Go, if needed: 
```
$ rm -rf /usr/local/go
```

Download Go and untar: 
```
$ wget https://go.dev/dl/go1.19.linux-amd64.tar.gz
$ sudo tar -C /usr/local -xzf go1.19.linux-amd64.tar.gz
```

Add to path and check installation of Go: 
```
$ echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
$ go version
```

Need glibc for Singularity install:
```
$ sudo apt-get install -y libglib2.0-dev
```

Download Singularity and untar:
```
$ wget https://github.com/sylabs/singularity/releases/download/v3.10.2/singularity-ce-3.10.2.tar.gz
$ tar -xzf singularity-ce-3.10.2.tar.gz
```

Move to the directory and run installation commands:
```
$ ./mconfig
$ make -C builddir
$ sudo make -C builddir install
```

