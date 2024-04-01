#!/bin/bash

while getopts c:n: flag
do
    case "${flag}" in
        c) clusterid=${OPTARG};;
        n) clustername=${OPTARG};;
    esac
done

# condor_q overview, and check for unique hold resons
condor_q $clusterid
condor_q $clusterid -held -af HoldReason | wc -l | xargs printf "%s total jobs on hold\n"
echo "Hold reasons:"
condor_q $clusterid -held -af HoldReason | cut -c-11 | sort | uniq -c
printf "\n\n"

# define hold reasons to grep for (might need to add others if they come up)
declare -a HoldReasons=(
 "memory usage exceeded request_memory"
 "The job (shadow) restarted too many times"
 "The job restarted too many times"
 "Transfer input files failure"
 "Job was held manually."
 "The job exceeded allowed execute duration"
)

filename_out=held_jobs/$clustername-$clusterid.csv
>| $filename_out
# print number of jobs that fulfill the 
totalcounter=0
for holdreason in "${HoldReasons[@]}"; do
    counts=$(condor_q $clusterid -hold | grep "$holdreason" | wc -l);
    printf "%5i - $holdreason\n" $counts;
    totalcounter=$(expr $totalcounter + $counts);
    condor_q $clusterid -held | grep "$holdreason" | awk -F'[. ]+' '{printf "logs/*-%s_*-%s.log*\n",$1,$2;}' | while read x; do ls $x | xargs printf "%s," >> $filename_out; echo "$holdreason" >> $filename_out; done  
done

printf "%5i - TOTAL\n\n" $totalcounter;
printf "\n%s\n" $filename_out
cat $filename_out | wc -l | xargs printf "%s lines in file\n\n"
