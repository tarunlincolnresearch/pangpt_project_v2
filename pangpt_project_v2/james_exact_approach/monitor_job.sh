#!/bin/bash

echo "======================================"
echo "YOUR CURRENT JOBS"
echo "======================================"
squeue -u tgangil -o "%.18i %.12j %.8T %.10M %.10L %.6D %R"

echo ""
echo "======================================"
echo "RECENT JOB HISTORY (Last 7 days)"
echo "======================================"
sacct -u tgangil --starttime $(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName%20,Partition,State,Elapsed,Start,End -X

echo ""
echo "======================================"
echo "LATEST LOG FILES"
echo "======================================"
ls -lht logs/*.out logs/*.err 2>/dev/null | head -10

echo ""
echo "======================================"
echo "USEFUL COMMANDS"
echo "======================================"
echo "View live output:     tail -f logs/james_full_<JOBID>.out"
echo "View live errors:     tail -f logs/james_full_<JOBID>.err"
echo "Cancel a job:         scancel <JOBID>"
echo "Cancel all jobs:      scancel -u tgangil"
echo "Job details:          scontrol show job <JOBID>"
echo "GPU availability:     sinfo -p gpu"
echo "======================================"