'''
Automatically schedules
1 sequential task per node on 
1 gpu 
2 cpus-per-task
1 task

There are 24 splits with 20 files each
'''
import os
import time
import numpy as np

def run_last_subreddit(): 
    for i in range(1, 20): 
        letter = chr(ord('`')+i)
        job_contents = '#!/bin/bash\n' + \
            '# Job name:\n' + \
            '#SBATCH --job-name=sr_' + letter + '\n' + \
            '# Partition:\n' + \
            '#SBATCH --partition=savio2_gpu\n' + \
            '# Wall clock limit:\n' + \
            '#SBATCH --time=3-00:00:00\n' + \
            '#SBATCH --nodes=1\n' + \
            '#SBATCH --ntasks=1\n' + \
            '#SBATCH --cpus-per-task=2\n' + \
            '#SBATCH --gres=gpu:1\n' + \
            '#SBATCH --mail-user=lucy3_li@berkeley.edu\n' + \
            '#SBATCH --mail-type=END,FAIL\n' + \
            '#SBATCH --account=fc_dbamman\n' + \
            'source /global/scratch/lucy3_li/anaconda3/bin/activate /global/scratch/lucy3_li/anaconda3/envs/wsi\n' + \
            'time python match_reddit.py tofreda' + letter + ' \n' + \
            'source /global/scratch/lucy3_li/anaconda3/bin/deactivate'
        bashname = 'tofreda' + letter + '.job'
        with open(bashname, 'w') as outfile: 
            outfile.write(job_contents)
        os.system('sbatch tofreda' + letter + '.job')
        time.sleep(2)

def run_jobs(): 
    for i in range(1, 25): 
        letter = chr(ord('`')+i)
        job_contents = '#!/bin/bash\n' + \
            '# Job name:\n' + \
            '#SBATCH --job-name=sr_' + letter + '\n' + \
            '# Partition:\n' + \
            '#SBATCH --partition=savio2_gpu\n' + \
            '# Wall clock limit:\n' + \
            '#SBATCH --time=3-00:00:00\n' + \
            '#SBATCH --nodes=1\n' + \
            '#SBATCH --ntasks=1\n' + \
            '#SBATCH --cpus-per-task=2\n' + \
            '#SBATCH --gres=gpu:1\n' + \
            '#SBATCH --mail-user=lucy3_li@berkeley.edu\n' + \
            '#SBATCH --mail-type=END,FAIL\n' + \
            '#SBATCH --account=fc_dbamman\n' + \
            'source /global/scratch/lucy3_li/anaconda3/bin/activate /global/scratch/lucy3_li/anaconda3/envs/wsi\n' + \
            'time awk -F \",\" \'{print $1}\' /global/scratch/lucy3_li/bertwsi/subreddit_splits/xa' + letter + ' | parallel -j 1 --joblog ./tasklogs/task_' + letter + '.log --resume --tmpdir /global/scratch/lucy3_li/ \'python match_reddit.py {}\' & \n' + \
            'wait\n' + \
            'source /global/scratch/lucy3_li/anaconda3/bin/deactivate'
        bashname = 'xa' + letter + '.job'
        with open(bashname, 'w') as outfile: 
           outfile.write(job_contents)
        os.system('sbatch xa' + letter + '.job')
        time.sleep(2)

def subreddit_status(): 
    d = '/global/scratch/lucy3_li/bertwsi/tasklogs/'
    failed = set()
    success = set()
    times = []
    for f in os.listdir(d): 
        with open(d + f, 'r') as infile: 
            for line in infile: 
                contents = line.strip().split('\t')
                command = contents[-1]
                sr = command.split()[-1]
                if sr != 'Command': 
                    signal = contents[-2]
                    if signal != '0': 
                        failed.add(sr)
                    else:
                        t = float(contents[3])
                        times.append(t)
                        success.add(sr)
    times.append(165244.10511565208)
    times.append(97282.57520508766)
    times.append(110326.66728830338)
    times.append(2458069.1145219803)
    failed = failed - success
    print("FAILED", len(failed), failed)
    print("SUCCESS", len(success))
    print("Median time", np.median(times))
    print("Standard dev time", np.std(times))
    print("Total time", sum(times))

subreddit_status()


