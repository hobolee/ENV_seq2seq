#!/disk/hq246/yhsha/anaconda3/bin/python
"""
A module for copying CMAQ variables (PM2.5, O3, NO2, SO2 from netCDF4 format to csv format. 
Read CMAQ data from day <start>-1 12:00 to <end>+1 11:00 (UTC+0).


General rules for parsing CMAQ files:
Date format: D-m-Y
1. cd to /cmaq_fc/Y/Ym/YmD12/27km
2. Read 20XXn file (n is the D+1th day)
3. The variables corresponds to the record from 12:00 in the n-th day to 11:00 in the n+1th day (UTC+0)

Example: reading the variables from 31-12-2013 12:00 to 01-01-2014 11:00
1. cd /cmaq_fc/2013/201312/2013123012/27km
2. Open CCTM_v47_ebi_cb05cl_ae5_aq_mpi_pg64_mpich2.ACONC.2013365
3. Read the variables 
"""

import argparse
from netCDF4 import Dataset
from multiprocessing import Pool
import os
import pandas as pd
import numpy as np
import sys
import tqdm

# define constants
PM = [
    'ASO4J', 'ASO4I', 'ANO3J', 'ANO3I', 'ANH4J', 'ANH4I', 'AALKJ', 'AXYL1J', 'AXYL2J', 'AXYL3J', 'ATOL1J', 
    'ATOL2J', 'ATOL3J', 'ABNZ1J', 'ABNZ2J', 'ABNZ3J', 'ATRP1J', 'ATRP2J', 'AISO1J', 'AISO2J', 'ASQTJ',
    'AORGCJ', 'AORGPAJ', 'AORGPAI', 'AECJ', 'AECI', 'A25J', 'A25I', 'ANAJ', 'ANAI', 'ACLJ', 'ACLI', 'AISO3J',
    'AOLGAJ', 'AOLGBJ', 'ANAI'
]
FN1  = 'CCTM_v47_ebi_cb05cl_ae5_aq_mpi_pg64_mpich2.ACONC.'
FN2  = 'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.'
ONE_DAY_DELTA = pd.Timedelta(days = 1)
CMAQ_PATH = ''

def _divide_daterange(start, end, interval = 32):
    start = pd.to_datetime(start, format = '%Y%m%d')
    end   = pd.to_datetime(end, format = '%Y%m%d')
    diff  = (end - start)/interval
    diff  = pd.Timedelta(days = diff.days+1)

    date_list = []
    while start < end:
        date_list.append(start)
    start += diff
    date_list.append(end)
    return date_list

def _get_variables(current, cll):
    ar = np.zeros((len(cll), 24, 4))
    # open the netCDF4 file, use FN1 for (almost all) files before 2015 
    try:
        a = Dataset(FN2 + current.strftime("%Y%j"))
    except:
        a = Dataset(FN1 + current.strftime("%Y%j"))
    for i, (row, col) in enumerate(cll):
        acc_pm = sum((a.variables[p][:, 0, row, col] for p in PM))
        o3 = a.variables['O3'][:, 0, row, col]
        no2 = a.variables['NO2'][:, 0, row, col]
        so2 = a.variables['SO2'][:, 0, row, col]
        ar[i] = np.array([acc_pm, o3, no2, so2]).T
    a.close()
    return ar


def _read_cmaq(args):
    start, end, i, save_dir, cll_map, stat_range = args
    cur_path = os.getcwd()
    cll = pd.read_csv(cll_map, index_col = 0, na_filter = False).values
    # if stat_range is int, start copying from stat_range; if list only copy those within the list
    if isinstance(stat_range, int):
        stat_range = [x for x in range(stat_range, len(cll))]
    cll = cll[stat_range]
    di = pd.date_range(start + pd.Timedelta(hours = 12), end + pd.Timedelta(hours = 11), freq = 'H')
    # define empty dataframes
    dfs = [pd.DataFrame(index = di, columns = ['CMAQ_PM2.5', 'CMAQ_O3', 'CMAQ_NO2', 'CMAQ_SO2']) for _ in cll]
    
    for current in pd.date_range(start, end - ONE_DAY_DELTA, freq = 'D'):
        os.chdir(CMAQ_PATH)
        dir_time = current - ONE_DAY_DELTA
        os.chdir(os.path.join(dir_time.strftime('%Y'), dir_time.strftime('%Y%m'), 
                              f'{dir_time.strftime("%Y%m%d")}12', '27km')) 
        # define index for dataframes        
        time_idx = pd.date_range(start = current + pd.Timedelta(hours = 12), 
                                 periods = 24, freq = 'H', name = 'Time')      
        # get an iterator for the dataframes at current timestep,
        # then concatenate each frame with the previously generated frame to get another iterator
        # of the concatenated frames. If no frames had been created, registor a new iterator.
        li = _get_variables(current, cll)
        for df, l in zip(dfs, li):
            df.loc[time_idx] = l

    # save dataframes according to i 
    os.chdir(cur_path)
    for index, df in zip(stat_range, dfs):
        if i == 0:
            df.to_csv(os.path.join(save_dir, f'{index}.csv'))
        else:
            df.to_csv(os.path.join(save_dir, f'{index}_{i}.csv'))

            
def read_cmaq(start, end, save_dir, cll_path, stat_range = 0):
    """
    Copying CMAQ data from start 12:00 to end 11:00, no offset in this function. 
    Process only stations specified in stat_range. Copied data are in save_dir.
    This function only uses 1 process.
    
    :param datetime start: Starting date in copying (no offset)
    :param datetime end: Ending date in copying (no offset)
    :param path save_dir: Path for saving copied data
    :param path wll_path: Path to the file for storing CMAQ station indices
    :stat_range int_or_list: Range of stations to be copied in station indices. 
        If int, stations with index greater than or equals to stat_range will be copied. 
        If list, only stations with index within the list is copied.
    """
    _read_cmaq((start, end, 0, save_dir, cll_path, stat_range))


def concat_frames(save_dir, cll_path, num_frag, start_index = 0):
    cur_path = os.getcwd()
    os.chdir(save_dir)
    cll = pd.read_csv(cll_path, index_col = 0, na_filter = False)
    for i in range(start_index, len(wll)):
        dfs = [pd.read_csv(f'{i}_{x}.csv', index_col = 0, na_filter = False) 
                    for x in range(1, num_frag)]
        dfs.insert(pd.read_csv(f'{i}.csv', index_col = 0, na_filter = False), 0)
        df = pd.concat([dfs])
        df.to_csv(f'{i}.csv')
        for x in range(1, num_frag):
            os.remove(f'{i}_{x}.csv')
    os.chdir(cur_path)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('start', type = str, help = 'Starting date for reading, format: d/m/Y') 
    parser.add_argument('end', type = str, help = 'Ending date for reading, format: d/m/Y')
    parser.add_argument('-n', '--num_proc', type = int, default = 10, help = 'Number of processes')
    parser.add_argument('-p', '--path', type = str, default = None, help = 'Path to save files')
    parser.add_argument('-m', '--cmaq_map', type = str, default = None, help = 'Path to list of cmaq row col indices')
    parser.add_argument('-i', '--start_index', type = int, default = 0, help = 'Starting index of cmaq pair')
    
    args = parser.parse_args()
    start = pd.to_datetime(args.start, format = '%Y%m%d') - ONE_DAY_DELTA
    end   = pd.to_datetime(args.end, format = '%Y%m%d') + ONE_DAY_DELTA
    num_proc = args.num_proc 
    start_index = args.start_index
    save_dir = args.path if args.path is not None else '/data/cmaq/data'
    cll_map = args.cmaq_map if args.cmaq_map is not None \
        else '/data/wrf/settings/cmaq_idx.csv'
    
    date_range = _divide_daterange(start, end, num_proc)
    
    inputs = ((d1, d2, i, save_dir, cll_map, start_index) \
              for i, (d1, d2) in enumerate(zip(date_range[:-1], date_range[1:])))    
    pool = Pool(5)
    for _ in tqdm.tqdm(pool.imap_unordered(_read_cmaq, inputs), total = num_proc):
        pass
    
    if num_proc > 1:
        concat_frames(save_dir, cll_map, num_proc, start_index)
    print('Finish reading cmaq data.')
#     _read_cmaq(start, end, 0, save_dir, cmaq_map)