import argparse
from netCDF4 import Dataset
import numpy as np
from multiprocessing import Pool
# from paths import WRF_PATH
import os
import pandas as pd
import sys
import tqdm

WRF_PATH = '/home/dataop/data/nmodel/wrf_fc'

"""
A module for copying WRF variables (U10, V10, T2, QVAPOR, PBLH, PSFC) from netCDF4 format to csv format. 
Read WRF data from day <start>-1 12:00 to <end>+1 11:00 (UTC+0).


General steps to read wrf data:
    1. cd to /home/dataop/data/nmodel/wrf_fc
    2. cd /Y/Ym/Ymd12
    3. read wrfout_d01_Y-m-(d+1)_H:M:s, with H from 
       12:00 on d+1 to 11:00 on d+2
    4. for each file, read hourly readings of
       'u', 'v', 'temp', 'q_vapor', 'pblh', 'psfc' 
       at (UTC+0)

Example: reading data from 31/12/2013 12:00 to 1/1/2014 11:00
    1. cd /home/dataop/data/nmodel/wrf_fc
    2. cd 2013/201312/2013123012/
    3. open 'wrfout_d01_2013-12-31_12:00:00' to
      'wrfout_d01_2014-01-01_11:00:00' and read the 
      variables.
"""

# define constants
FILEHEAD = 'wrfout_d01_'
ONE_DAY_DELTA = pd.Timedelta(days=1)


def _divide_daterange(start, end, interval=32):
    """
    Divide the range from start to end into smaller segments, rounded to day. Number of intervals is specified by interval.
    
    :param str/datetime start: The first date of range for division.
    :param str/datetime end: The last date of range for division.
    :param int interval: Numbers of intervals after division.
    
    :return list: A list of datetime of divided length.
    """
    start = pd.to_datetime(start, format='%Y%m%d')
    end = pd.to_datetime(end, format='%Y%m%d')
    diff = (end - start) / interval
    diff = pd.Timedelta(days=diff.days + 1)

    date_list = []
    while start < end:
        date_list.append(start)
        start += diff
    date_list.append(end)
    return date_list


def _get_variables(current, wll):
    ar = np.zeros((len(wll), 6))
    a = Dataset(FILEHEAD + current.strftime('%Y-%m-%d_%H:%M:%S'))
    for i, (row, col) in enumerate(wll):
        u = a.variables['U10'][0, row, col]
        v = a.variables['V10'][0, row, col]
        temp = a.variables['T2'][0, row, col]
        q_vapor = a.variables['QVAPOR'][0, 0, row, col]
        pblh = a.variables['PBLH'][0, row, col]
        psfc = a.variables['PSFC'][0, row, col]
        ar[i] = [u, v, temp, q_vapor, pblh, psfc]
    a.close()
    return ar


def _read_wrf(args):
    # unpack arguments
    start, end, i, save_dir, wll_path, stat_range = args
    cur_path = os.getcwd()
    # read wll
    wll = pd.read_csv(wll_path, index_col=0, na_filter=False).values
    # if stat_range is int, start copying from stat_range; if list only copy those within the list
    if isinstance(stat_range, int):
        stat_range = [x for x in range(stat_range, len(wll))]
    wll = wll[stat_range]
    # define the range of time
    di = pd.date_range(start + pd.Timedelta(hours=12), end + pd.Timedelta(hours=11), freq='H')
    # define empty dataframes
    dfs = [pd.DataFrame(index=di,
                        columns=['WRF_WIND_U', 'WRF_WIND_V', 'WRF_TEMP', 'WRF_Q_VAPOR', 'WRF_PBLH', 'WRF_PSFC'])
           for _ in wll]
    # offset dir_time by 2 days to fit the start of loop
    dir_time = start - pd.Timedelta(days=2)

    for current in di:
        # advance dir_time by one day when current is 12:00:00
        if current.hour == 12:
            dir_time += ONE_DAY_DELTA
        os.chdir(os.path.join(WRF_PATH, str(dir_time.year),
                              dir_time.strftime('%Y%m'),
                              dir_time.strftime('%Y%m%d') + '12'))
        li = _get_variables(current, wll)
        # append value to each dataframe at current timestep
        for df, l in zip(dfs, li):
            df.loc[current] = l

    os.chdir(cur_path)
    for index, df in zip(stat_range, dfs):
        if i == 0:
            df.to_csv(os.path.join(save_dir, f'{index}.csv'))
        else:
            df.to_csv(os.path.join(save_dir, f'{index}_{i}.csv'))


def read_wrf(start, end, save_dir, wll_path, stat_range=0):
    """
    Copying WRF data from start 12:00 to end 11:00, no offset in this function. 
    Process only stations specified in stat_range. Copied data are in save_dir.
    This function only uses 1 process.
    
    :param datetime start: Starting date in copying (no offset)
    :param datetime end: Ending date in copying (no offset)
    :param path save_dir: Path for saving copied data
    :param path wll_path: Path to the file for storing WRF station indices
    :stat_range int_or_list: Range of stations to be copied in station indices. 
        If int, stations with index greater than or equals to stat_range will be copied. 
        If list, only stations with index within the list is copied.
    """
    _read_wrf((start, end, 0, save_dir, wll_path, stat_range))


def concat_frames(save_dir, wll_path, num_frag, start_index=0):
    cur_path = os.getcwd()
    os.chdir(save_dir)
    wll = pd.read_csv(wll_path, index_col=0, na_filter=False)
    for i in range(start_index, len(wll)):
        dfs = [pd.read_csv(f'{i}_{x}.csv', index_col=0, na_filter=False)
               for x in range(1, num_frag)]
        dfs.insert(pd.read_csv(f'{i}.csv', index_col=0, na_filter=False), 0)
        df = pd.concat([dfs])
        df.to_csv(f'{i}.csv')
        for x in range(1, num_frag):
            os.remove(f'{i}_{x}.csv')
    os.chdir(cur_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read wrf data from day (start-1) 12:00 to (end+1) 11:00 (UTC+0).')
    parser.add_argument('start', type=str, help='Starting date for reading, format: Ymd')
    parser.add_argument('end', type=str, help='Ending date for reading, format: Ymd')
    parser.add_argument('-n', '--num_proc', type=int, default=10, help='Number of processes')
    parser.add_argument('-p', '--path', type=str, default=None, help='Path to save files')
    parser.add_argument('-m', '--wrf_map', type=str, default=None, help='Path to list of wrf row col indices')
    parser.add_argument('-i', '--start_index', type=int, default=0, help='Starting index of wrf pair')

    args = parser.parse_args()
    start = pd.to_datetime(args.start, format='%Y%m%d') - ONE_DAY_DELTA
    end = pd.to_datetime(args.end, format='%Y%m%d') + ONE_DAY_DELTA
    num_proc = args.num_proc
    start_index = args.start_index
    save_dir = args.path if args.path is not None else '/localdisk/r103/export/lihaobo/data/wrf'
    wrf_map = args.wrf_map if args.wrf_map is not None \
        else '/localdisk/r103/export/lihaobo/data/wrf/settings/wrf_idx.csv'

    date_range = _divide_daterange(start, end, num_proc)

    inputs = ((d1, d2, i, save_dir, wrf_map, start_index) \
              for i, (d1, d2) in enumerate(zip(date_range[:-1], date_range[1:])))
    pool = Pool(5)
    for _ in tqdm.tqdm(pool.imap_unordered(_read_wrf, inputs), total=num_proc):
        pass

    if num_proc > 1:
        concat_frames(save_dir, wrf_map, num_proc, start_index)
    print('Finish reading wrf data.')
