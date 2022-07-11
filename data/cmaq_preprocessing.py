import os
from netCDF4 import Dataset


# file_prefix = 'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.'
# cmaq_path = '/Volumes/4T/cmaq/'
# os.chdir(cmaq_path)

file_path = '/Volumes/4T/cmaq/201901/2019010112/1km/CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.2019002'
a = Dataset(file_path)
pass
