import os
import re
import numpy as np
import glob
import shutil

from pyspark.sql.functions import *
from pyspark.sql.types import StringType

commas_re = re.compile('(\d+,\d+)(,\d*)*')
symbols_re = re.compile('(^\W+|\W+$)')

def repl(m):
    return "".join(m.group(0).split(','))

def replace_commas_symbols(ele):
    if not ele:
        return 'None'

    removed_commas = re.sub(commas_re, repl, ele)
    # clean $ symbols and other symbols that are at the start or end of string
    return re.sub(symbols_re, '', removed_commas)
    
def is_double(df, col):
    vals = np.array(df.select(col)
                      .head(500))

    not_double = np.sum(vals == None)/len(vals)
    return not_double < .3

def get_outliers(df, col, vals=None, side='both'):
    '''
    vals: if present should have two columns: first is the column col and second the column count

    This method computes outliers on col values or on vals.
    If vals is given, outputs the rows that have the col with an unusual value in vals. 
    '''
    bounds = {}
    df_temp = vals if vals else df
    counts_col = 'count' if vals else col  
    quantiles = df_temp.approxQuantile(counts_col,[0.25,0.75],0.05)
    IQR = quantiles[1] - quantiles[0]
    bounds[col] = [quantiles[0] - 4 * IQR,quantiles[1] + 4 * IQR]
    if 'rid' not in df.columns:
        df = df.withColumn('rid', monotonically_increasing_id())
    if vals:
        df_temp = df.join(vals, df[col] == vals[col]).drop(vals[col])

    if side == 'both':
        out = df_temp.where((df_temp[counts_col] < bounds[col][0]) | (df_temp[counts_col] > bounds[col][1])).select('rid', col, counts_col)
    elif side == 'left':
        out = df_temp.where(df_temp[counts_col] < bounds[col][0]).select('rid', col, counts_col)
    return out

def write_tsv(df, f_out):
    out_folder = "{}_temp".format(f_out)
    df.coalesce(1).write.csv(out_folder, sep='\t', header=True)
    csv_file = glob.glob("{}/*.csv".format(out_folder))[0]
    os.rename(csv_file, f_out)
    shutil.rmtree(out_folder)

replace_commas_symbols_udf = udf(replace_commas_symbols, StringType())
