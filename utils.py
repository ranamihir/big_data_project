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

def write_tsv(df, f_out):
    out_folder = "{}_temp".format(f_out)
    df.coalesce(1).write.csv(out_folder, sep='\t', header=True)
    csv_file = glob.glob("{}/*.csv".format(out_folder))[0]
    os.rename(csv_file, f_out)
    shutil.rmtree(out_folder)

replace_commas_symbols_udf = udf(replace_commas_symbols, StringType())
