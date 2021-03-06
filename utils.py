import os
import re
import numpy as np
import glob
import shutil

import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType

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

    not_double = np.sum(np.equal(vals, None))/len(vals)
    return not_double < .3

def write_tsv(df, f_out):
    out_folder = "{}_temp".format(f_out)
    df.coalesce(1).write.csv(out_folder, sep='\t', header=True)
    organize_cleaned_data(out_folder, f_out)

def organize_cleaned_data(out_folder, f_out):
    try:
        csv_file = glob.glob("{}/*.csv".format(out_folder))[0]
        os.rename(csv_file, f_out)
        shutil.rmtree(out_folder)
    except:
        # running in dumbo with hdfs
        return

def col_len(ele):
    return len(ele)

import re

def replace_na(ele):
    empty_re = re.compile(r'^$|^nan$|^n/a$|^none$|^n\\a$', flags=re.I)
    return None if re.search(empty_re, ele) else ele

replace_na_udf = F.udf(replace_na)
replace_commas_symbols_udf = F.udf(replace_commas_symbols, StringType())
col_len_udf = F.udf(col_len, IntegerType())
