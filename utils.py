import re
import numpy as np
from pyspark.sql.functions import UserDefinedFunction
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
    vals = np.array(df.select(col).limit(500).rdd.map(lambda x: x[0]).collect())
    not_double = np.sum(vals == None)/len(vals)
    return not_double < .3

replace_commas_symbols_udf = UserDefinedFunction(replace_commas_symbols, StringType())
