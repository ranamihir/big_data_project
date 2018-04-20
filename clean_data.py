#import os
#os.environ["PYSPARK_PYTHON"]='/Users/diogomesquita/anaconda/envs/py3.6/bin/python'

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.functions import *

import utils

import argparse

parser = argparse.ArgumentParser(description='Clean data and remove null values.')
parser.add_argument('files_in', metavar='fin', type=str, nargs='+',
                    help='path to tsv files to clean')
parser.add_argument('--verbose', metavar='v', type=bool, default=False,
                    help='to print messages')

def correct_bad_classified_cols(df, verbose=False):
    replaced_df = df
    for col, dtype in replaced_df.dtypes:
        if 'string' in dtype:
            tmp_col = "{}_temp".format(col)

            replaced_df = replaced_df.withColumn(
                tmp_col,
                utils.replace_commas_symbols_udf(replaced_df[col]).cast(DoubleType()))

            if utils.is_double(replaced_df, tmp_col):
                replaced_df = replaced_df.withColumn(col, replaced_df[tmp_col])
            replaced_df = replaced_df.drop(tmp_col)
    return replaced_df

def remove_na_string_cols(df):
    string_cols = [col for col, dtype in df.dtypes if 'string' in dtype]
    return df.dropna(subset=string_cols)

def main(files_in, verbose):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()

    for tsv in files_in:
        df = spark.read.csv(path = tsv, header = True,inferSchema = True, sep='\t')
        replaced_df = correct_bad_classified_cols(df, verbose)

        # remove null values
        cleaned_df = remove_na_string_cols(replaced_df)
        
        #TODO: remove nas other types

        f_out = "{}_clean".format(tsv.split('.')[0])
        cleaned_df.coalesce(1).write.csv(f_out, mode='overwrite', sep='\t', header=True)
        
        spark.stop()
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))