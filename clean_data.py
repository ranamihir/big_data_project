import os
import glob
import shutil

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

def dropna(df):
    return df.dropna()

def main(files_in, verbose):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()

    for tsv in files_in:
        df = spark.read.csv(path = tsv, header = True,inferSchema = True, sep='\t')
        replaced_df = correct_bad_classified_cols(df, verbose)

        # remove null values
        cleaned_df = dropna(replaced_df)
        
        #TODO: remove nas other types
        out_folder = "{}_clean".format(tsv.split('.')[0])
        cleaned_df.coalesce(1).write.csv(out_folder, mode='overwrite', sep='\t', header=True)

        csv_file = glob.glob("{}/*.csv".format(out_folder))[0]
        f_out = "{}_clean.tsv".format(tsv.split('.')[0])
        os.rename(csv_file, f_out)       
        shutil.rmtree(out_folder)

        spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))