import argparse

#os.environ["PYSPARK_PYTHON"]='/Users/diogomesquita/anaconda/envs/py3.6/bin/python'
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *

import utils

parser = argparse.ArgumentParser(description='Clean data and remove null values.')
parser.add_argument('files_in', metavar='fin', type=str, nargs='+',
                    help='path to tsv files to clean')
parser.add_argument('--verbose', metavar='v', type=bool, default=False,
                    help='to print messages')

def correct_bad_classified_cols(df, verbose=False):
    replaced_df = df
    print(replaced_df.dtypes)
    for col, dtype in replaced_df.dtypes:
        if 'string' in dtype:
            tmp_col = "{}_temp".format(col)

            replaced_df = replaced_df.withColumn(
                tmp_col,
                utils.replace_commas_symbols_udf(replaced_df[col]).cast(DoubleType()))
            #print(col,': after 1')
            if utils.is_double(replaced_df, tmp_col):
                replaced_df = replaced_df.withColumn(col, replaced_df[tmp_col])
                
            #print('after 2')
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
        #df = spark.read.csv(path = tsv, header = True,inferSchema = True, sep='\t')
        #df = spark.read.format("csv").option("header", "true").option("sep","\t").option("inferSchema", "true").load(tsv)
        df = spark.read.load(tsv,format="csv", sep="\t", inferSchema="true", header="true")
        #print('read file')
        cleaned_df = remove_na_string_cols(df)
        replaced_df = correct_bad_classified_cols(cleaned_df, verbose)
        #print('after replaced file')

        # remove null values
        cleaned_df = remove_na_string_cols(replaced_df)
        
        #TODO: remove nas other types
        f_out = "{}_clean.tsv".format(tsv.split('.')[0])
        utils.write_tsv(cleaned_df, f_out)

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))