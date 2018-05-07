import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F

import utils

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

def remove_disguised_missing_vals(df):
    clean_df = df
    for col, dtype in clean_df.dtypes:
        if "string" in dtype:
            clean_df = clean_df.withColumn(col, F.trim(F.col(col)))
            clean_df = clean_df.withColumn(col, utils.replace_na_udf(clean_df[col]))
    return clean_df.dropna()

def dropna(df):
    return df.dropna()

def main(files_in, verbose):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()

    for tsv in files_in:
        df = spark.read.load(tsv,format="csv", sep="\t", inferSchema="true", header="true")
        df = dropna(df)
        replaced_df = correct_bad_classified_cols(df, verbose)
        # Remove null values
        cleaned_df = dropna(replaced_df)
        # remove disguised data
        cleaned_df = remove_disguised_missing_vals(cleaned_df)
        f_out = "{}_clean.tsv".format(tsv.split('.')[0])
        utils.write_tsv(cleaned_df, f_out)

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
