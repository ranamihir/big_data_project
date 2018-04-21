import glob
import shutil
import argparse

#os.environ["PYSPARK_PYTHON"]='/Users/diogomesquita/anaconda/envs/py3.6/bin/python'
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from clustering_outliers import kmeans_outliers

import utils

parser = argparse.ArgumentParser(description='Write outliers')
parser.add_argument('files_in', metavar='fin', type=str, nargs='+',
                    help='path to tsv files to clean')

def main(files_in):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()

    print("IQR method...")
    for tsv in files_in:
        df = spark.read.csv(tsv, header = True,inferSchema = True, sep='\t')
        df_count = df.count()

        keyval = {}
        kmeans = {}
        if 'rid' not in df.columns:
            df = df.withColumn("rid", monotonically_increasing_id())

        for col, dtype in df.dtypes:
            print("### Col: {} | dtype: {} ###".format(col, dtype))
            if col == 'rid':
                continue
            vals = df.groupBy(col).count() if 'string' in dtype else None
            side = 'left' if 'string' in dtype else 'both'
            keyval[col] = utils.get_outliers(df, col, vals, side)

            if 'string' not in dtype:
                kmeans[col] = kmeans_outliers(df, col)
                print("\tKmeans:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(kmeans[col].count(), 100*(kmeans[col].count()/df_count)))    
            print("\tIQR:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(keyval[col].count(), 100*(keyval[col].count()/df_count)))

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))