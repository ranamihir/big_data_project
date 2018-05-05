import glob
import shutil
import time
import json
import argparse
from collections import defaultdict
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.mllib.clustering import KMeans, KMeansModel

import utils

parser = argparse.ArgumentParser(description='Write outliers')
parser.add_argument('files_in', metavar='fin', type=str, nargs='+',
                    help='path to tsv files to clean')

def print_outlier_summary(outliers, df_count, title):
    print("\t{}:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(title, outliers.count(), 100*(outliers.count()/df_count)))

def str_length(df, col):
    target_col = "{}_len".format(col)
    df = df.withColumn(target_col, utils.col_len_udf(df[col]))
    out, out_target = iqr_outliers(df, col, target_col)
    print_outlier_summary(out, df.count(), "Length")
    return out

def str_frequency(df, col):
    vals = df.groupBy(col).count()
    vals_col = "{}_temp".format(col)
    vals = vals.withColumnRenamed(col, vals_col)
    df = df.join(vals, df[col] == vals[vals_col]).drop(vals[vals_col])
    out, _ = iqr_outliers(df, col, 'count', 'left')
    print_outlier_summary(out, df.count(), "Frequency")
    return out

def kmeans_outliers(df,col,k=3,maxIterations = 100):
    def addclustercols(x):
        point = np.array(float(x[1]))
        center = clusters.centers[0]
        mindist = np.abs(point - center)
        c1 = 0
        for i in range(1,len(clusters.centers)):
            center = clusters.centers[i]
            dist = np.abs(point - center)
            if dist < mindist:
                c1 = i
                mindist = dist
        return (int(x[0]),float(x[1]),int(c1),float(mindist))
    df_col_rdd = df.select(df['rid'],df[col]).rdd
    vso = df_col_rdd.map(lambda x: np.array(float(x[1])))
    clusters = KMeans.train(vso,k,initializationMode='random',maxIterations=maxIterations)
    rdd_w_clusts = df_col_rdd.map(lambda x: addclustercols(x))
    kmeans_df = rdd_w_clusts.toDF(['rid',col,'c_no','dist_c'])
    outlier_all, _ = iqr_outliers(kmeans_df.where(kmeans_df['c_no']==0),'dist_c')
    for i in range(1,k):
        outlier_c, _ = iqr_outliers(kmeans_df.where(kmeans_df['c_no']==i),'dist_c')
        outlier_all = outlier_all.unionAll(outlier_c)
    #outliers, _ = iqr_outliers(kmeans_df,'dist_c')
    print_outlier_summary(outlier_all, df.count(), "Kmeans")
    return outlier_all

def iqr_outliers(df, col, target_col=None, side='both', to_print=False):
    '''
    This method computes outliers on target_col values
    If vals is given, outputs the rows that have the col with an unusual value in vals. 
    '''
    bounds = {}
    if not target_col:
        target_col = col

    if df.count() == 0:
        return df.select('rid', col), df.select(target_col)

    quantiles = df.approxQuantile(target_col,[0.25,0.75],0.05)
    IQR = quantiles[1] - quantiles[0]
    threshold = 1.5
    not_done = True
    out = None
    while not_done:
        bounds[col] = [quantiles[0] - threshold * IQR, quantiles[1] + threshold * IQR]
        if 'rid' not in df.columns:
            df = df.withColumn('rid', monotonically_increasing_id())
        if side == 'both':
            out = df.where((df[target_col] < bounds[col][0]) | (df[target_col] > bounds[col][1]))
        elif side == 'left':
            out = df.where(df[target_col] < bounds[col][0])
        not_done = float(out.count())/df.count() > .02 and threshold < 10
        threshold += .5

    if to_print:
        print_outlier_summary(out, df.count(), "IQR")
    return out.select('rid', col), out.select('rid', target_col)

def main(files_in):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()

    for tsv in files_in:
        df = spark.read.csv(tsv, header = True,inferSchema = True, sep='\t')
        df_count = df.count()
        outliers = defaultdict(dict)
        if 'rid' not in df.columns:
            df = df.withColumn("rid", monotonically_increasing_id())

        for col, dtype in df.dtypes:
            print("### Col: {} | dtype: {} ###".format(col, dtype))
            if col == 'rid':
                continue

            if 'string' in dtype:
                outliers[col]['frequency'] = str_frequency(df, col)
                outliers[col]['length'] = str_length(df, col)
            else:
                outliers[col]['iqr'], _ = iqr_outliers(df, col, to_print=True)
                outliers[col]['kmeans'] = kmeans_outliers(df, col)

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
