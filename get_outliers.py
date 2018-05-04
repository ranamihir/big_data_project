import glob
import shutil
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
    outlier_all = iqr_outliers(kmeans_df.where(kmeans_df['c_no']==0),'dist_c')
    for i in range(1,k):
        outlier_c = iqr_outliers(kmeans_df.where(kmeans_df['c_no']==i),'dist_c')
        outlier_all = outlier_all.unionAll(outlier_c)
    #outliers = iqr_outliers(kmeans_df,'dist_c')
    return outlier_all

def iqr_outliers(df, col, vals=None, side='both'):
    '''
    vals: if present should have two columns: first is the column col and second the column count

    This method computes outliers on col values or on vals.
    If vals is given, outputs the rows that have the col with an unusual value in vals. 
    '''
    bounds = {}
    df_temp = vals if vals else df
    if vals:
        vals_col = "{}_temp".format(col)
        vals = vals.withColumnRenamed(col, vals_col)

    counts_col = 'count' if vals else col  
    if df_temp.count() == 0:
        return df_temp.select('rid', col, counts_col)

    quantiles = df_temp.approxQuantile(counts_col,[0.25,0.75],0.05)
    try:
        IQR = quantiles[1] - quantiles[0]
    except IndexError:
        import pdb; pdb.set_trace()
    bounds[col] = [quantiles[0] - 4 * IQR, quantiles[1] + 4 * IQR]
    if 'rid' not in df.columns:
        df = df.withColumn('rid', monotonically_increasing_id())
    if vals:
        df_temp = df.join(vals, df[col] == vals[vals_col]).drop(vals[vals_col])

    if side == 'both':
        out = df_temp.where((df_temp[counts_col] < bounds[col][0]) | (df_temp[counts_col] > bounds[col][1])).select('rid', col, counts_col)
    elif side == 'left':
        out = df_temp.where(df_temp[counts_col] < bounds[col][0]).select('rid', col, counts_col)
    return out

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
                vals = df.groupBy(col).count()
                outliers[col]['frequency'] = iqr_outliers(df, col, vals, 'left')
                print("\tFrequency:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(outliers[col]['frequency'].count(), 100*(outliers[col]['frequency'].count()/df_count)))
            else:
                outliers[col]['iqr'] = iqr_outliers(df, col)
                outliers[col]['kmeans'] = kmeans_outliers(df, col)
                print("\tKmeans:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(outliers[col]['kmeans'].count(), 100*(outliers[col]['kmeans'].count()/df_count)))    
                print("\tIQR:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(outliers[col]['iqr'].count(), 100*(outliers[col]['iqr'].count()/df_count)))

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
