try:
    from pyspark.sql import SparkSession
except:
    import findspark
    findspark.init()

import glob
import shutil
import time
import warnings
import json
import argparse
from collections import defaultdict
from scipy.spatial.distance import euclidean,mahalanobis
import scipy.stats as stats
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.clustering import KMeans, KMeansModel,GaussianMixture, GaussianMixtureModel
from pyspark.ml.feature import Bucketizer

import utils

parser = argparse.ArgumentParser(description='Write outliers')
parser.add_argument('files_in', metavar='fin', type=str, nargs='+',
                    help='path to tsv files to clean')

def print_outlier_summary(outliers_count, df_count, title):
    print("\t{}:\n\t\tcounts: {}\n\t\tcounts % = {:.2f}%".format(title, outliers_count, 100*(outliers_count/df_count)))

def str_length(df, col):
    target_col = "{}_len".format(col)
    df = df.withColumn(target_col, utils.col_len_udf(df[col]))
    out, out_target = iqr_outliers(df, col, target_col)
    print_outlier_summary(out.count(), df.count(), "Length")
    return out

def bucket_frequency(df, col, title='Frequency'):
    vals = df.groupBy(col).count()
    vals_col = "{}_temp".format(col)
    vals = vals.withColumnRenamed(col, vals_col)
    df = df.join(vals, df[col] == vals[vals_col]).drop(vals[vals_col])
    out, _ = iqr_outliers(df, col, 'count', 'left')
    out_count, df_count = out.count(), df.count()
    print_outlier_summary(out_count, df_count, title)
    return out

def histogram_outliers(df, col, bins=50):
    max_val = df.agg({col: 'max'}).collect()[0][0]
    min_val = df.agg({col: 'min'}).collect()[0][0]
    splits = np.linspace(min_val, max_val, num=bins+1, endpoint=True)
    bucketizer = Bucketizer(splits=splits, inputCol=col, outputCol='{}_bucket'.format(col))
    df_buck = bucketizer.setHandleInvalid('keep').transform(df)
    return bucket_frequency(df_buck, '{}_bucket'.format(col), 'Frequency @ {} bins'.format(bins))

def kmeans_outliers(df, numeric_cols, k=3, maxIterations=100):
    def addclustercols(x):
        points = np.array(x[1].toArray()).astype(float)
        center = clusters.centers[0]
        mindist = euclidean(points, center)
        c1 = 0

        for i in range(1, len(clusters.centers)):
            center = clusters.centers[i]
            dist = euclidean(points, center)
            if dist < mindist:
                c1 = i
                mindist = dist
        return (int(x[0]), float(mindist), int(c1))

    # Convert to array if only one column (univariate) passed
    if not isinstance(numeric_cols, list):
        numeric_cols = [numeric_cols]
    cols = ['rid']
    cols.extend(numeric_cols)
    df_col_rdd = df[cols].rdd
    label = df_col_rdd.map(lambda x: x[0])
    vso = df_col_rdd.map(lambda x: np.array(x[1:]).astype(float))
    scaler = StandardScaler(withMean=True, withStd=True).fit(vso)
    vso = scaler.transform(vso)

    clusters = KMeans.train(vso, k, initializationMode='k-means||', maxIterations=maxIterations)
    df_col_rdd = label.zip(vso).toDF().rdd
    rdd_w_clusts = df_col_rdd.map(lambda x: addclustercols(x))
    cols = ['rid', 'dist_c','c_no']
    kmeans_df = rdd_w_clusts.toDF(cols)
    outlier_all, _ = iqr_outliers(kmeans_df.where(kmeans_df['c_no'] == 0), 'dist_c')
    for i in range(1, k):
        outlier_c, _ = iqr_outliers(kmeans_df.where(kmeans_df['c_no'] == i), 'dist_c')
        outlier_all = outlier_all.unionAll(outlier_c)
    outlier_count = outlier_all.count()
    df_count = df.count()
    kmeans_type = 'multivariate' if len(numeric_cols) > 1 else 'univariate'
    print_outlier_summary(outlier_count, df_count, 'kMeans ({})'.format(kmeans_type))
    return outlier_all

# def gmm_outliers(df, numeric_cols, k=3, maxIterations=100):
#     def getDistances(x):
#         clust_center = x[0]
#         rid = x[1][0]
#         point = np.array(x[1][1].toArray()).astype(float)
#         dist = mahalanobis(clust_center,point,sigmas_inv[clust_center])
#         return (int(rid),int(clust_center),float(dist))
    
    
#     # Convert to array if only one column (univariate) passed
#     if not isinstance(numeric_cols, list):
#         numeric_cols = [numeric_cols]
#     cols = ['rid']
#     cols.extend(numeric_cols)
#     df_col_rdd = df[cols].rdd
#     rid = df_col_rdd.map(lambda x: x[0])
#     vso = df_col_rdd.map(lambda x: np.array(x[1:]).astype(float))
#     scaler = StandardScaler(withMean=True, withStd=True).fit(vso)
#     vso = scaler.transform(vso)
    
#     gmm = GaussianMixture.train(vso, k, maxIterations=maxIterations,seed=10)
    
#     df_col_rdd = rid.zip(vso).toDF().rdd
#     labels = gmm.predict(vso)
#     df_col_rdd = labels.zip(df_col_rdd)
#     mus = []
#     sigmas = []
#     sigmas_inv = []
#     for i in range(k):
#         mus.append(np.array(gmm.gaussians[i].mu.toArray()).astype(float))
#         sigmas.append(np.array(gmm.gaussians[i].sigma.toArray()).astype(float))
#         sigmas_inv.append(np.linalg.inv(sigmas[i]))
    
#     #print(df_col_rdd.collect())
#     rdd_w_clusts = df_col_rdd.map(lambda x: getDistances(x))
#     cols = ['rid', 'c_no', 'dist_c']
#     gmm_df = rdd_w_clusts.toDF(cols)
#     outlier_all, _ = iqr_outliers(gmm_df.where(gmm_df['c_no'] == 0), 'dist_c')
#     for i in range(1, k):
#         outlier_c, _ = iqr_outliers(gmm_df.where(gmm_df['c_no'] == i), 'dist_c')
#         outlier_all = outlier_all.unionAll(outlier_c)
#     outlier_count = outlier_all.count()
#     df_count = df.count()
#     gmm_type = 'multivariate' if len(numeric_cols) > 1 else 'univariate'
#     print_outlier_summary(outlier_count, df_count, 'GMM ({})'.format(gmm_type))
#     #print_outlier_summary(outlier_all.count(), df.count(), "kMeans (multivariate)")
#     return outlier_all

def likelihood(data, model, params):
    return np.sum(np.log(model.pdf(data, *params)))

def prob_models_outliers(df, col, max_points=100000):
    models = [stats.norm, stats.beta, stats.gamma]
    models_names = ["Gaussian", "Beta", "Gamma"]
    n = df.count()
    df_temp = df if n <= max_points else sample(fraction=max_points/float(n))
    data = np.array(df.select(col).rdd.map(lambda x: x[0]).collect()).reshape(-1, 1)
    params = [model.fit(data) for i, model in enumerate(models)]
    best_ix = np.argmax([likelihood(data, models[i], params[i]) for i, model in enumerate(models)])
    best_params = params[best_ix]
    best_model = models[best_ix]
    low, high = best_model.ppf([0.01, 0.99], *best_params)
    out = df_temp.select('rid', col).where((df_temp[col] < low) | (df_temp[col] > high))
    print_outlier_summary(out.count(), n, "Probability distributions {}".format(models_names[best_ix]))
    return out

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
    if(len(quantiles) < 2):
        return spark.createDataFrame(sc.emptyRDD(),df.schema[:2])
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
        out_count, df_count = out.count(), df.count()
        print_outlier_summary(out_count, df_count, "IQR")
    return out.select('rid', col), out.select('rid', target_col)

def main(files_in):
    spark = SparkSession \
            .builder \
            .appName("Nulls and Outliers Detection") \
            .getOrCreate()
    warnings.filterwarnings("ignore")
    for tsv in files_in:
        print("####### Analysing: {} #######".format(tsv))
        df = spark.read.csv(tsv, header=True, inferSchema=True, sep='\t')
        df_count = df.count()
        numeric_cols = []
        outliers = defaultdict(dict)
        outliers_all = defaultdict(dict)
        if 'rid' not in df.columns:
            df = df.withColumn("rid", monotonically_increasing_id())
        
        uni_cols = ['kMeans_uni','iqr','histogram','prob_models']
        for col, dtype in df.dtypes:
            dtype = dtype.lower()
            if col == 'rid' or 'time' in dtype or 'date' in dtype or 'array'\
                in dtype or 'boolean' in dtype or 'byte' in dtype or 'binary' in dtype:
                continue
            print("### Col: {} | dtype: {} ###".format(col, dtype))
            if 'string' in dtype:
                outliers[col]['frequency'] = bucket_frequency(df, col, 'Frequency')
                outliers[col]['length'] = str_length(df, col)
            else:
                numeric_cols.append(col)
                outliers[col]['iqr'], _ = iqr_outliers(df, col, to_print=True)
                outliers[col]['kMeans_uni'] = kmeans_outliers(df, col)
                outliers[col]['histogram'] = histogram_outliers(df, col)
                #outliers[col]['gmm_uni'] = gmm_outliers(df,col)
                outliers[col]['prob_models'] = prob_models_outliers(df, col)
                #get the agreeing outliers
                outliers_all[col] = outliers[col][uni_cols[0]].unionAll(outliers[col][uni_cols[1]])
                for i in range(2,len(uni_cols)):
                    outliers_all[col] = outliers_all[col].unionAll(outliers[col][uni_cols[i]])
                outliers_all_rdd = outliers_all[col].rdd
                outliers_all_rdd = outliers_all_rdd.map(lambda x: (x[0],1)).reduceByKey(lambda a,b:a+b).filter(lambda x:x[1]>=(len(uni_cols))/2)
                temp_cols = ['rid','no_appeared']
                outliers_all[col] = outliers_all_rdd.toDF(temp_cols)
                print_outlier_summary(outliers_all[col].count(),df.count(),"No. of outliers occuring more than {} times: ".format(len(uni_cols)/2))
                
                

        outliers['kMeans_multi'] = kmeans_outliers(df, numeric_cols)
        #outliers['gmm_multi'] = gmm_outliers(df,numeric_cols)

    spark.stop()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
