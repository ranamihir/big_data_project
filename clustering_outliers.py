import utils
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel

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
    outlier_all = utils.get_outliers(kmeans_df.where(kmeans_df['c_no']==0),'dist_c')
    for i in range(1,k):
        outlier_c = utils.get_outliers(kmeans_df.where(kmeans_df['c_no']==i),'dist_c')
        outlier_all = outlier_all.unionAll(outlier_c)
    #outliers = utils.get_outliers(kmeans_df,'dist_c')
    return outlier_all