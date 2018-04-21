from pyspark.sql.functions import *


def get_outliers(df,col):
    bounds = {}
    quantiles = df.approxQuantile(col,[0.25,0.75],0.05)
    IQR = quantiles[1] - quantiles[0]
    bounds[col] = [quantiles[0] - 4 * IQR,quantiles[1] + 4 * IQR]
    #print(bounds)
    if 'rid'not in df.columns:
        df = df.withColumn("rid", monotonically_increasing_id())
    outliers = df.select(*['rid'] + [((df[col] < bounds[col][0]) | (df[col] > bounds[col][1])).alias(col+'_o')])
    df_outliers = df.join(outliers, on='rid')
    df_outliers = df_outliers.filter(col + '_o').select('rid', col)
    return df_outliers


