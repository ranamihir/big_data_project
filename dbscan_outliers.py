import argparse
import numpy as np
import pandas as pd
import pyproj
import folium
import os
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser(description='Plot DBSCAN outliers')
parser.add_argument('file_in', metavar='fin', type=str, \
    help='path to tsv file for running DBSCAN')
parser.add_argument('file_out', metavar='fout', type=str, \
    help='name of output map html file')
parser.add_argument('lat_col', metavar='lat', type=str, \
    help='name of latitude column for running DBSCAN')
parser.add_argument('long_col', metavar='long', type=str, \
    help='name of longitude column for running DBSCAN')

def get_dbscan_clusters(df, lat_col, long_col, min_samples=4, epsilon=2):
    # Set constants
    kms_per_radian = 6371.0088
    epsilon /= kms_per_radian

    # Get coordinates and run DBSCAN
    print('Running DBSCAN... ', end='', flush=True)
    coordinates = df.as_matrix(columns=[lat_col, long_col])
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', \
                    metric='haversine')
    dbscan.fit(np.radians(coordinates))
    print('Done.')
    return dbscan.labels_

def get_cluster_color(cid):
    if cid == -1:
        return 'red'
    return 'blue'

def plot_circles(lat_data, long_data, cluster_ids, file_out, plots_path='dbscan_plots/'):
    print('Plotting map... ', end='', flush=True)
    folium_map = folium.Map(location=[40.738, -73.98],
                            zoom_start=11,
                            tiles="OpenStreetMap")
    for rid, lat, long, cid in zip(lat_data.index, lat_data, long_data, cluster_ids):
        color = get_cluster_color(cid)
        marker = folium.CircleMarker(location=[lat, long], radius=5, color=color, \
            fill=color, fill_opacity='0.3', popup=folium.Popup('lat: {:.5f}, \
                long: {:.5f}\nrid: {}, clusterID: {}'.format(lat, long, rid, cid)))
        marker.add_to(folium_map)
    print('Done.')

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plot_path = '{}{}.html'.format(plots_path, file_out)
    print('Saving map to {}... '.format(plot_path), end='', flush=True)
    folium_map.save(plot_path)
    print('Done.')

def main(file_in, file_out, lat_col, long_col):
    df = pd.read_csv(file_in, sep='\t')
    df = df[(df[[lat_col, long_col]] != 0).all(axis=1)]
    df.reset_index(drop=True, inplace=True)

    cluster_ids = get_dbscan_clusters(df, lat_col, long_col)

    num_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print('Number of clusters: {}'.format(num_clusters))
    print('Number of outliers: {}'.format(cluster_ids[cluster_ids == -1].size))

    plot_circles(df[lat_col], df[long_col], cluster_ids, file_out)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
