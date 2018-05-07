# Project for 'DS-GA 1004 Big Data'
 ## Nulls and Outliers Detection


### Requirements
Run the following command:
<br>
`pip3 install -r requirements.txt`

### Usage
Run the following commands (back to back):
 - `python3 src/clean_data.py <folder_path/{file}.tsv>`
 - `python3 src/get_outliers.py <folder_path/{file}_clean.tsv>`

For DBSCAN outliers:
 - `python3 src/dbscan_outliers.py <folder_path/{file}_clean.tsv> <latitude_column> <longitude_column> <outfile_name>`
