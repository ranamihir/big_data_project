#!/bin/sh

module load python/gnu/3.4.4
module load spark/2.2.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python3
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python3

export PYTHONHASHSEED=0
export SPARK_YARN_USER_ENV=PYTHONHASHSEED=0

OUT_DIR=/user/dam740/null_outliers_out
hadoop fs -mkdir -p $OUT_DIR
if [ "$#" -eq 0 ]; then
  echo "At least one input file required"
  exit;
fi

spark-submit --py-files=utils.py clean_data.py "$@"

clean_dir=$OUT_DIR/clean
hadoop fs -mkdir -p $clean_dir

infiles=( "$@" )
clean=()
for f in ${infiles[*]}; do
  out_dir=${f%.tsv}_clean.tsv_temp # out dir from clean_data.py
  file_out=${out_dir##*/}
  clean_file=$clean_dir/${file_out%%_temp}
  hadoop fs -mv $out_dir/*.csv $clean_file
  hadoop fs -rm -r $out_dir

  clean+=("$clean_file")
done

spark-submit --py-files=utils.py get_outliers.py ${clean[*]}
