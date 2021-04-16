#!/bin/sh
seed=$1

for json_file in colab_reduplicate.json ; do
  output_dir=/content/gdrive/MyDrive/Humor-Detection/"${json_file%.json}"_"$seed"
  echo "$output_dir"
  rm -rf "$output_dir"
  mkdir "$output_dir"
  echo Beginning training. Writing to "$output_dir"

  PYTHON_PATH=/root/Humor-Detection python3 train.py --overwrite_output_dir --json "$json_file" \
  --seed "$seed" --output_dir "$output_dir"
done
