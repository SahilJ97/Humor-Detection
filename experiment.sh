#!/bin/sh

for seed in 1 2 3 ; do
  for json_file in colab_ourmodel_csi.json colab_simplebert_none.json colab_ourmodel_wordnet.json colab_ourmodel_none.json ; do
    output_dir=/content/gdrive/MyDrive/Humor-Detection/"${json_file%.json}"_"$seed"
    echo $output_dir
    rm -rf "$output_dir"
    mkdir "$output_dir"
    echo Beginning training. Writing to "$output_dir"
    PYTHON_PATH=/root/Humor-Detection python3 train.py --overwrite_output_dir --json "$json_file" \
    --batch_size 4 --grad_steps 8 --seed "$seed" --output_dir "$output_dir"
  done
done