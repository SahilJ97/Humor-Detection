#!/bin/sh
seed=$1
#colab_simplebert_none.json colab_ourmodel_csi.json colab_ourmodel_wordnet.json

for json_file in colab_ourmodel_none.json ; do
  output_dir=/content/gdrive/MyDrive/Humor-Detection/"${json_file%.json}"_"$seed"
  echo "$output_dir"
  rm -rf "$output_dir"
  mkdir "$output_dir"
  echo Beginning training. Writing to "$output_dir"
  if [ "$json_file" = "colab_ourmodel_csi.json" ] || [ "$json_file" = "colab_ourmodel_wordnet.json" ] || [ "$json_file" = "colab_ourmodel_none.json" ]
  then
    batch_size=8
    grad_steps=4
  else
    batch_size=16
    grad_steps=2
  fi
  PYTHON_PATH=/root/Humor-Detection python3 train.py --overwrite_output_dir --json "$json_file" \
  --batch_size "$batch_size" --grad_steps "$grad_steps" --seed "$seed" --output_dir "$output_dir"
done
