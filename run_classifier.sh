#!/bin/bash

model=$1

if [ $# -eq 2 ]
then
  suffix=$2
else
  suffix=temp
fi

full_model="${model}_${suffix}"
config="classifier_$full_model.jsonnet"
serial_dir="`pwd`/model_$full_model"

echo "==> full model: $full_model"
echo "==> config file: $config"
echo "==> serialization dir: $serial_dir"
# exit

if [ -d $serial_dir ];then
  echo "==> remove existing serial folder"
  rm -rf $serial_dir
fi

if [ ! -f $config ];then
  echo "==> config file doesn't exist"
  exit
fi

allennlp train $config --serialization-dir $serial_dir --include-package my_text_classifier
