#!/bin/bash

MODELS=(
  #'BRISQUE'
  #'vgg19'
  #'resnet50'
  #'TLVQM'
  #'RAPIQUE'
  'GAME'
  #'NDNet'
)

DATASETS=(
  #'YT-UGC-Gaming'
  #'LIVE-YT-Gaming'
  #'GamingVideoSET'
  #'KUGVD'
  #'CGVDS'
  'LIVE-Meta-Gaming'
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do
for FOLDER in "${FOLDERS[@]}"
do

  feature_file=feat_files/${DS}_${m}_feats.mat
  out_file=result/${DS}_${m}
  predicted_score=pre_score/${DS}_${m}_predicted_score.mat
  best_pamtr=best_pamtr/${DS}_${m}_pamtr.mat
  log_file=logs/${DS}_regression.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python evaluate_bvqa_features_regression.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --predicted_score ${predicted_score}"
  cmd+=" --best_parameter ${best_pamtr}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
#   cmd+=" --use_parallel"
  cmd+=" --log_short"

  echo "${cmd}"

  eval ${cmd}
done
done
done