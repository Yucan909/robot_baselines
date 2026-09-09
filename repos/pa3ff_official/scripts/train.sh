#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None
LABEL="None"


while getopts "p:d:c:n:w:g:r:o:l:t:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    l)
      LABEL=$OPTARG
      ;;
    t)
      CATEGORY=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${GPU}" = 'None' ]
then
  GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "GPU Num: $GPU"

EXP_DIR=exp/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/pa3ff.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts launch pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" label="$LABEL" data.train.category="$CATEGORY"
else
    $PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT" data.train.category="$CATEGORY"
fi