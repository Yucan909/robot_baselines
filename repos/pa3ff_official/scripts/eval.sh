#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TEST_CODE=eval.py

CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
GPU=None

while getopts "p:d:c:n:w:g:t:" opt; do
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
    g)
      GPU=$OPTARG
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
echo "Category: $CATEGORY"

EXP_DIR=exp/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/pa3ff.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
mkdir -p "$MODEL_DIR" "$CODE_DIR"
cp -r scripts launch pointcept "$CODE_DIR"

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

$PYTHON -u launch/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${CATEGORY}"/"${WEIGHT}".pth data.train.category="$CATEGORY" data.train.data_root="./partnet/dataset"
