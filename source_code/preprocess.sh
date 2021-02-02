#!/bin/bash

TASK=bart/
TASK=poet/

fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref data_process/$TASK/train \
  --validpref data_process/$TASK/valid \
  --testpref data_process/$TASK/test \
  --destdir data-bin/$TASK \
  --workers 60 \
  --srcdict data_process/$TASK/dict.txt \
  --tgtdict data_process/$TASK/dict.txt