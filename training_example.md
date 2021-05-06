# Training a NetTIME model

Here we demonstrate how one can train a NetTIME model using our [example training data](data/datasets/training_example/). Check out [this tutorial](preprocess/README.md#generating-data-to-train-a-nettime-model-from-scratch) on how to generate training data like this from a bed file.

## Train and evaluate a NetTIME model

NetTIME training workflow is defined in `NetTIME_train.py`. Let's try to train a model (named `training_example`) using DNase-seq as cell-type-specific feature and HOCOMOCO motif as TF-specific feature. Specify a checkpoint file path using flag `--start_from_checkpoint` if you wish to start training from a pretrained model.
```
python NetTIME_train.py \
--batch_size 900 \
--num_workers 20 \
--num_epochs 50 \
--learning_rate 1e-4 \
--ct_feature \
--tf_feature \
--print_every 5 \
--evaluate_every 25 \
--dataset "data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5" \
--index_file "data/embeddings/example.pkl" \
--experiment_name "training_example" \
--input_size 3 \
--dropout 0.1
```
Model evaluation workflow is defined in `NetTIME_evaluate.py`. Evaluation is implemented in `NetTIME/evaluator.py`, which will search for all checkpoints that haven't been evaluated and evaluate them one by one. This workflow can be run multiple times in parallel to shorten evaluation time. The checkpoint that achieves the highest Area Under the Precision Recall Curve (AUPR) score on the validation set is selected as the best checkpoint.
```
python NetTIME_evaluate.py \
--batch_size 2700 \
--num_workers 10 \
--experiment_name "training_example" \
--dataset "data/datasets/training_example/validation_minOverlap200_maxUnion600_example.h5" \
--index_file "data/embeddings/example.pkl"
```

Make prediction on validation set.
```
python NetTIME_predict.py \
--batch_size 2700 \
--num_workers 10 \
--dataset "data/datasets/training_example/validation_minOverlap200_maxUnion600_example.h5" \
--dtype "VALIDATION" \
--index_file "data/embeddings/example.pkl" \
--experiment_name "training_example"
```


## Train and evaluate a conditional random field classifier 

Train a linear-chain conditional random field (CRF) classifier from NetTIME binding probability predictions generated from the best model checkpoint. 
```
python NetTIME_CRF_train.py \
--batch_size 900 \
--num_workers 20 \
--num_epochs 50 \
--print_every 5 \
--evaluate_every 25 \
--experiment_name "training_example" \
--index_file "data/embeddings/example.pkl" \
--dataset "data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5" \
--class_weight "data/datasets/training_example/training_minOverlap200_maxUnion600_example_weight.npy"
```

Evaluate all checkpoints of a trained classifier using NetTIME predictions on validation set. This workflow can be run multiple times in parallel to shorten evaluation time. The checkpoint that achieves the lowest CRF loss on validation set is selected as the best checkpoint.
```
python NetTIME_CRF_evaluate.py \
--batch_size 2700 \
--num_workers 10 \
--dataset "data/datasets/training_example/validation_minOverlap200_maxUnion600_example.h5" \
--prediction_dir "experiments/training_example/validation_predict/" \
--class_weight "data/datasets/training_example/validation_minOverlap200_maxUnion600_example_weight.npy" \
--experiment_name "training_example"
```

Make CRF classification predictions on validation set.
```
python NetTIME_CRF_predict.py \
--batch_size 2700 \
--num_workers 10 \
--prediction_dir "experiments/training_example/validation_predict/" \
--class_weight "data/datasets/training_example/validation_minOverlap200_maxUnion600_example_weight.npy" \
--dtype "VALIDATION" \
--experiment_name "training_example"
```

## Report performance on test set

Report model performance by generating NetTIME and CRF predictions using test data.
```
# NetTIME binding probability predictions
python NetTIME_predict.py \
--batch_size 2700 \
--num_workers 10 \
--dataset "data/datasets/training_example/test_minOverlap200_maxUnion600_example.h5" \
--index_file "data/embeddings/example.pkl" \
--experiment_name "training_example"

# CRF classification predictions
python NetTIME_CRF_predict.py \
--batch_size 2700 \
--num_workers 10 \
--prediction_dir "experiments/training_example/test_predict/" \
--experiment_name "training_example"
```