# MultiBPE

A multitask learning framework for predicting cell-type specific transcription factor binding sites with base-pair resolution.
<p align="center"><img width="90%" src="figures/architecture.png" /></p>

<!---
## Citation
If you use MultiBPE in your research, please cite:
-->

## Installation

Run the following commands to clone the repository and install MultiBPE:
```
git clone https://github.com/ryi06/MultiBPE.git
cd MultiBPE

# Create a Conda environment with Python 3.6 and install required packages
conda env create --file environment.yml
conda activate multibpe
```
Please refer to [Pytorch documentation](https://pytorch.org/) for instructions on setting up CUDA.

## Making predictions using a trained MultiBPE model

We use `MultiBPE_predict.py` to make predictions from a trained MultiBPE model. Our pretrained models can be found [here](pretrained/). We provide an [example prediction dataset](data/datasets/prediction_example/). Check out [this tutorial](preprocess/README.md#generating-data-to-make-predictions-from-a-trained-model) on how to generate dataset like this from a bed file.

Making binding probability predictions for  JUN.K562 and JUNB.GM12878 from a trained MultiBPE model can be achieved by running the following. See `MultiBPE_predict.py -h` for all available arguments. 
```
python MultiBPE_predict.py \
--batch_size 2700 \
--num_workers 10 \
--dataset "data/datasets/prediction_example/predict_example.h5" \
--dtype "prediction" \
--index_file "data/embeddings/pretrained.pkl" \
--experiment_name "prediction_example" \
--model_config "pretrained/seq_CT/seq_CT.config" \
--best_ckpt "pretrained/seq_CT/seq_CT_270409.ckpt" \
--eval_metric "aupr" \
--no_target \
--predict_groups "JUN.K562" "JUNB.GM12878"
```
Binding probability predictions will be saved in `experiments/prediction_example/prediction_predict`. If you wish to further perform binary classification on the predicted binding probabilities using a pretrained conditional random field (CRF) classifier, run `MultiBPE_CRF_predict.py` as follows:
```
python MultiBPE_CRF_predict.py \
--batch_size 2700 \
--num_workers 10 \
--prediction_dir "experiments/prediction_example/prediction_predict" \
--experiment_name "prediction_example" \
--dtype "prediction" \
--model_config "pretrained/seq_CT/seq_CT_crf.config" \
--best_ckpt "pretrained/seq_CT/seq_CT_crf_224422.ckpt"
```

## Training a MultiBPE model

A tutorial on how to train a MultiBPE model using [example training data](data/datasets/training_example/) can be found in [training_example.md](training_example.md).



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI0MTA5ODkzMCwtMTkxMzM4NzgyNCwxND
U0MDAxMTc0LC0xNDU3MTk4ODA1LC03NTY3MjE2MzIsMTg4MjY1
NzI5LDkzNjcxNzc3MSwyMTExMDA1OTEyLC0xNjI5NDI2MDA0LD
E2NTUzODI1MzMsLTE5OTIxMzY1NDAsODQyNTE1MjQ0LC02NDAw
NTYyNjUsLTExNDYyNTk1ODYsMTQ2Mzk3MDA5MywtMjY5NTMwOD
YxLC0yMTMwMDU4NjkxLDE3OTYyODY2MTksMTg3MjQ3NjEyNF19

-->