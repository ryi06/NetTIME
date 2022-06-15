In this directory, we provide two example datasets.

1. `training_example`: example dataset used for training NetTIME models and evaluating model performance. The dataset includes DNA sequence feature, DNase-seq as cell-type-specific feature, and HOCOMOCO motif enrichment as TF-specific feature.
2. `prediction_example`: example dataset used for making model predictions. The dataset includes DNA sequence feature, DNase-seq, H3K4me1 ChIP-seq, H3K4me3 ChIP-seq and H3K27ac ChIP-seq as cell-type-specific features.

NetTIME dataset used to generate main results in the manuscript can be downloaded from [here](https://drive.google.com/drive/folders/1hOTpf1eNw7Eb2QwHgqrnNBrDl1fBl3p8?usp=sharing). If you wish to use this dataset, we recommend saving the dataset files under `data/dataset/seq_CTTF`. This dataset includes DNA sequence feature, DNase-seq, H3K4me1 ChIP-seq, H3K4me3 ChIP-seq and H3K27ac ChIP-seq as cell-type-specific features, and HOCOMOCO motif enrichment as TF-specific features.