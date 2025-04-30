# Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks

This repository contains the code and scripts used for our paper published at NAACL 2025 [Paper Link](https://aclanthology.org/2025.findings-naacl.426.pdf). <br/>

The code and scripts in this repository make use of the code in [HaRT-Wrapper Gitub repository](https://github.com/soni-n/HaRT-Wrapper/tree/main)(this repository has the code for training HaRT-concat and HaRT-ODPB variants of HaRT referred in our paper.

# Install HaRT

### Requires Python 3.x (tested with Python 3.8)
```
pip install hart-wrapper
```


# Generating embeddings
Use <i>generate_embeddings.py</i> to fetch user, document, and word level representations from HaRT. It can also generate document-level representations from GPT2, BERT, and RoBERTa.

Template for HaRT embeddings
```
python hart_util.py <embedding type> --model_path 'hlab/hart-gpt2sml-twt-v1' --mode 'concat' --block_size 1024 --representative_layer second_last --data <path to .pkl dataset> --user_id_column <user_id> --text_id_column <message_id> --text_column <message> --order_by_column <updated_time> --save_dir <path to store embeds>
```

Template for other models
```
python --model_name <gpt2/bert/roberta> --data <data_path> --batch_size <batch_size> --text_column <text_column> --save_dir <path to store embeds>
```

The embeddings are returned in the formats -
| Embedding Type | `return_pt=True` (PyTorch Tensor)            | `return_pt=False` (Python List)                                |
|----------------|-----------------------------------------------|---------------------------------------------------------------|
| **User**       | Tensor of shape `(num_users, 768)`             | List of length `num_users`, each element a list of 768 floats  |
| **Document**   | Tensor of shape `(num_docs, 768)`              | List of length `num_docs`, each element a list of 768 floats   |
| **Word**       | List of `num_docs` Tensors, each of shape `(num_words, 768)` | Nested list: outer length `num_docs`, each containing a list of `num_words` lists of 768 floats |


# Evaluation
We evaluate the embeddings using the RidgeCV model. You may use scikit-learn's implementation directly. We use DLATK for this.

First, we convert the embeddings into [DLATK feature table](https://dlatk.github.io/dlatk/tutorials/tut_feat_tables.html) format and upload them to an SQL database.

<i>hart_embed_to_dlatk_csv.py</i> converts HaRT embeddings to DLATK feat. table in CSV format. This utility also helps aggregate embeddings at various levels such as words to documents and documents to users, waves, etc.

<i>otherlm_embed_to_dlatk_csv.ipynb</i> contains the code for this task for GPT2, BERT, and RoBERTa.

We then use DLATK commands of the following template for regression.

```
python3.8 dlatkInterface.py -d <SQL DB name> -t <table with user ids, text ids, and texts> -g <aggregation column> -f <DLATK feature table> --outcome_table <table with ground truths (can be same as -t)> --outcomes <column name for label> --fold_column <column with fold allocation> --nfold_test_reg --model ridgehighcv --group_freq_thresh 10
```
