# Embeddings generation mechanisms


- CPC: Contrastive predictive coding. Possible application of BERT-like Cloze task for event sequences would lead to the CPC algorithm.
- MeLES: Metric Learning for Event Sequences
- Alien events detection similar to ELECTRA
- Entity mix: share of the same transactions define distance between entities. Entities are created as chimeras by sampling transactions from several entities in different proportions.
- Time-based algebra: temporal difference based loss for embeddings ($t_l - t_k = t_{k:l}$)
- Divergent search for embeddings generation, compression and tabular data embeddings (Divergent search for few-shot image classification)
- Dual transfer learning (Ddtcdr: Deep dual transfer cross domain recommendation.)
- Autoencoder for embedding compression and tabular data embeddings
- BYOL embeddings
- Sequence order prediction embeddings, like sentence order prediction task in ALBERT
- Next sequence prediction embeddings, like sentence order prediction task in BERT
- Graph-based embeddings. Graph embeddings can be created on top of other type of node or edge embeddings, or trained in the end-to-end manner. Synthetic tasks may include link prediction.
- Prototypical embeddings (Prototypical contrastive learning of unsupervised represen-tations)

# Embeddings quality test

Different synthetic downstream tasks can be used to assess the quality of the embeddings. Possible synthetic downstream tasks include expence categories distribution prediction for transactional embeddings and link prediction for graph embeddings.