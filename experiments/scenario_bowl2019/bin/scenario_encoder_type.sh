# GRU encoder
export SC_SUFFIX="encoder_gru"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.rnn.type="gru" \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.rnn.type="lstm" \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/mles_params.hocon


# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_4head_64hs_4layers"
python -m dltranz.pl_train_module \
    ata_module.train.batch_size=32 \
    params.model_type="transf" \
    params.transf.n_heads=4 \
    params.transf.n_layers=4 \
    params.train.batch_size=64 \
    params.valid.batch_size=64 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=128 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_4head_64hs_8layers"
python -m dltranz.pl_train_module \
    data_module.train.batch_size=32 \
    params.model_type="transf" \
    params.transf.n_heads=4 \
    params.transf.n_layers=8 \
    params.train.batch_size=64 \
    params.valid.batch_size=64 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    params.valid.batch_size=32 \
    output.path="data/emb__$SC_SUFFIX" \
    --conf conf/mles_params.hocon

# Compare
rm results/scenario_bowl2019__encoder_types.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler\
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__encoder_types.txt",
      auto_features: ["../data/emb__encoder_*.pickle"]'
