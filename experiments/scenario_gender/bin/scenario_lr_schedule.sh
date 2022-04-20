# ReduceLROnPlateau
export SC_SUFFIX="lr_reduce_on_plateau"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="lr_reduce_on_plateau_x2epochs"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    trainer.max_epochs=300 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# CosineAnnealing
export SC_SUFFIX="lr_cosine_annealing"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.CosineAnnealing=true \
    params.train.lr_scheduler.n_epoch=150 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_lr_schedule.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_lr_schedule.txt",
      auto_features: ["../data/emb__lr_*.pickle"]'
