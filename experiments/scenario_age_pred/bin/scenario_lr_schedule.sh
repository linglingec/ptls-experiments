# ReduceLROnPlateau
export SC_SUFFIX="reduce_on_plateau"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    +params.lr_scheduler.ReduceLROnPlateau=true \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="reduce_on_plateau_x2epochs"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    trainer.max_epochs=200 \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# CosineAnnealing
export SC_SUFFIX="cosine_annealing"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lr_scheduler.n_epoch=150 \
    params.lr_scheduler.CosineAnnealing=true \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params


# Compare
rm results/rm results/scenario_lr_schedule.txt

python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_lr_schedule.txt",
      auto_features: [
          "../data/emb__reduce_on_plateau.pickle", 
          "../data/emb__reduce_on_plateau_x2epochs.pickle",
          "../data/emb__cosine_annealing.pickle"]'
