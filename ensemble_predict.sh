conda activate chemprop_VP1

python predict.py \
    --checkpoint_dir saved_models/molecule_split/Wagner_mols \
    --test_path data/sample_test.csv \
    --preds_path data/output.csv \
    --no_features_scaling \
    --predict_only

conda deactivate