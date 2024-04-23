dataset="mark_forest"

declare -a models=(
    "yolox_s_cana100_try2_norm.onnx"
    "yolox_s_canawiki200_frozen.onnx"
    "yolox_s_canawiki200_l1.onnx"
    "yolox_s_canawiki325_sparse_nol1.onnx"
    "yolox_s_canawiki325_sparse_l1.onnx"
)

for model in "${models[@]}"
do
    python3 kpts_onnx_inference.py \
    --model pretrained_models/"$model" \
    --output_dir YOLOX_outputs/inf_"$model" \
    --images_path ./datasets/"$dataset" \
    --score_thr 0.6
done

