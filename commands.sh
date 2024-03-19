# train on synthetic data
python3 -m yolox.tools.train -n yolox_s_tree_pose --task human_pose --dataset tree_kpts -b 32 --fp16 --occupy\
    -c './pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth' --workers 12

# export onnx
python3 tools/export_onnx.py \
    --output-name pretrained_models/yolox_s_tree_pose.onnx \
    -f exps/default/yolox_s_tree_pose.py \
    --task human_pose \
    --export-det \
    -c YOLOX_outputs/yolox_s_tree_pose/best_ckpt.pth

# run inference
python3 kpts_onnx_inference.py \
    --model pretrained_models/yolox_s_tree_pose.onnx \
    --output_dir YOLOX_outputs/inference \
    --score_thr 0.8 \
    --image_path ./datasets/mark_forest/IMG_20240318_124218.png

python3 -m yolox.tools.train -n yolox_s_tree_pose --task human_pose --dataset tree_kpts -b 32 --fp16 --occupy\
    -c './pretrained_models/2024-03-18_synth43_best.pth' --workers 12