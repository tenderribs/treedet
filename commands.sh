# train on synthetic data
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 \
    -c pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth --model-size m --dataset synth43k

# train on real data
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 50 \
    -c pretrained_models/2024-03-27_yolox_l_synth43k_best.pth --model-size l --dataset cana100

# export onnx
python3 tools/export_onnx.py \
    --output-name pretrained_models/yolox_l_cana100.onnx \
    --model-size l \
    -f exps/default/yolox_tree_pose.py \
    -c YOLOX_outputs/yolox_l_tree_pose/best_ckpt.pth

# run inference
python3 kpts_onnx_inference.py \
    --model pretrained_models/yolox_s_cana100_tree_pose_inline_decode.onnx \
    --output_dir YOLOX_outputs/inference_s \
    --images_path ./datasets/mark_forest \
    --score_thr 0.5

# run evaluation
python3 eval_kpts.py \
    -f exps/default/yolox_tree_pose.py \
    --model-size s \
    -c pretrained_models/2024-03-19_cana100_best.pth \
    --testset \
    --dataset synth43k