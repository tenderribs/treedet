# train on synthetic data
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 75 \
    -c pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth --model-size s --dataset synth43k

# train with frozen backbone weights
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 75 \
    -c pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth --model-size s --dataset synth43k --freeze --no_normalize_ds

# train on real data
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 50 \
    -c pretrained_models/2024-04-10_yolox_s_synth43k_norm_nomosaic.pth --model-size s --dataset canawikisparse325

# train with frozen backbone weights
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 50 \
    -c pretrained_models/2024-04-22_yolox_s_synth43k_frozen.pth --model-size s --dataset canawiki200 --freeze --no_normalize_ds

# export onnx
python3 -m yolox.tools.export_onnx \
    --output-name pretrained_models/yolox_s_canawiki325sparse.onnx \
    --model-size s \
    -f exps/default/yolox_tree_pose.py \
    -c pretrained_models/2024-04-22_yolox_s_canawiki325sparse.pth \
    --dataset canawiki200

# run inference
python3 kpts_onnx_inference.py \
    --model pretrained_models/yolox_s_canawiki325sparse.onnx \
    --output_dir YOLOX_outputs/inf_canawiki325sparse \
    --images_path ./datasets/mark_forest \
    --score_thr 0.85

# create trt engine
trtexec \
        --onnx=pretrained_models/yolox_s_cana100_tree_pose_inline_decode.onnx \
        --saveEngine=pretrained_models/yolox_s_cana100_tree_pose_inline_decode.engine

python3 kpts_trt_inference.py \
    --model pretrained_models/yolox_s_cana100_tree_pose_inline_decode.engine \
    --output_dir YOLOX_outputs/trtout \
    --images_path ./datasets/mark_forest \
    --score_thr 0.8
    --score_thr 0.5

# run evaluation
python3 eval_kpts.py \
    -f exps/default/yolox_tree_pose.py \
    --model-size s \
    -c pretrained_models/2024-03-19_cana100_best.pth \
    --testset \
    --dataset synth43k
