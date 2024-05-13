# train on synthetic data
python3 -m yolox.tools.train -n yolox_tree_pose -b 32 --fp16 --occupy --workers 12 --max-epoch 75 \
-c pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth --model-size s --dataset synth43k

# train on real data
python3 -m yolox.tools.train -n yolox_tree_pose -b 16 --fp16 --occupy --workers 12 --max-epoch 50 \
-c pretrained_models/2024-03-27_yolox_l_synth43k_best.pth --model-size l --dataset canawikisparse325

# export onnx
python3 -m yolox.tools.export_onnx \
--output-name pretrained_models/yolox_l_canawikisparse325_l1.onnx \
--model-size l \
-f exps/default/yolox_tree_pose.py \
-c pretrained_models/2024-04-24_yolox_l_canawikisparse325_l1.pth \
--dataset canawikisparse325

# run inference
python3 kpts_onnx_inference.py \
--model pretrained_models/yolox_s_canawiki325_sparse_nol1.onnx \
--output_dir YOLOX_outputs/inf_canawiki325sparse \
--images_path ./datasets/mark_forest \
--score_thr 0.85

# create trt engine
trtexec \
--onnx=treedet/pretrained_models/yolox_s_canawiki325_sparse_nol1.onnx \
--saveEngine=treedet/pretrained_models/yolox_s_canawiki325_sparse_nol1.engine

# run evaluation
python3 eval_kpts.py \
-f exps/default/yolox_tree_pose.py \
--model-size s \
-c pretrained_models/2024-03-19_cana100_best.pth \
--testset \
--dataset synth43k
