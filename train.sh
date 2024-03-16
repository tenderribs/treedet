python3 -m yolox.tools.train -n yolox_s_tree_pose --task human_pose --dataset tree_kpts -b 1 --fp16 --occupy\
    -c './pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth'
# batch size 1, # mixed precision, occupy memory in advance