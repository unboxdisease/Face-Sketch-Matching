    # --resume_from_checkpoint ./pretrained/adaface_ir101_webface4m.ckpt \
CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --resume_from_checkpoint ./experiments/ir101_ms1m_baseline_04-17_4/last.ckpt \
    --data_root /scratch0 \
    --train_data_path GenFaceSketch_aligned_75_3 \
    --val_data_path webfaces \
    --prefix ir101_GenFaceSketchMAX+ \
    --use_wandb \
    --gpus 2 \
    --use_16bit \
    --arch ir_101 \
    --batch_size 64 \
    --num_workers 16 \
    --epochs 60 \
    --lr_milestones 45,47,49 \
    --lr 0.001 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.5 \
    --crop_augmentation_prob 0.5 \
    --photometric_augmentation_prob 0.5

