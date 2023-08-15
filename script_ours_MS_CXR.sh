# MS_CXR train 2022
CUDA_VISIBLE_DEVICES=1 python train.py \
--model_name TransVG_ca \
--seed 2022 \
--batch_size 8 --lr 0.00005 --lr_bert 0.00001 \
--aug_crop --aug_scale --aug_translate \
--backbone resnet50 \
--bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 \
--CAsampleType random --CAsampleNum 5 --CAlossWeightBase 0.05 --CATextPoolType cls --CATemperature 0.1 --CAMode max_image_lcpTriple \
--dataset MS_CXR \
--output_dir checkpoint/Ablation_TaCo_2022 \
--resume "pretrained/TransVG_R50_unc.pth" \
--resume_model_only

# # MS_CXR train 2023
# CUDA_VISIBLE_DEVICES=1 python train.py \
# --model_name TransVG_ca \
# --seed 2023 \
# --batch_size 8 --lr 0.00005 --lr_bert 0.00001 \
# --aug_crop --aug_scale --aug_translate \
# --backbone resnet50 \
# --bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 \
# --CAsampleType random --CAsampleNum 5 --CAlossWeightBase 0.05 --CATextPoolType cls --CATemperature 0.1 --CAMode max_image_lcpTriple \
# --dataset MS_CXR \
# --output_dir checkpoint/Ablation_TaCo_2023 \
# --resume "pretrained/TransVG_R50_unc.pth" \
# --resume_model_only

# # MS_CXR train 2024
# CUDA_VISIBLE_DEVICES=1 python train.py \
# --model_name TransVG_ca \
# --seed 2024 \
# --batch_size 8 --lr 0.00005 --lr_bert 0.00001 \
# --aug_crop --aug_scale --aug_translate \
# --backbone resnet50 \
# --bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 \
# --CAsampleType random --CAsampleNum 5 --CAlossWeightBase 0.05 --CATextPoolType cls --CATemperature 0.1 --CAMode max_image_lcpTriple \
# --dataset MS_CXR \
# --output_dir checkpoint/Ablation_TaCo_2024 \
# --resume "pretrained/TransVG_R50_unc.pth" \
# --resume_model_only

# MS_CXR test 2022
python -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py \
--model_name TransVG_ca \
--batch_size 32 --num_workers 4 \
--bert_enc_num 12 --detr_enc_num 6 \
--backbone resnet50 --dataset MS_CXR \
--max_query_len 20 --eval_set test \
--CAMode max_image_lcpTriple \
--eval_model checkpoint/Ablation_TaCo_2022/best_miou_checkpoint.pth \
--output_dir checkpoint/Ablation_TaCo_2022

# # MS_CXR test 2023
# python -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py \
# --model_name TransVG_ca \
# --batch_size 32 --num_workers 4 \
# --bert_enc_num 12 --detr_enc_num 6 \
# --backbone resnet50 --dataset MS_CXR \
# --max_query_len 20 --eval_set test \
# --CAMode max_image_lcpTriple \
# --eval_model checkpoint/Ablation_TaCo_2023/best_miou_checkpoint.pth \
# --output_dir checkpoint/Ablation_TaCo_2023

# # MS_CXR test 2024
# python -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py \
# --model_name TransVG_ca \
# --batch_size 32 --num_workers 4 \
# --bert_enc_num 12 --detr_enc_num 6 \
# --backbone resnet50 --dataset MS_CXR \
# --max_query_len 20 --eval_set test \
# --CAMode max_image_lcpTriple \
# --eval_model checkpoint/Ablation_TaCo_2024/best_miou_checkpoint.pth \
# --output_dir checkpoint/Ablation_TaCo_2024