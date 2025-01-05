ckpt_path=results/qd_detr/qvhighlights-1-2024_03_03_22/model_best.ckpt
eval_split_name=test
eval_path=/mnt/data2/tvg/UniVTG/data/qvhighlights/metadata/qvhighlights_test.jsonl

PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
