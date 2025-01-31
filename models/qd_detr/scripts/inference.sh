ckpt_path=results/qd_detr/folder/model_best.ckpt
eval_split_name=test
eval_path=/root/data/qvhighlights/metadata/qvhighlights_test.jsonl

PYTHONPATH=$PYTHONPATH:. python models/qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
