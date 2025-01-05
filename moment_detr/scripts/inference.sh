ckpt_path=results/moment_detr/qvhighlights-1-2024_02_26_17/model_best.ckpt
eval_split_name=test
eval_path=data/qvhighlights/metadata/qvhighlights_test.jsonl

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
