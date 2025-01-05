## Data Preparation

Here we show the tree structure in the `data` folder under the project (root) path `$root`.

```
|--- data
|   |--- videos
|   |   |--- TaCoS
|   |   |--- QVHighlights
|   |   |--- Charades
|   |--- tacos
|   |--- qvhighlights
|   |--- charades
...
```

- Download raw videos for the three datasets and put them under `data/videos/$dataset_name`

- Download metadata files and pre-extracted features of frozen extractors from [here](https://github.com/showlab/UniVTG/blob/main/install.md), and put them under `data/$dataset_name`

## Training and Evaluation

- Run our ActPrompt with the following scripts (You should first change `$root` path in these scripts)

```bash
cd extractor
bash ./script/run_cha.sh # for Charades-STA
bash ./script/run_tacos.sh # for TACoS
bash ./script/run_qv.sh # for QVHighlights
```

Notably, when you run the above scripts for the first time, it will take a bit longer (fewer than 10 hours) to extract cache files for all videos in a specific dataset according to each moment-query pair's video interval automatically. After that, training a domain-adaptive model will take 1-2 hours every time.

- Train and evaluate the model using ActPrompt's features on downstream datasets (you should first change `$root` path in these scripts)

```bash
# e.g. training on Charades-STA based on Moment-DETR
bash ./moment_detr/scripts/train_cha.sh

# e.g. evaluating on QVHighlights based on QD-DETR (Only on QVHighlights do you need to test specifically on the test set and run the inference script)
bash ./qd_detr/scripts/inference.sh
```

When conducting experiments on QVHighlights, you can find `hl_val_submission.jsonl` and `hl_test_submission.jsonl` in the `$result` directory. You should zip them and submit them on the official platform.
