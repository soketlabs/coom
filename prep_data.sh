python ./tools/preprocess_data.py \
    --input /projects/data/teams/coom_team/training/data_prep/wikitext_train.jsonl \
    --output-prefix /projects/data/teams/coom_team/training/data_prep/wikitext \
    --vocab-file /projects/data/teams/coom_team/training/model_info/tokenizer/vocab.json \
    --merge-file /projects/data/teams/coom_team/training/model_info/tokenizer/merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --workers 32 \
    --append-eod
