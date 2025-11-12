total_shard_num=24
MODEL=gpt-5-mini

for ((i=0; i<total_shard_num; i++)); do
    python src/text_prompt_qa/api.py \
        --shard_num $total_shard_num \
        --shard_index $i \
        --model ${MODEL} &
done

wait