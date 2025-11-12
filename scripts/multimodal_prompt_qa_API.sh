total_shard_num=24
MODEL=gpt-5-mini

for ((i=0; i<total_shard_num; i++)); do
    python src/multimodal_prompt_qa/api.py \
        --shard_num $total_shard_num \
        --shard_index $i \
        --prompt_type rectangle \
        --model ${MODEL} &
done

wait