total_shard_num=24
# INPUT="output/multimodal_prompt_qa_rectangle/gpt-5-mini.jsonl"

for ((i=0; i<total_shard_num; i++)); do
    python src/eval_multimodal_prompt_qa.py \
        --shard_num ${total_shard_num} \
        --shard_index ${i} \
        --input ${INPUT} &
done

wait