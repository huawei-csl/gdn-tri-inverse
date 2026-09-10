export ATTENTION_BACKEND="${ATTENTION_BACKEND:-ascend}"
export ASCEND_RT_VISIBLE_DEVICES=3 # Comma-separated device ids
export BATCH_SIZE=8
export INPUT_LEN=4096
export OUTPUT_LEN=32
export N_ITERS=3
export MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export NUM_NPUS=2

export SOLVE_TRIL_BACKEND="default"
export GDN_USE_MEGA_GDN=0
echo " "
echo "#"
echo "# Starting case: BACKEND=${SOLVE_TRIL_BACKEND} / MEGA_GDN=${GDN_USE_MEGA_GDN}"
echo "#"
echo " "
for i in $(seq 1 $N_ITERS)
do
python3 -m sglang.bench_one_batch \
    --model-path ${MODEL_NAME} \
    --attention-backend ${ATTENTION_BACKEND} \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size ${NUM_NPUS} \
    --mem-fraction-static 0.7 \
    --batch-size ${BATCH_SIZE} \
    --input-len ${INPUT_LEN} \
    --output-len ${OUTPUT_LEN}
done
mv result.jsonl result_${BATCH_SIZE}_${INPUT_LEN}_default.jsonl


export SOLVE_TRIL_BACKEND="pto-mxr"
export GDN_USE_MEGA_GDN=0
echo " "
echo "#"
echo "# Starting case: BACKEND=${SOLVE_TRIL_BACKEND} / MEGA_GDN=${GDN_USE_MEGA_GDN}"
echo "#"
echo " "
for i in $(seq 1 $N_ITERS)
do
python3 -m sglang.bench_one_batch \
    --model-path ${MODEL_NAME} \
    --attention-backend ${ATTENTION_BACKEND} \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size ${NUM_NPUS} \
    --mem-fraction-static 0.7 \
    --batch-size ${BATCH_SIZE} \
    --input-len ${INPUT_LEN} \
    --output-len ${OUTPUT_LEN}
done
mv result.jsonl result_${BATCH_SIZE}_${INPUT_LEN}_pto.jsonl


export SOLVE_TRIL_BACKEND="pto-mxr"
export GDN_USE_MEGA_GDN=1
echo " "
echo "#"
echo "# Starting case: BACKEND=${SOLVE_TRIL_BACKEND} / MEGA_GDN=${GDN_USE_MEGA_GDN}"
echo "#"
echo " "
for i in $(seq 1 $N_ITERS)
do
python3 -m sglang.bench_one_batch \
    --model-path ${MODEL_NAME} \
    --attention-backend ${ATTENTION_BACKEND} \
    --disable-cuda-graph \
    --disable-radix-cache \
    --tp-size ${NUM_NPUS} \
    --mem-fraction-static 0.7 \
    --batch-size ${BATCH_SIZE} \
    --input-len ${INPUT_LEN} \
    --output-len ${OUTPUT_LEN}
done
mv result.jsonl result_${BATCH_SIZE}_${INPUT_LEN}_mega.jsonl