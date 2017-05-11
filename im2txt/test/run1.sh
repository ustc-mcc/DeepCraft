HOME='/home/nimbix'
CHECKPOINT_PATH="${HOME}/im2txt/data/model/model.ckpt-1000000"
VOCAB_FILE="${HOME}/im2txt/data/mscoco/word_counts.txt"
IMAGE_FILE="${HOME}/Server/image-caption-demo/video_cap/video_cap/static/imgs/hackxsjtu/tmp.jpg"
#IMAGE_FILE="${HOME}/im2txt/g3doc/COCO_val2014_000000224477.jpg"
#bazel build -c opt im2txt/run_inference
python /home/nimbix/im2txt/im2txt/run_inference.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}
