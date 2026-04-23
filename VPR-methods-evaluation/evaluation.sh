python /teamspace/studios/this_studio/Visual_Place_Recognition_Project/VPR-methods-evaluation/main.py \
  --num_workers 8 \
  --batch_size 32 \
  --method megaloc \
  --image_size 512 512 \
  --database_folder '/teamspace/studios/this_studio/data/svox/images/test/gallery' \
  --queries_folder '/teamspace/studios/this_studio/data/svox/images/test/queries_sun' \
  --distance_metric L2 dot_product \
  --log_dir 'megaloc/svox_sun'