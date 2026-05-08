python main.py \
  --database_folder /teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/svox/images/train/gallery \
  --queries_folder /teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/svox/images/train/queries_sun \
  --method mixvpr \
  --distance_metric L2 \
  --log_dir mixvpr_prediction \
  --num_preds_to_save 20