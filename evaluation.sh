python main.py \
  --database_folder /teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/svox/images/test/gallery \
  --queries_folder /teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/svox/images/test/queries_night \
  --method netvlad \
  --distance_metric L2 \
  --log_dir netvlad_predictions \
  --num_preds_to_save 20