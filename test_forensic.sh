python inference_dir.py --ckpt ir_101 --save_dir Results/TAR@FAR_Adaface-Forensic/  > Results/TAR@FAR_Adaface-Forensic/outputs.txt 
python inference_dir.py --ckpt ir_101-sketch --save_dir Results/TAR@FAR_Adaface-Real-Forensic/  > Results/TAR@FAR_Adaface-Real-Forensic/outputs.txt
python inference_dir.py --ckpt ir_101-synthsketch --save_dir Results/TAR@FAR_Adaface-Synthetic-Forensic/ > Results/TAR@FAR_Adaface-Synthetic-Forensic/outputs.txt
python inference_dir.py --ckpt ir_101-synthreal --save_dir Results/TAR@FAR_Adaface-Synthreal-Forensic/  > Results/TAR@FAR_Adaface-Synthreal-Forensic/outputs.txt

python eval_identification.py --f Results/TAR@FAR_Adaface-Forensic/Viewed_scores.txt  > Results/TAR@FAR_Adaface-Forensic/outputs_open.txt
python eval_identification.py --f Results/TAR@FAR_Adaface-Real-Forensic/Viewed_scores.txt > Results/TAR@FAR_Adaface-Real-Forensic/outputs_open.txt
python eval_identification.py --f Results/TAR@FAR_Adaface-Synthetic-Forensic/Viewed_scores.txt > Results/TAR@FAR_Adaface-Synthetic-Forensic/outputs_open.txt
python eval_identification.py --f Results/TAR@FAR_Adaface-Synthreal-Forensic/Viewed_scores.txt  > Results/TAR@FAR_Adaface-Synthreal-Forensic/outputs_open.txt






