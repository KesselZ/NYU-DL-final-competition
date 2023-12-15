# NYU-DL-final-competition
**Classification_Evaluation**: Use to quickly check the performance and visualization

**Label_the_Unlabel**: Use Semantic segmentation model to label the unlabeled dataset.i.e.generate the mask.npy for each vedio

**main_classification**: main file, for training

**Result_Generator**: for generating the final pt file, used for submission and grade.

**Seg_evaluation**: Use to quickly check the performance and visualization for the Semantic segmentation model.

**seg_main**: for training the Semantic segmentation model.

**utils**: some useful functions.

If you want to use the pre-trained weight of my work, you can download them here and put them in the weight folder:

[Semantic segmentation model(UNet)](https://drive.google.com/file/d/1pGPOE57lN367BO2R_0nXfuJARZlRyAJG/view?usp=drive_link)

[Vedio frame prediction(SimVP)](https://drive.google.com/file/d/1D7Grb93zwZYjZZJfsmb58O_m39ZWBmdt/view?usp=sharing)


**How to reproduce**:

1. Make sure we have four dataset in this directory(val, hidden, train, test). They are not here for now because that is too large.

2. We need to first train a semantic segmentation model. Run seg_main.py, it will train a UNet for 20 epochs and give us this model. To skip this part, you can download the pre-trained weight(links above).

3. Make sure we have the path "weight/unet_20.pth". Then run Label_the_Unlabel.py, and it will label 2000 unlabeled data for the hidden set. (generate mask.npy for each video).

4. Now change line 7 of Label_the_Unlabel.py to LABEL_DATASET="unlabeled", then run it again. We do this because we want to generate labels for unlabeled datasets as well.

5. Make sure each unlabeled data has a pseudo-label, then run main_classification.py. It will train the SimVP model for 100 epochs. Then we can get the final model that could be used to generate results.

6. Run Result_Generator.py, it will read the hidden dataset and generate the final .pt file with shape (2000,160,240).


Result: Our model achieved IOU of 44.44 in the hidden set and we won first place in the competition.
