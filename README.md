# Logs for the competition

# ADPOS diabetic retina

# Models on training:

* `248_efficientnet-b0_f1_test`: First model, 224x, B0, 1e-5, 50 epochs.
training looks good
ckpt50: 0.845 on LB, same with ckpt80, train/val: 0.86/0.90
* b3 is 0.90 on LB, 0.98 on train set. overfitting.
* b2 b5 are 0.87

resizing images to square is shitty.


* `248_efficientnet-b3_f0_test`: with resize_sa, padifneeded border_mode=1, size=300.

ckpt25: train/val: 0.97/0.97, LB: 0.9278
ckpt50 and 20 give lower scores.
*strang* ensemble of b3-f1's 0.90 and b3-f0's 0.92 gives: 0.878

* `248_efficientnet-b3_f2_test`: with strong_aug()




# Questions and Ideas:





# TODO:


# Revelations:

* mpimg.imread returns normalized image in [0, 1], cv2.imread returns it in [0, 255] (255 as maximum pixel value)
* plt.imshow needs image array to be in [0, 1] or [0, 255]
* albumentations' rotate shits in the interpolated area.
* .detach() makes variable' grad_required=False, the variable still remains on CUDA, make as many computations as possible on CUDA, use .cpu() in rarest of the cases, or when you wanna change the tensors to numpy use .cpu().numpy() because it can't be directly changed to numpy variable without shifting it to host memory, makes sense.
* test.py with tta=False, takes about 2 mins for first predictions, about 16 seconds for subsequent predictions, boi now you know what pin_memory=True does.
* for tta you don't have to pass image from each augmentation and then take the average, one other approach is to predict multiple times and take average and as the augmentationss are random, you get whachyouwantmyboi.
* loc for pd series and iloc for pd df.
* The resume code had a bug, if you'd resume you'll start with base_lr = top_lr * 0.001, and if the start epoch was greater than say 10, it will remain the same.
* The public test data is 15% of total test data and is used for public LB scoring. The private test set (85% of total) Will be used for private LB scoring.
* Updated the dataset to a new version? Just reboot the kernel to reflect that update (no remove and add shit)
* First epoch may not have full utilization, next epoch it'll be full thanks to pin_memory.
* pil image.size returns w, h, np.array/ cv2 return h, w


# Things to check, just before starting the model training:

* config
* get item in dataloader.



# Observations:


# NOTES:



# Files informations:



# remarks shortcut/keywords



# Experts speak






EXTRAS:

EfficientNet params:
  * (width_coefficient, depth_coefficient, resolution, dropout_rate)
  'efficientnet-b0': (1.0, 1.0, 224, 0.2),
  'efficientnet-b1': (1.0, 1.1, 240, 0.2),
  'efficientnet-b2': (1.1, 1.2, 260, 0.3),
  'efficientnet-b3': (1.2, 1.4, 300, 0.3),
  'efficientnet-b4': (1.4, 1.8, 380, 0.4),
  'efficientnet-b5': (1.6, 2.2, 456, 0.4),
  'efficientnet-b6': (1.8, 2.6, 528, 0.5),
  'efficientnet-b7': (2.0, 3.1, 600, 0.5),


General Lessons:
* Don't use full words in model folder names, use f0, instead of fold0
