Chat SegFormer Road Segmentation - organized version

1. Project structure
   configs/     training/path configuration
   datasets/    dataset loader
   models/      model construction
   losses/      loss functions
   utils/       metrics and utility functions
   train.py     training entry
   evaluate.py  validation metrics
   predict.py   SegFormer-only prediction
   weights/     trained weights
   results/     prediction outputs

2. Train
   python train.py

3. Evaluate
   python evaluate.py

4. Predict
   python predict.py

5. Important
   The original training logic is preserved:
   - input size: 384x640
   - batch size: 2
   - 20 epochs
   - first 5 epochs freeze encoder
   - from epoch 6 unfreeze encoder
   - AMP on CUDA
   - weighted CE + Dice loss
   - best model saved as weights/best_segformer_road.pth

6. Note
   fusion_predict.py was separated from the baseline prediction entry because
   the current YOLO fusion uses bounding boxes as rectangular masks. The
   reorganized predict.py intentionally tests SegFormer alone so that the
   SegFormer baseline can be measured independently before compression or
   fusion experiments.
