### Circle Object Detection

#### Development Env
Python 3.6
  - numpy==1.16.4
  - tensorflow==1.13.1
  - scikit-image==0.16.2
  - matplotlib==2.2.2
  - shapely==1.6.4.post 

#### Run Experiement
To evaluate with pre-trained model, simply run:
```
python main.py
```

To train model:
```
python train_model.py
```
(The log loss history will be written to __output.txt__)

#### Model Description
- Preprocess:
    1. Min-max normalization (divide by 3)
    2. Thresholding (0.7)
    3. Subsampling (2x2, max-sampling)
- Train-eval split:
    1. Small: 800 train, 200 eval
    2. Medium: 8000 train, 2000 eval
    3. Large: 40000 train, 10000 eval
- Network Topology (defined in __train_model.py__): Conv\_3\*10 + Conv\_3\*10 + Conv\_2\*10 + Conv\_2\*10 + Dense(50) + Dense(20) + Dense(10)
- Loss Function: mean squared error (performed better than huber loss)
- Optimizer: ADAM (lr = 0.001)
- Regularization: None (due to time issue)
- Trained for 150 epochs, batch size = 50
- Pretrained model checkpoint: **model-0150.ckpt**

#### Evaluation
- (Averge over 1000 samples w/t noise level = 2) IOU@0.7: 0.9673 (std\_dev = 0.00642 for 6 experiments)

