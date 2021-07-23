import time

import cv2
import numpy as np
import torch
import torchvision.models as models

# PyTorch Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_list = []
model = model_list.append(('AlexNet', models.alexnet(pretrained=True)))

model = model_list.append(('ResNet-152', models.resnet152(pretrained=True)))

model = model_list.append(('ResNext50_32x4d', models.resnext50_32x4d(pretrained=True)))

model = model_list.append(('ResNext101_32x8d', models.resnext101_32x8d(pretrained=True)))

model = model_list.append(('Wide ResNet50_2', models.wide_resnet50_2(pretrained=True)))

model = model_list.append(('Wide ResNet101_2', models.wide_resnet101_2(pretrained=True)))

# load test data
x = []
cap = cv2.VideoCapture('data/gebaerdenlernen/testdata/mp4/null.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (227, 227))
    x.append(frame)
x = np.array(x)

# swap color axis because
# numpy image: I x H x W x C
# torch image: I x C X H X W
x = x.transpose((0, 3, 1, 2))

# convert to float tensor
x = torch.tensor(x).float().to(device)

with torch.no_grad():
    for model_name, model in model_list:
        # gpu acceleration
        model = model.to(device)

        time.sleep(1)
        for idx in range(0, 10):
            then = time.time()
            y = model(x)
            now = time.time()

            print(
                f"Idx: {idx}\t{model_name}\tparam count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\tComputation time: {now-then}\tPred-size: {y.size()}"
            )

    print("Comparison done.")

### TEST RESULTS ###

# On IISY 001 Machine @ 1x GTX 2080Ti | i9 9920

#                                               Top-1   Top-5   Speed
# AlexNet Small param count:      61100840      43.45   20.91   ~0.0004
# ResNet-152 Small param count:   60192808      21.69    5.94   ~0.45
# Resnext Small param count:      25028904      22.38    6.30   ~0.23   <- fatest w/ "low" params w/ best error
# Resnext Large param count:      88791336      20.69    5.47   ~0.69
# Wide ResNet Small param count:  68883240      21.49    5.91   ~0.32
# Wide ResNet Large param count: 126886696      21.16    5.72   ~0.55

# Idx: 0  AlexNet               param count: 61100840   Computation time: 0.20440053939819336   Pred-size: torch.Size([138, 1000])
# Idx: 1  AlexNet               param count: 61100840   Computation time: 0.000514984130859375  Pred-size: torch.Size([138, 1000])
# Idx: 2  AlexNet               param count: 61100840   Computation time: 0.0004763603210449219 Pred-size: torch.Size([138, 1000])
# Idx: 3  AlexNet               param count: 61100840   Computation time: 0.0004723072052001953 Pred-size: torch.Size([138, 1000])
# Idx: 4  AlexNet               param count: 61100840   Computation time: 0.0004711151123046875 Pred-size: torch.Size([138, 1000])
# Idx: 5  AlexNet               param count: 61100840   Computation time: 0.00047135353088378906Pred-size: torch.Size([138, 1000])
# Idx: 6  AlexNet               param count: 61100840   Computation time: 0.0004665851593017578 Pred-size: torch.Size([138, 1000])
# Idx: 7  AlexNet               param count: 61100840   Computation time: 0.00046515464782714844Pred-size: torch.Size([138, 1000])
# Idx: 8  AlexNet               param count: 61100840   Computation time: 0.0004646778106689453 Pred-size: torch.Size([138, 1000])
# Idx: 9  AlexNet               param count: 61100840   Computation time: 0.0004639625549316406 Pred-size: torch.Size([138, 1000])
# Idx: 0  ResNet-152            param count: 60192808   Computation time: 0.03603720664978027   Pred-size: torch.Size([138, 1000])
# Idx: 1  ResNet-152            param count: 60192808   Computation time: 0.3710296154022217    Pred-size: torch.Size([138, 1000])
# Idx: 2  ResNet-152            param count: 60192808   Computation time: 0.44860315322875977   Pred-size: torch.Size([138, 1000])
# Idx: 3  ResNet-152            param count: 60192808   Computation time: 0.45116591453552246   Pred-size: torch.Size([138, 1000])
# Idx: 4  ResNet-152            param count: 60192808   Computation time: 0.4488818645477295    Pred-size: torch.Size([138, 1000])
# Idx: 5  ResNet-152            param count: 60192808   Computation time: 0.45058465003967285   Pred-size: torch.Size([138, 1000])
# Idx: 6  ResNet-152            param count: 60192808   Computation time: 0.45017313957214355   Pred-size: torch.Size([138, 1000])
# Idx: 7  ResNet-152            param count: 60192808   Computation time: 0.4513535499572754    Pred-size: torch.Size([138, 1000])
# Idx: 8  ResNet-152            param count: 60192808   Computation time: 0.4514124393463135    Pred-size: torch.Size([138, 1000])
# Idx: 9  ResNet-152            param count: 60192808   Computation time: 0.4518623352050781    Pred-size: torch.Size([138, 1000])
# Idx: 0  ResNext50_32x4d       param count: 25028904   Computation time: 0.02229785919189453   Pred-size: torch.Size([138, 1000])
# Idx: 1  ResNext50_32x4d       param count: 25028904   Computation time: 0.011590957641601562  Pred-size: torch.Size([138, 1000])
# Idx: 2  ResNext50_32x4d       param count: 25028904   Computation time: 0.03041696548461914   Pred-size: torch.Size([138, 1000])
# Idx: 3  ResNext50_32x4d       param count: 25028904   Computation time: 0.239227294921875     Pred-size: torch.Size([138, 1000])
# Idx: 4  ResNext50_32x4d       param count: 25028904   Computation time: 0.23975682258605957   Pred-size: torch.Size([138, 1000])
# Idx: 5  ResNext50_32x4d       param count: 25028904   Computation time: 0.2390286922454834    Pred-size: torch.Size([138, 1000])
# Idx: 6  ResNext50_32x4d       param count: 25028904   Computation time: 0.2397322654724121    Pred-size: torch.Size([138, 1000])
# Idx: 7  ResNext50_32x4d       param count: 25028904   Computation time: 0.23899292945861816   Pred-size: torch.Size([138, 1000])
# Idx: 8  ResNext50_32x4d       param count: 25028904   Computation time: 0.2396528720855713    Pred-size: torch.Size([138, 1000])
# Idx: 9  ResNext50_32x4d       param count: 25028904   Computation time: 0.24013662338256836   Pred-size: torch.Size([138, 1000])
# Idx: 0  ResNext101_32x8d      param count: 88791336   Computation time: 0.036200523376464844  Pred-size: torch.Size([138, 1000])
# Idx: 1  ResNext101_32x8d      param count: 88791336   Computation time: 0.4391016960144043    Pred-size: torch.Size([138, 1000])
# Idx: 2  ResNext101_32x8d      param count: 88791336   Computation time: 0.6923861503601074    Pred-size: torch.Size([138, 1000])
# Idx: 3  ResNext101_32x8d      param count: 88791336   Computation time: 0.6928420066833496    Pred-size: torch.Size([138, 1000])
# Idx: 4  ResNext101_32x8d      param count: 88791336   Computation time: 0.6937403678894043    Pred-size: torch.Size([138, 1000])
# Idx: 5  ResNext101_32x8d      param count: 88791336   Computation time: 0.6935386657714844    Pred-size: torch.Size([138, 1000])
# Idx: 6  ResNext101_32x8d      param count: 88791336   Computation time: 0.6943824291229248    Pred-size: torch.Size([138, 1000])
# Idx: 7  ResNext101_32x8d      param count: 88791336   Computation time: 0.6944069862365723    Pred-size: torch.Size([138, 1000])
# Idx: 8  ResNext101_32x8d      param count: 88791336   Computation time: 0.6950252056121826    Pred-size: torch.Size([138, 1000])
# Idx: 9  ResNext101_32x8d      param count: 88791336   Computation time: 0.6956069469451904    Pred-size: torch.Size([138, 1000])
# Idx: 0  Wide ResNet50_2       param count: 68883240   Computation time: 0.017294883728027344  Pred-size: torch.Size([138, 1000])
# Idx: 1  Wide ResNet50_2       param count: 68883240   Computation time: 0.008900642395019531  Pred-size: torch.Size([138, 1000])
# Idx: 2  Wide ResNet50_2       param count: 68883240   Computation time: 0.006266593933105469  Pred-size: torch.Size([138, 1000])
# Idx: 3  Wide ResNet50_2       param count: 68883240   Computation time: 0.20997214317321777   Pred-size: torch.Size([138, 1000])
# Idx: 4  Wide ResNet50_2       param count: 68883240   Computation time: 0.3263254165649414    Pred-size: torch.Size([138, 1000])
# Idx: 5  Wide ResNet50_2       param count: 68883240   Computation time: 0.3257770538330078    Pred-size: torch.Size([138, 1000])
# Idx: 6  Wide ResNet50_2       param count: 68883240   Computation time: 0.3261747360229492    Pred-size: torch.Size([138, 1000])
# Idx: 7  Wide ResNet50_2       param count: 68883240   Computation time: 0.3270268440246582    Pred-size: torch.Size([138, 1000])
# Idx: 8  Wide ResNet50_2       param count: 68883240   Computation time: 0.32589197158813477   Pred-size: torch.Size([138, 1000])
# Idx: 9  Wide ResNet50_2       param count: 68883240   Computation time: 0.32631921768188477   Pred-size: torch.Size([138, 1000])
# Idx: 0  Wide ResNet101_2      param count: 126886696  Computation time: 0.024753093719482422  Pred-size: torch.Size([138, 1000])
# Idx: 1  Wide ResNet101_2      param count: 126886696  Computation time: 0.21550226211547852   Pred-size: torch.Size([138, 1000])
# Idx: 2  Wide ResNet101_2      param count: 126886696  Computation time: 0.5512797832489014    Pred-size: torch.Size([138, 1000])
# Idx: 3  Wide ResNet101_2      param count: 126886696  Computation time: 0.5519230365753174    Pred-size: torch.Size([138, 1000])
# Idx: 4  Wide ResNet101_2      param count: 126886696  Computation time: 0.5513582229614258    Pred-size: torch.Size([138, 1000])
# Idx: 5  Wide ResNet101_2      param count: 126886696  Computation time: 0.5529220104217529    Pred-size: torch.Size([138, 1000])
# Idx: 6  Wide ResNet101_2      param count: 126886696  Computation time: 0.5528371334075928    Pred-size: torch.Size([138, 1000])
# Idx: 7  Wide ResNet101_2      param count: 126886696  Computation time: 0.5517127513885498    Pred-size: torch.Size([138, 1000])
# Idx: 8  Wide ResNet101_2      param count: 126886696  Computation time: 0.5529181957244873    Pred-size: torch.Size([138, 1000])
# Idx: 9  Wide ResNet101_2      param count: 126886696  Computation time: 0.553051233291626     Pred-size: torch.Size([138, 1000])
