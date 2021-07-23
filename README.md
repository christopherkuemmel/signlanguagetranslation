# Neural Sign Language Translation

This repository contains implementations for Neural Sign Language Translation. 
The code includes two approaches for dealing with the task.

1. NSLT - The first implementation is inspired by the [implementation](https://github.com/neccam/nslt) from GitHub User [neccam](https://github.com/neccam) (written in tensorflow).
   1. ENTRYPOINT: `nslt.py`
   2. It's based on a simple AlexNet which is used to generate the input embeddings
   3. The main encoder-decoder network is implemented with RNNs
2. ROSITA - The second approach is based on my own research.
   1. ENTRYPOINT: `rosita.py`
   2. It's based on a more complex 3D ResNext for input embeddings
   3. The encoder-decoder part is implemented with pre-trained language models (transformers)

## Example Usage

**docker**

```shell
docker run -d \
--name=rosita-dev \
--gpus=all \
--shm-size=50gb \
-v /home/christopher/Documents/data/PHOENIX-2014-T-release-v3:/workspace/data/PHOENIX-2014-T-release-v3/ \
-v /home/christopher/Documents/workspace/signlanguagetranslation/data/bpe:/workspace/data/bpe/ \
-v /home/christopher/Documents/workspace/signlanguagetranslation/output:/workspace/output/ \
-v /home/christopher/Documents/workspace/signlanguagetranslation/model/resnext_101_kinetics.pth:/workspace/model/resnext_101_kinetics.pth \
-v /home/christopher/Documents/workspace/signlanguagetranslation/model/torch_cache:/root/.cache/torch/ \
registry.beuth-hochschule.de/iisy/signlanguagetranslation:rosita-0.0.1 \
python src/rosita.py \
--batch_size=1 \
--epochs=30 \
--learning_rate=0.01 \
--optimizer=sgd \
--num_workers=12 \
--dataset_path=data/PHOENIX-2014-T-release-v3 \
--bpe_path=data/bpe \
--output_dir=output/dev \
```