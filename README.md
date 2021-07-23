# Neural Sign Language Translation Pytorch

This repository contains the implementation from the Neural Sign Language Translation [Paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html) in Pytorch. 

The code is based on the official [implementation](https://github.com/neccam/nslt) from GitHub User [neccam](https://github.com/neccam).

## Table of Contents

- [Neural Sign Language Translation Pytorch](#neural-sign-language-translation-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Credits](#credits)
  - [License](#license)

## Installation

## Usage

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

## Contributing

## Credits

## License

Copyright 2019 Christopher Kümmel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.