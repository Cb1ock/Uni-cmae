Installation
---
Requirements

- python=3.12
- Pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- simplejson `pip install simplejson`
- fvcore `pip install 'git+https://github.com/facebookresearch/fvcore'`
- GCC `conda install -c conda-forge gcc`
- iopath `pip install -U iopath`
- OpenCV `pip install opencv-python`
- sklearn `pip install scikit-learn`
- timm `pip install timm=0.3.2`, which need [fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842)