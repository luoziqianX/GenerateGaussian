# Generate Gaussian

## Install

```bash
conda create -n generategaussian python=3.10
conda activate generategaussian
pip install -r require1.txt
pip install -r require2.txt
pip install ./diff-gaussian-rasterization_LGM
pip install git+https://github.com/NVlabs/nvdiffrast
pip install transformers==4.40.1
```

## Pretrained Weights

### For sv3d_p model

Download sv3d_p.safetensors from https://huggingface.co/stabilityai/sv3d to checkpoints/sv3d_p.safetensors
### For LGM model

You can download the pretrained weights for LGM model from [huggingface](https://huggingface.co/ashawkey/LGM)
For example,to download the fp16 LGM model for inference:

```bash
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```

For [MVDream](https://github.com/bytedance/MVDream)
and [ImageDream](https://github.com/bytedance/ImageDream),
we use a [diffuser implementation](https://github.com/ashawkey/mvdream_diffusers).
The weight will be downloaded automatically.

## Inference
You can test the baseline by running the following command
```Bash
python test.py --input_path IMAGE_PATH
```
**Command Line Arguments for test.py**
* --input_path the image path
* --num_frames the frame for sv3d_p model
* --model_config the path of model config
* --num_step the step for sv3d diffusion model
* --device the test device
* --elevation the generate elevation
* --output_path the path for model's output
* --decoding_t the num of images decoding one time
* --seed the seed for torch
* --lgm_checkpoints_path the path of lgm_pretrained model

