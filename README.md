## Install Environment
```bash
conda create --name llm python=3.10
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers==4.44.2
pip install tf-keras
pip install geomloss
pip install scikit-learn
```

## Download Datasets

[MedQA-USMLE](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle)
[GSM8k](https://huggingface.co/datasets/openai/gsm8k)
[OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)

You can either use the following script to preprocess the data or download the preprocessed three dataset via this [link](https://drive.google.com/file/d/1UriYueL9bdxiwE_PbeFV0Yyy7xPSpiRe/view?usp=sharing).

Modify `dataset`, `data_path`, `data_format`, `split` in `preprocess/formatting.sh`, and then run:
```angular2html
cd preprocess
bash formatting.sh
```

## Fine-tuning
Modify `dataset`, `data_path`, `data_format`, `split`, `model` in `scripts/train_w_sft.sh` and `scripts/train_w_ccsft.sh`, and then run:
```angular2html
CUDA_VISIBLE_DEVICES=0 bash scripts/train_w_sft.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_w_ccsft.sh
```

## Inference
Modify `dataset`, `data_path`, `model`, `identity` and set `data_format=sharegpt` in `scripts/infer_twice.sh`, and then run:
```angular2html
bash scripts/infer.sh
```

## Evaluation
Run
```angular2html
bash postprocess/analyzing.sh
```


