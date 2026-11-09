# MediTwin AI — Model Credits & Licenses

Not for clinical use.

This document records the third-party pretrained models used within MediTwin AI, acknowledging their original authors, licenses, source URLs, and acquisition dates.

---

## 1. Medical Image Agent — Pediatric Model

- **Model Name**: ResNet18 Binary Chest X-Ray Classifier (`checkpoints/binary/resnet18.pt`)
- **Authors / Developers**: Luana Carolina Reis and Jakub Błaszczyk (University of Aveiro, Portugal; Lodz University of Technology, Poland)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Repository URL**: [https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier](https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier)
- **Base Architecture**: `torchvision.models.resnet18` (ImageNet-1k pretrained, fine-tuned on pediatric chest X-rays)
- **Training Data**: Kermany et al. pediatric chest X-ray dataset (Guangzhou Women and Children's Medical Center)
- **Download Date**: 2026-11-01
- **Original File**: `models/medical_image/resnet18_xray.pt` (129 MB, contains model weights, optimizer state, and training metrics)
- **Original File SHA-256**: `c1f4b6de43f5c6f4844b4e7d5215c8e78a468ff1f75edccc3b26d859a8f40767`
- **Slim Weights File**: `models/medical_image/resnet18_xray_weights.pt` (42.71 MB, extracted `model_state_dict` for fast CPU loading)
- **Age Routing**: Used when `age_years < 18`
- **Clinical Disclaimer**: Not for clinical use. Research prototype for educational and screening assistance evaluation only.

---

## 1b. Medical Image Agent — Adult Model

- **Model Name**: `densenet121-res224-all` (TorchXRayVision DenseNet121)
- **Library**: [TorchXRayVision](https://github.com/mlmed/torchxrayvision) v1.5.7
- **License**: Apache-2.0 *(from model card and Hugging Face hub entry)*
- **Papers**:
  - Cohen JP et al. (2021). *TorchXRayVision: A library of chest X-ray datasets and models.* arXiv:2111.00595
  - Cohen JP et al. (2020). arXiv:2002.02497
- **Training Data**: NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, Google Chestx-ray, OpenI, RSNA Pneumonia Detection
- **Weights Location**: `~/.torchxrayvision/models_data/` (downloaded by library on first use; 28 MB file)
- **Download Date**: Downloaded automatically by TorchXRayVision at first run; verified present 2026-11-09
- **Output**: 18 pathology labels; op-normalized probabilities (sigmoid + op_norm so threshold → 0.5)
- **Pneumonia op-threshold**: 0.056810737 (source: `model_urls['all']['op_threshs'][8]` in `torchxrayvision/models.py`)
- **Age Routing**: Used when `age_years >= 18`
- **Independent Accuracy Measurement**: NOT performed in this project — training data overlaps every large public adult CXR dataset
- **Peak RSS (both models in memory)**: ~1,268 MB (measured 2026-11-09 with `/usr/bin/time -v`)
- **Clinical Disclaimer**: Screening aid only. Not validated for clinical use. A licensed radiologist must interpret all results.

---

## 2. NLP Agent

- **Model Name**: Biomedical NER All (`d4data/biomedical-ner-all`)
- **Architecture**: DistilBertForTokenClassification (6-layer, 768-dim hidden, 84 NER labels)
- **Authors / Developers**: d4data team
- **License**: Apache-2.0
- **Repository URL**: [https://huggingface.co/d4data/biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all)
- **Local Path**: `models/nlp/d4data_biomedical_ner_all/`
- **Model File**: `model.safetensors` (254 MB; preferred over `pytorch_model.bin`; loaded with `use_safetensors=True, local_files_only=True`)
- **Training Data**: MACCROBAT biomedical case reports dataset (2018 + 2020 annotations). **Note**: MediTwin AI does NOT evaluate on MACCROBAT — that would be data leakage since the model was trained on it.
- **Download Date**: 2026-11-01
- **Label Count**: 84 (O + 41 entity types × B/I prefixes). MediTwin maps 5 types to 4 groups (DISEASE, SYMPTOM, MEDICATION, PROCEDURE); 38 types map to OTHER.
- **Measured Latency**: avg ~211 ms/sentence (first call ~1,500 ms warmup); CPU only.
- **Measured Memory**: +162 MB RSS when loaded after ResNet18; ~1,081 MB total peak during dual-model inference.
- **Clinical Disclaimer**: Not for clinical use. Research prototype for educational purposes only.
