# MediTwin AI — Model Credits & Licenses

Not for clinical use.

This document records the third-party pretrained models used within MediTwin AI, acknowledging their original authors, licenses, source URLs, and acquisition dates.

---

## 1. Medical Image Agent

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
- **Clinical Disclaimer**: Not for clinical use. Research prototype for educational and screening assistance evaluation only.

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

