# STAC: Saliency-Guided Transformer Attention with Pixel-Level Contrastive Learning for Weakly Supervised Defect Localization
This work proposes **STAC**, a novel framework for **weakly supervised defect localization** that leverages saliency-guided transformer attention and pixel-level contrastive learning to achieve precise defect maps using only image-level labels. 

🚀 **Code and pretrained models will be released soon!** Stay tuned.

## 📄 Abstract
[Weakly supervised learning has emerged as a powerful paradigm for image segmentation, providing a practical solution to reduce the dependency on costly, pixel-level annotated datasets. By leveraging partial annotations, such as image-level tags, this approach significantly reduces the labeling burden while maintaining competitive performance. However, weakly supervised methods inherently face challenges, including overlapping activation regions and insufficient localization due to sparse and noisy signals from image-level tags, often leading to suboptimal segmentation performance. These issues are further exacerbated in industrial defect localization, where datasets present unique complexities, including low-contrast object boundaries, inconsistent shapes, inter-class similarities, and substantial intra-class variations. To address these limitations, we introduce a novel Saliency-guided Transformer Attention with Contrastive Learning (STAC) framework. The proposed framework leverages Transformer attention to generate localization maps and enhances the learning of challenging foreground and background information through saliency-guided cues. Additionally, a pixel-level contrastive learning module is employed to refine feature map representations by bringing positive pairs closer together and pushing negative pairs apart, effectively addressing challenges such as overlapping activations and ambiguous boundaries. Through extensive experiments and ablation studies, the proposed method demonstrates superior performance compared to state-of-the-art approaches across three defect segmentation datasets. We also evaluated the generalization of our approach on the PASCAL VOC segmentation and MVTec anomaly detection datasets.]

## 🔥 Highlights
- **Saliency-Guided Transformer Attention**: Focuses on relevant regions early for better feature representation
- **Pixel-Level Contrastive Learning**: Enhances discriminative power without dense annotations
- **Weakly Supervised**: Only image-level labels required — huge savings on labeling costs
- Efficient training with minimal annotation effort
- Outperforms existing weakly supervised methods on standard datasets (e.g., NEU-Seg, DAGM, MTD, MVTec.)

## 📊 Results
(Coming soon: Tables with mIoU on benchmark datasets)

## 🚀 Quick Start
(Will be updated with installation, training, and inference scripts )
