# UdaciSense: Model Optimization Technical Report

## Executive Summary

The goal of this project was to optimize a vision model for mobile deployment, reducing the baseline model's size by 70% and its inference time for 60% while staying within 5% of the baseline model's accuracy. To do this, I implemented a compression pipeline of quantization-aware training (QAT) followed by graph optimizations, resulting in the following:

* A 71.5% reduction in model size (5.96MB -> 1.70MB)
* A 3.8x speedup in inference time (161.05ms -> 42.03ms)
* Only a 4.6% loss in accuracy (89.90% -> 85.80%)

This allows UdaciHome customers to effectively use the UdaciSense model on their mobile devices, expanding our business reach and market penetration.

## 1. Baseline Model Analysis

### 1.1 Model Architecture

The original model was MobileNetV3, which contains a series of convolutional layers important for object recognition, but also successive Conv2d + BatchNorm + ReLU layers which are amenable to operator fusion (i.e., graph optimization).

### 1.2 Performance Metrics


| Metric                     | Value  |
| -------------------------- | ------ |
| Model Size (MB)            | 5.96   |
| Inference Time - CPU (ms)  | 161.05 |
| Accuracy (%)               | 89.90  |
| \[Other relevant metrics\] |        |

### 1.3 Optimization Challenges

The convolutional layers cannot be excessively pruned, or the model will lose an unacceptable amount of accuracy.

## 2. Compression Techniques

### 2.1 Overview

#### Technique 1: Quantization-Aware Training (QAT)

##### Implementation Approach

Implemented quantization-aware training (QAT) starting from epoch 0 because the baseline model's weights had already been trained.

##### Results


| Metric                    | Baseline | After Technique 1 | Change (%) |
| ------------------------- | -------- | ----------------- | ---------- |
| Model Size (MB)           | 5.96     | 1.76              | 70.5%      |
| Inference Time - CPU (ms) | 161.05   | 93.02             | 1.8x       |
| Accuracy (%)              | 89.90%   | 86.80%            | -3.4%      |
| [Other relevant metrics]  |          |                   |            |

##### Analysis

QAT dramatically decreased model size and inference speed while maintaining high accuracy.

#### Technique 2: Graph Optimizations

##### Implementation Approach

Implemented TorchScript graph optimizations to improve inference speed with zero accuracy loss.

##### Results


| Metric                    | Baseline | After Technique 2 | Change (%) |
| ------------------------- | -------- | ----------------- | ---------- |
| Model Size (MB)           | 5.96     | 5.94              | 2.0%       |
| Inference Time - CPU (ms) | 161.05   | 131.99            | 1.4x       |
| Accuracy (%)              | 89.90%   | 89.90%            | 0.0%       |
| [Other relevant metrics]  |          |                   |            |

##### Analysis

Graph optimizations provide literally free improvements to model size and inference speed.

### 2.2 Comparative Analysis

QAT dramatically improved model size and inference speed while maintaining high accuracy, while graph optimizations improved model size and inference speed even further with zero impact to accuracy.

## 3. Multi-Stage Compression Pipeline

### 3.1 Pipeline Design

I first applied QAT to shrink the model's size and inference speed, and then graph optimizations to shrink them even further.

### 3.2 Implementation

As above.

### 3.3 Results


| Metric                   | Baseline | Final Optimized Model | Change (%) | Requirement Met? |
| ------------------------ | -------- | --------------------- | ---------- | ---------------- |
| Model Size (MB)          | 5.96     | 1.70                  | 71.5       | [30% reduction]  |
| Inference Time CPU (ms)  | 161.05   | 42.03                 | 3.8x       | [40% reduction]  |
| Accuracy (%)             | 89.90%   | 85.80                 | -4.6%      | [Within 5%]      |
| [Other relevant metrics] |          |                       |            | -                |

### 3.4 Analysis

The compression pipeline shrunk the model size by nearly 75% and increased inference speed by almost 4x while maintaining accuracy within 5% of baseline. QAT performed the bulk of the model size reduction and also led to an increase in inference speed, while graph optimizations further reduced model size and inference time. QAT came with a trade-off of a hit to model accuracy, which was not seen with graph optimizations.

## 4. Mobile Deployment

### 4.1 Export Process

I used `torch.utils.mobile_optimizer`'s `optimize_for_mobile` method, which performs graph optimizations and other fine-tuning for ARM processors on mobile devices.

### 4.2 Mobile-Specific Considerations

Mobile environments use ARM processors, whereas I am testing on x86. Getting an accurate read of the model's mobile performance would require testing on an actual mobile device.

### 4.3 Performance Verification

As stated above, accurate performance verification would require testing on mobile rather than on x86.

## 5. Conclusion and Recommendations

### 6.1 Summary of Achievements

My compression pipeline successfully shrunk the model to fit on mobile, while also increasing inference speed and maintaining high accuracy. This will allow UdaciHome customers to employ UdaciSense's object recognition functionality on their mobile phones, increasing UdaciHome's business reach and market penetration.

### 6.2 Key Insights

Quantization-aware training (QAT) dramatically shrinks model size while maintaining high accuracy. Graph optimizations offer literally *free* model size and inference speed improvements. Pruning can zero weights, but obtaining any model size or inference speed benefits from this requires specialized hardware and/or libraries.

### 6.3 Recommendations for Future Work

As discussed in notebook 4, one possible future direction could be employing continuous training to account for model drift and ensuring that the model remains performant even as input distributions shift.

### 6.4 Business Impact

As stated above, alllowing UdaciHome customers to employ UdaciSense's object recognition functionality on their mobile phones will increase UdaciHome's business reach and market penetration.
