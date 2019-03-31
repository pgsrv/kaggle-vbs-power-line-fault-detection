
# My Bronze Solution for [Kaggle VSB Power Line Fault Detection](https://www.kaggle.com/c/vsb-power-line-fault-detection/submissions?sortBy=date&group=all&page=1)


This repository is for my solution for the kaggle competition.  
I got bronze with these codes. 

## My best single models

I share two of my single "possibly" silver models.

To my disappointment, I don't select these model as final model, 
so I missed the silver medal because I underestimated the local scores. 
However, these models contributed to my final ensemble ones and brought me the bronze medal. 
These 2 models are inspired by TensorFlow Speech Recognition Challenge I joined in past.

bigru + attention + mix-up + hard voting (private LB 0.668 (same as 38th) / public LB 0.568)
It's similar model with Bruno Aquino's kenel, but 2 main different points.

hard voting of each cv model's classification (threshold=0.5)
Instead of tuning the threshold with the averaged probabilities of those by cv models, 
the classes of the samples are decided by majority of the cv models with the fixed threshold 0.5. I guess this hard voting bring the model robustness.
mix-up
This augmentation method is introduced by the paper mixup: Beyond Empirical Risk Minimization. 
The augmentation generates a new sample from 2 training samples based of the adding.
   \tilde{x} = λx_i +(1−λ)x_j
   \tilde{y} = λy_i +(1−λ)y_j
If \tilde{y} >= 0.5 \, I labeled the new sample as class 1. 
I used 3 as the mix-up ratio, but I don't have enough time to tune the ratio. 
Please see the blog post [the detail](http://www.inference.vc/mixup-data-dependent-data- 
augmentation/) about mix-up. 
The mix-up improved private LB to 0.668 from 0.599 but worsened public LB to 0.568 from 
0.632. Therefore, it requires more careful evaluation to know whether mix-up really works.

In addition, I implement it with PyTorch instead of Keras (I don't know what effect that brings the model performance. ) 


SE-ResNet 50 + window (chunk) 40 step 20 summary feature + hard voting (private LB 0.663 (same as 53th) / public LB 0.587)
This model composed of 2 specific features.

2D-1D SE-Resnet 50
I learned cnn works well for signal classification tasks in the above speech recognition 
competition. That drives me to adopt cnn models for the power line signals. 
Instead of simple cnn models, I used SE-ResNet 50 expecting it to find more deep features. 
(I also tried more simple CNN models with attention, but they are inferior to SE-ResNet in both public/private LB. )

To support signal inputs, I changed 3 points from original 2D SE-ResNet for image.

input concatenated features from 3 phase in height axis.
2D 1st conv with kernel size = (n feature(height), your kernel size for time step axis) . It weaves the features into 1D time steps and the channels.
following 1D conv blocks
These steps are inspired by Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization by Schirrmeister etc. and DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing by Yao etc. arXiv:1611.01942v2.

smaller window size to extract summarized feature
In the public kernels, the features are extracted with window size 5000. 
However, I want to leave feature extracting to cnn as much as possible . 
Therefore the inputted features are extracted with more smaller window (40) and step (20) size. 
I also tried to train the model with the raw 3 phase signals or the features from smaller window 
sizes, but I cannot do due to oom.
The SE-ResNet also seems over-fitting to privte LB and it leaves improvements for the model parameters.

In conclusion, the both of them seem to include some hints but requires more improvements 
to obtain the high scores in both private and public LB.
I'll share my codes in near feature.  

## my final submitted ensemble models

- hard voting of 10 models


## Summary for What I Tried to reduce high variance of models (also want to know yours)

In this competition, I was suffering from the high variance of models (especially RNN/CNN).  
To reduce the variances, I tried some approaches.  

### worked

- hard voting of k-fold models
- small window size to extract feature
- ensemble of locally good and public LB
- original random seed different from public kernels  
    (Of course, I didn't tune the random seed)
    This helps me a lot to judge whether the kernel models really work or get a better public LB score by luck.

### worked partially 

- soft voting / naive bayes / svm / logistic regression of cv models  
    (inferior to hard voting)
- mix-up 
    (worked for private LB, not worked)
- bagging + under-sampling of class 0  
    worked, generate the most robust model, but lower LB scores than those wo bagging
- [SMOTH (worked for private LB, but worsen public LB)](https://www.kaggle.com/yatzhash/smote-to-learn-from-a-few-anomaly-sample/edit)

### not worked at all
- fine-tuning pre-trained model with ImageNet.  
  The signals are convert into images by plotting them. The models are inferior to the non-image models.  
  This plotting approach is introduced in [this eeg paper](https://www.sciencedirect.com/science/article/pii/S2213158219300348)
  I can't find any pre-trained models by signal datasets.   
- cosine loss  
  not worked at all, training didn't progress. It is possible my implementation of the cosine loss isn't correct.  
  about cosine loss, please refer to [this paper](https://arxiv.org/abs/1901.09054)
- multiple dropouts  
  introduced in [Deep Neural Networks for High Dimension, Low Sample Size Data](https://www.ijcai.org/proceedings/2017/0318.pdf)  
  It caused over-fitting to local data and public LB
- threshold tuning  
    caused over-fitting
- data augmentation by splitting a signal into 2 parts and swapping them  
    not worked at all, causing over-fitting to augmented samples

## each model scores

| model                                                                                                    | private LB | public LB | LB diff  | avg_valid_score | std_valid_score | avg_train_score | std_train_score | avg_train – valid_score |
|----------------------------------------------------------------------------------------------------------|------------|-----------|----------|-----------------|-----------------|-----------------|-----------------|-------------------------|
| senet/resnet_50/summary_not_scaled/summary_window_40_step_20                                             | 0.6637     | 0.58799   | -0.07571 | 0.7704479123    | 0.040493426     | 0.8425798233    | 0.0406163756    | 0.072131911             |
| bigru_attention/fft/fft_5000_pca_200_scaled_mixup_3                                                      | 0.64809    | 0.43756   | -0.21053 | 0.7005664499    | 0.0954152228    | 0.9966534313    | 0.0031825431    | 0.2960869814            |
| Vote 5 by 10 models (submitted)                                                                          | 0.64573    | 0.67097   | 0.02524  |                 |                 |                 |                 | 0                       |
| Vote 2 by 10 models (submitted)                                                                          | 0.63521    | 0.67728   | 0.04207  |                 |                 |                 |                 | 0                       |
| Lgb smoth with summarized features  / window 5000 (my kernel)                                            | 0.60413    | 0.46814   | -0.13599 |                 |                 |                 |                 | 0                       |
| Base Neural Network  (my kernel / fork of Tarun Sriranga Paparaju's)                                     | 0.60128    | 0.67756   | 0.07628  |                 |                 |                 |                 | 0                       |
| bigru_attention/summary_not_scaled/window_5000_summary_adabound                                          | 0.60107    | 0.59639   | -0.00468 | 0.7905776706    | 0.0472729464    | 0.8188164191    | 0.0522787047    | 0.0282387485            |
| bigru_attention/summary_not_scaled/window_5000_summary                                                   | 0.59926    | 0.63203   | 0.03277  | 0.737201588     | 0.0657890232    | 0.7278606397    | 0.0329050406    | -0.0093409483           |
| Lgb baseline with summarized features  / window 5000(my kernel)                                          | 0.59825    | 0.57432   | -0.02393 |                 |                 |                 |                 | 0                       |
| cnn/attention/summary_window_40_step_20                                                                  | 0.592      | 0.5586    | -0.0334  | 0.7968742123    | 0.0478793898    | 0.8848245502    | 0.0370868434    | 0.087950338             |
| lstm,keras,kearnel,window1000, prercentile summarized (my kernel / change window size of runo Aquino's ) | 0.58723    | 0.58417   | -0.00306 |                 |                 |                 |                 | 0                       |
| Transformer multiple dropouts (my kernel / add multiple dropout to Khoi Nguyen 2's )                     | 0.58218    | 0.63815   | 0.05597  |                 |                 |                 |                 | 0                       |
| bilstm_attention/window_5000_summary                                                                     | 0.576      | 0.60917   | 0.03317  | 0               | 0               | 0               | 0               | 0                       |
| cnn/attention_multiple_dropout_glu/summary_window_40_step_20                                             | 0.57271    | 0.51742   | -0.05529 | 0.7732131238    | 0.0472060149    | 0.9751040672    | 0.0093121909    | 0.2018909434            |
| bigru_attention/fft/fft_5000_pca_200_scaled                                                              | 0.57136    | 0.22358   | -0.34778 | 0.6733431604    | 0.0861268474    | 0.9983384712    | 0.0022334284    | 0.3249953108            |
| bigru attention\nwindow 5000\nsummsary\nbagging\nhard-hard vote                                          | 0.56166    | 0.56729   | 0.00563  |                 |                 |                 |                 | 0                       |
| hierarchical_gru_context/summary_not_scaled/window_200_step_200                                          | 0.55697    | 0.50773   | -0.04924 | 0.7640551614    | 0.0547853114    | 0.8922545967    | 0.0486670168    | 0.1281994354            |
| hierarchical_gru/summary/window_200_step_200                                                             | 0.54983    | 0.51372   | -0.03611 | 0.7653735125    | 0.0515660403    | 0.8477982792    | 0.0367964827    | 0.0824247667            |
| Lgb with plot feature from imagenet pretrained resnet18\nwindow800000                                    | 0.52132    | 0.31006   | -0.21126 |                 |                 |                 |                 | 0                       |
| plot_image/xception pretrained,\n10fold 10 baggng hard hard                                              | 0.42124    | 0.43992   | 0.01868  |                 |                 |                 |                 | 0                       |
| bigru_attention/fft/fft_5000_pca_200                                                                     | 0.08013    | 0.03401   | -0.04612 | 0.7396792297    | 0.0532865745    | 0.8840468977    | 0.0317707373    | 0.144367668             |
| hierarchical_gru/summary/window_400_step_200_not_scaled                                                  |            |           | 0        | 0.7795642853    | 0.0430654655    | 0.8619797012    | 0.0318342473    | 0.082415416             |
| bigru_attention/summary_not_scaled/window_5000_summary_adabound_fixed_attention                          |            |           | 0        | 0.7733391786    | 0.0547682797    | 0.8829171245    | 0.083526586     | 0.1095779459            |
| senet/resnet_50/summary_not_scaled/aug_summary_window_40_step_20_oversampling                            |            |           | 0        | 0.7681722669    | 0.0440787202    | 0.9488968856    | 0.0356257465    | 0.1807246187            |
| senet/resnet_50/summary_not_scaled/aug_summary_window_40_step_20                                         |            |           | 0        | 0.765468338     | 0.0158233148    | 0.7790296125    | 0.3148757294    | 0.0135612744            |
| cnn/attention/layer_3_dilation_summary_window_40_step_20                                                 |            |           | 0        | 0.7649039755    | 0.0403176339    | 0.8689986741    | 0.0230227488    | 0.1040946987            |
| bigru_attention/mixup/window_5000_summary_mixup_3                                                        |            |           | 0        | 0.750878024     | 0.0629679006    | 0.6147919619    | 0.033218479     | -0.1360860621           |
| cnn/attention/multiple_dropouts/layer_8_summary_window_40_step_20                                        |            |           | 0        | 0.7500082943    | 0.0338723182    | 0.9894916558    | 0.0099192088    | 0.2394833615            |
| plot_image/resnet18_pretrained_kfold/window_800000_stride_800000                                         |            |           | 0        | 0.7496013443    | 0.1035896812    | 1               | 0               | 0.2503986557            |
| hierarchical_gru_context/summary_not_scaled/window_200_step_200_10fld                                    |            |           | 0        | 0.7259868204    | 0.0472379816    | 0.9984955643    | 0.0030088714    | 0.2725087439            |
| bigru_attention/summary_not_scaled/hidden_64_window_5000_summary                                         |            |           | 0        | 0.7236540053    | 0.0789522569    | 0.7087902626    | 0.0345793849    | -0.0148637426           |
| senet/resnet_50/summary_not_scaled/aug_summary_window_40_step_20_ratio_4                                 |            |           | 0        | 0.7220189096    | 0.0321808091    | 0.9501078494    | 0.0491061887    | 0.2280889398            |
| hierarchical_gru/summary/window_400_step_200                                                             |            |           | 0        | 0.7121102149    | 0.0531124581    | 0.8790451773    | 0.1050255724    | 0.1669349624            |
| bigru_attention_1layer/fft/fft_5000_pca_200_scaled                                                       |            |           | 0        | 0.683906986     | 0.0752258091    | 0.9961403753    | 0.0083853813    | 0.3122333893            |
| senet/resnet_50/summary_window_40_step_20                                                                |            |           | 0        | 0.6585946333    | 0.0325341711    | 1               | 0               | 0.3414053667            |
| senet/resnet_10/summary_not_scaled/aug_summary_window_40_step_20_cancelled                               |            |           | 0        | 0.5879330898    | 0.0390600348    | 0.9899028935    | 0.0142794649    | 0.4019698037            |
| plot_image/resnet18_pretrained_kfold/window_10000_stride_5000                                            |            |           | 0        | 0.5738912546    | 0.0267053787    | 0.7727539571    | 0.1775875701    | 0.1988627025            |
| plot_image/resnet10_kfold/window_800000_stride_800000                                                    |            |           | 0        | 0.570559577     | 0.2950392701    | 0.8784202678    | 0.2416770508    | 0.3078606908            |
| gru/fft/fft_5000_pca_200_scaled                                                                          |            |           | 0        | 0.5175065547    | 0.1041856576    | 0.7466612351    | 0.1334295432    | 0.2291546805            |
| senet/resnet_50/avg_window_8_step_4                                                                      |            |           | 0        | 0.4717577145    | 0               | 0.5040333382    | 0               | 0.0322756237            |
| bigru_attention/aug/percentile_summary_window_5000_step_5000                                             |            |           | 0        | 0.4162296948    | 0.3400685033    | 0.4626842009    | 0.3508078985    | 0.0464545061            |
| bigru_attention/mae/window_5000_summary (not bce)                                                        |            |           | 0        | 0               | 0               | 0.0272139915    | 0.0272139915    | 0.0272139915            |
| bilstm_attention/window_5000_summary_adabound                                                            |            |           | 0        | 0               | 0               | 0               | 0               | 0                       |
| bilstm_attention/window_5000_summary_adabound_original_param                                             |            |           | 0        | 0               | 0               | 0.0287608089    | 0.0498151823    | 0.0287608089            |
| hierarchical_gru/cosine_loss/summary/window_100_step_200 (not bce)                                       |            |           | 0        | 0               | 0               | 0.0188631135    | 0.0215095338    | 0.0188631135            |