
# [SongDriver: Real-time Music Accompaniment Generation without Logical Latency nor Exposure Bias](https://dl.acm.org/doi/10.1145/3503161.3548368)

> **News**: SongDriver is now fully open-source!  
> Code, pre-trained weights and training data are all released.  
> - **One-stop download** (weights + code + data): [🤗 Hugging Face repo](https://huggingface.co/karl-wang/SongDriver)  
> - **GitHub** (code + data only, due to file-size limits): [📁 GitHub repo](https://github.com/CarlWangChina/SongDriver-Real-time-Music-Accompaniment-Generation-without-Logical-Latency-nor-Exposure-Bias)

SongDriver uses a parallel mechanism of prediction and arrangement phases to achieve **zero logical latency** in real-time accompaniment generation, while significantly reducing exposure bias.

# SongDriver：零逻辑延迟、零曝光偏差的实时音乐伴奏生成系统

> **最新信息**：SongDriver 现已完全开源！  
> 代码、预训练权重、训练数据均已发布。  
> - **完整下载**（权重+代码+数据）：[🤗 Hugging Face 仓库](https://huggingface.co/karl-wang/SongDriver)  
> - **GitHub**（仅代码+数据，因文件大小限制）：[📁 GitHub 仓库](https://github.com/CarlWangChina/SongDriver-Real-time-Music-Accompaniment-Generation-without-Logical-Latency-nor-Exposure-Bias)

SongDriver 通过「编排阶段」与「预测阶段」并行，实现**零逻辑延迟**的实时伴奏生成，并显著降低曝光偏差。

## 目录结构
## File Tree
Files description:

    ----data_process
        ---- dataset_select.py        # Select different dataset to process 
        ---- dataset_subdivide.py     # Divide dataset to a correct form for input
        ---- testset_subdivision.py   # Split the dataset into test set and train set
        ---- structural_chord         # Detail codes of extracting structrural chord feature
        ---- weighted_feature         # Detail codes of extracting weighted factor feature
        ---- chordPreprocess.py       # To determine two features: structural_chord & weighted_features()
        ---- data_preprocessing.py    # Preprocess the input data
        ---- phrase_level_segmentation.py   # Read MIDI file and separate the music into single input
    
    ----model
        ---- config.py                # Configuration of the model
        ---- core.py                  # Core codes for training
        ---- Inference.py             # Codes for inference
        ---- SDEmbedding.py           # Codes for the embedding of music
        ---- model.py                 # Model structure
        ---- core_sdri.py             # Ablation experiments for terminal chord  
        ---- core_sdrs.py             # Ablation experiments for structural chord
        ---- core_sdwf.py             # Ablation experiments for weighted factor
        ---- core_sdwn.py             # Ablation experiments for weighted note 
        ---- decode_results.py        # The result after decoding in the model
        
    ----test
        ---- test.py                  # Test the result of the model
        ---- test1.py                 # Test the result of the model
        ---- model_tesst.py           # Test the result of the model
    
## Citation
If you use this work in your research, please cite our paper:
## 引用
如果本工作对您有帮助，请引用：
```
@inproceedings{10.1145/3503161.3548368,
author = {Wang, Zihao and Zhang, Kejun and Wang, Yuxing and Zhang, Chen and Liang, Qihao and Yu, Pengfei and Feng, Yongsheng and Liu, Wenbo and Wang, Yikai and Bao, Yuntao and Yang, Yiheng},
title = {SongDriver: Real-time Music Accompaniment Generation without Logical Latency nor Exposure Bias},
year = {2022},
isbn = {9781450392037},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3503161.3548368},
doi = {10.1145/3503161.3548368},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {1057–1067},
numpages = {11},
keywords = {automatic improvisation, music accompaniment generation},
location = {Lisboa, Portugal},
series = {MM '22}
}
