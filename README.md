
# [SongDriver: Real-time Music Accompaniment Generation without Logical Latency nor Exposure Bias](https://dl.acm.org/doi/10.1145/3503161.3548368)
SongDriver uses a parallel mechanism of prediction and arrangement phases to achieve zero logical latency in real-time accompaniment generation, significantly reducing exposure bias.

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
abstract = {Real-time music accompaniment generation has a wide range of applications in the music industry, such as music education and live performances. However, automatic real-time music accompaniment generation is still understudied and often faces a trade-off between logical latency and exposure bias. In this paper, we propose SongDriver, a real-time music accompaniment generation system without logical latency nor exposure bias. Specifically, SongDriver divides one accompaniment generation task into two phases: 1) The arrangement phase, where a Transformer model first arranges chords for input melodies in real-time, and caches the chords for the next phase instead of playing them out. 2) The prediction phase, where a CRF model generates playable multi-track accompaniments for the coming melodies based on previously cached chords. With this two-phase strategy, SongDriver directly generates the accompaniment for the upcoming melody, achieving zero logical latency. Furthermore, when predicting chords for a timestep, SongDriver refers to the cached chords from the first phase rather than its previous predictions, which avoids the exposure bias problem. Since the input length is often constrained under real-time conditions, another potential problem is the loss of long-term sequential information. To make up for this disadvantage, we extract four musical features from a long-term music piece before the current time step as global information. In the experiment, we train SongDriver on some open-source datasets and an original \`{a}iMusic Dataset built from Chinese-style modern pop music sheets. The results show that SongDriver outperforms existing SOTA (state-of-the-art) models on both objective and subjective metrics, meanwhile significantly reducing the physical latency.},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {1057–1067},
numpages = {11},
keywords = {automatic improvisation, music accompaniment generation},
location = {Lisboa, Portugal},
series = {MM '22}
}
