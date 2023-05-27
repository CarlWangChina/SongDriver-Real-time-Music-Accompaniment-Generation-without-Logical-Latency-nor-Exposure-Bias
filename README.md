
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
    
