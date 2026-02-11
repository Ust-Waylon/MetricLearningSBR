# MetricLearningSBR
An official implementation of **A Metric Learning Framework for Session-based
Recommendation**.

## Requirements
For all models except LESSR:
- Python 3.10
- PyTorch 2.4.1
- NumPy 1.26.3
- Pandas 2.2.3
- SciPy 1.14.1
- scikit-learn 1.6.1

For LESSR:
- Python 3.7
- PyTorch 1.10.2
- DGL 0.6.1
- NumPy 1.21.6
- Pandas 1.1.3
- SciPy 1.7.3

Detailed requirements are in requirements.txt.

## Usage
1. Create conda environments and install the requirements.  
    - For SR-GNN, RAIN, SASRec, DIDN and A-Mixer, run:
        ```bash
        conda create -n mlsbr python=3.10
        conda activate mlsbr
        pip install -r requirements.txt
        ```
        Use the mlsbr environment for these models.
    - For LESSR, run:
        ```bash
        conda create -n lessr python=3.7
        conda activate lessr
        conda install cudatoolkit=11.8
        pip install -r requirements.txt
        ```
        Use the lessr environment for LESSR.

2. Prepare the datasets.
    - For SR-GNN, LESSR, SASRec and DIDN, the preprocessed datasets are already prepared in the 'datasets' folder.
    - For RAIN and A-Mixer, the datasets need to be further processed. Take RAIN as an example, run:
        ```bash
        cd RAIN
        python preprocess.py
        ```
        This will process the all the datasets and save them in the 'RAIN/datasets' folder.
    The procedure for the A-Mixer is similar.

3. Train the models with both the original and the metric learning frameworks.

    - To train the models with the original frameworks (we called it as the *embeddings-dot-product* framework in our paper), just run the main.py file in the corresponding folder. The default hyperparameters are already set for each model.

        For example, to train the original SASRec model on the Tmall dataset, run:
        ```bash
        cd SASRec
        python main.py --dataset-dir ../datasets/tmall/
        ```

        Similarly, to train the original A-Mixer model on the Tmall dataset, run:
        ```bash
        cd A-Mixer/src_pytorch
        python main_area_semantic.py --dataset tmall
        ```

    - To train the models with the *metric learning* frameworks, you just need to set the 'logit_type' to 'euclidean' in the main.py file. The optimal 'scale' factor (based on our experiments) will be set accordingly.
        
        For example, to train the metric learning SASRec model on the Tmall dataset, run:
        ```bash
        cd SASRec
        python main.py --dataset-dir ../datasets/tmall/ --logit_type euclidean
        ```

        Similarly, to train the metric learning A-Mixer model on the Tmall dataset, run:
        ```bash
        cd A-Mixer/src_pytorch
        python main_area_semantic.py --dataset tmall --logit_type euclidean
        ```

## Acknowledgments
Our code for the baseline models is adapted from the corresponding official repositories as follows. We thank the authors for their contributions to the community.

- SR-GNN: https://github.com/CRIPAC-DIG/SR-GNN  
    - we also refer to a more recent (unofficial) implementation of SR-GNN: https://github.com/DiMarzioBian/SR-GNN
- LESSR: https://github.com/twchen/lessr
- RAIN: https://github.com/zengxy20/RAIN
- SASRec: https://github.com/kang205/SASRec
- DIDN: https://github.com/Zhang-xiaokun/DIDN
- A-Mixer: https://github.com/Peiyance/Atten-Mixer-torch