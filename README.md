## Tensorized_MSM

This repository maintains the code for the paper [Weighted Tensor Completion for Time-Series Causal Inference](https://arxiv.org/abs/1902.04646).

* Terminology
  * We have two kinds of worlds -- thin and fat. They respectively refer to the worlds narrow and wide in the paper.
  * For the experiment, the values of N and T are set as N=500 and T=10 for the thin world, and N=10 and T=500 for fat world, but they can be adjusted in the code.
  * We have two kinds of policies -- I and II. They respectively refer to simple and complex policies in the paper.


- Description of the code
  * Folder [./GEN_DATA](https://github.com/debmandal/Tensorized_MSM/tree/main/GEN_DATA) contains the script to generate the datasets.
    * The main file is gendata.py. It takes as input three variables -- an iteration count (it), type of world, and type of policy. For example, suppose you want to generate 23-rd example for world thin and policy II, then run the following command: 
        ```
        python gendata.py --it=23 --world=thin --policy=II
        ```
    * This generates treatments A, outcomes Y, and covariates XX. In particular the script saves three files Aobs_thin_II_11.npy, Yobs_thin_II_11.npy, and XX_thin_II_11.npy in the folder [./GEN_DATA/data/thin/](https://github.com/debmandal/Tensorized_MSM/tree/main/GEN_DATA/data/thin)
  * Folder [./ESTIMATION](https://github.com/debmandal/Tensorized_MSM/tree/main/ESTIMATION) contains python files to run the tensor completion algorithms. In particular, this folder contains three main files.
    * File tensor_completion.py takes as input three variables -- an iteration count (it), type of world, and type of policy, runs tensor completion for fixed k and different possible values of rank r, and stores the result in the folder [./ESTIMATION/models/thin](https://github.com/debmandal/Tensorized_MSM/tree/main/ESTIMATION/models/thin). For example, if you want to generate the results for 23-rd example, run the following command:
    ```
    python tensor_completion.py --it=23 --world=thin --policy=II
    ```
    The code generates files result_est_atet_thin_II_11_r.npy for r=5 to 15. Each of these files contains the ATET with fixed k and the corresponding value of r.
    * File tensor_completion_hist.py operates same as tensor_completion.py except that it runs tensor completion for a fixed rank r=10 and all possible values of assumed length of history from k=3 to 8.
    * File wts_estimation.py contains several helper functions, including functions to compute the weight matrix.
  * Folder [./ANALYSIS](https://github.com/debmandal/Tensorized_MSM/tree/main/ANALYSIS) contains script files to compute various statistics.
    * File compute_mse.py computes the normalized MSE. 
    * File wta_robins_glm.py runs MSM and computes normalized MSE.
