# The Official Repository of MISELBO - Multiple Importance Sampling ELBO and Deep Ensembles of Variational Approximations

<div align="center">
  <a>Oskar Kviman</a> &emsp; <b>&middot;</b> &emsp;
  <a>Harald Melin</a> &emsp; <b>&middot;</b> &emsp;
  <a>Hazal Koptagel</a> &emsp; <b>&middot;</b> &emsp;
  <a>Víctor Elvira</a> &emsp; <b>&middot;</b> &emsp;
  <a>Jens Lagergren</a> &emsp; &emsp;
</div>
<br>
<br>

[MISELBO](https://arxiv.org/) (actual arxiv link will soon be added) .

<p align="center">
    <img src="img/miselbo-framework.png" width="800">
</p>

## MISELBO for NVAE
To train an ensemble of variational approximations using the [NVAE](https://arxiv.org/abs/2007.03898)
model:
1. Clone the NVAE [repository](https://github.com/NVlabs/NVAE).
2. Clone this repository. 
3. Replace/add the files "train.py", "model.py" and "miselbo_eval.py" from this repository to the NVAE repo.
4. Follow the instructions for MNIST in the Readme of NVAE to train p<sub>&theta;</sub> and q<sub>&Phi;<sub>1</sub></sub>.

To train addtional q<sub>&Phi;<sub>i</sub></sub>:
``` 
train.py --train_new_q True --seed "seed != seed of q_0" "same arguments as q_0" 
```
The train_new_q argument ensures that the decoder is not trained.

## Reproducing MISELBO-NVAE results using already trained p<sub>&theta;</sub>, q<sub>&Phi;<sub>1</sub></sub> and q<sub>&Phi;<sub>2</sub></sub>
The NVAE models trained for five different seeds, used in table 3 of [MISELBO](https://arxiv.org/), are too 
large to put in this repo. Therefore, one model is selected: p<sub>&theta;</sub>, q<sub>&Phi;<sub>1</sub></sub> trained 
with seed = 3 and q<sub>&Phi;<sub>2</sub></sub> trained with seed = 0. In order to run MISELBO for this ensemble follow
the steps below:
1. Follow steps 1-3 from the previous section.
2. Copy "models" folder, including subfolders and files to the NVAE root folder.
3. Install MISELBO specific packages using the requirements.txt file.
4. In the NVAE root folder, run command:
``` 
python miselbo_eval.py 
```

## MISELBO for own model




