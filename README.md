# Semantic Segmentation with Tumors
This project focuses on detecting Invasive Ductal Carcinoma (IDC) on whole mount slide images and segmenting the regions where the tumor is present. We use a CNN-based approach for binary image classification at mask level, trained on the [BraTS2020 Dataset (Training + Validation)](https://www.med.upenn.edu/cbica/brats2020/data.html).

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
Laia Camara, Laia Alcalde, Elena Gutiérrez, Cristina Huanca
<!-- TODO: add mail of the group members -->

Xarxes Neuronals i Aprenentatge Profund
Grau d'Enginyeria de Dades,
UAB, 2026
