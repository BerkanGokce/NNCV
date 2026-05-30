![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-green)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision-orange)
![Dataset](https://img.shields.io/badge/Dataset-Cityscapes-yellow)
![Benchmark](https://img.shields.io/badge/Benchmark-Performance%20vs%20Efficiency-purple)
![Platform](https://img.shields.io/badge/Platform-Snellius%20HPC-lightgrey)
![Tracking](https://img.shields.io/badge/Tracking-WandB-blueviolet)

# NNCV – Semantic Segmentation Project (Cityscapes)

This repository contains implementations of multiple deep learning models for semantic segmentation on the Cityscapes dataset. The goal of this project is to analyze the trade-off between segmentation accuracy and computational efficiency by comparing different model architectures under two benchmarks: peak performance and efficiency.

For a detailed explanation of the methodology, experiments, and results, please refer to the accompanying research paper:

**"Exploring Performance–Efficiency Trade-offs in Semantic Segmentation on Cityscapes"**

The codebase is written in a modular structure. Each model consists of separate files for architecture definition (`model.py`), training (`train.py`), inference (`predict.py`), and execution (`main.sh`). This design allows easy modification, testing, and extension of different models.

The folder named **All Models** includes all implemented models. One can test existing models or add new ones to this folder.

To train a new or existing model, copy and paste the files of the model you want to train (`main.sh`, `train.py`, `model.py`, and `predict.py`) inside the **HPC** folder.

Push the repository to your GitHub repository using:

```
git add .
git commit -m "Add new model"
git push
```

Then connect to the Snellius server through the MobaXterm terminal using your own Snellius account, and pull the changes from your GitHub repository:

```
git clone <your-repo-link>
cd NNCV
cd HPC
```

This will create the NNCV project folder on the Snellius system.

Make sure the dataset and the container (`container.sif`) are installed inside the **HPC** folder. If not, run the following commands:

```
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

Note: The dataset may take several runs to fully download. Keep running:

```
sbatch download_docker_and_data.sh
```

until everything is fully downloaded.

To start training:

- Enter your WandB API key and HuggingFace token inside the `.env` file  
- Make sure the correct hyperparameters are set in `main.sh`  
- Select an appropriate GPU and sufficient training time inside `jobscript_slurm.sh`  

Then run the following commands inside the **HPC** folder:

```
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

This will return a **JOBID**. Training will start when a compute node becomes available.

You can check the status of your job using:

```
squeue
```

Or monitor training logs in real time using:

```
tail -f slurm-JOBID.out
```

When the training ends, find your best trained model in the `checkpoints` folder. Download the `.pt` file and place it inside the **HPC** folder in your local repository.

To create a submission for the evaluation server, after placing the trained model in the **HPC** folder and naming it `model.pt`, run the following commands from the NNCV root directory:

```
docker build -t nncv-submission:latest -f "HPC/Dockerfile" "HPC"
docker save -o nncv_submission.tar nncv-submission:latest
```

This will create a `.tar` file ready for submission.

Submissions have been made using the email address: **y.b.gokce@student.tue.nl**

## Peak Performance Benchmark
http://131.155.126.249:5001/

| Team Name              | Model                                   |
|----------------------|-----------------------------------------|
| Berkan_baseline_sub1 | UNet (peak-performance baseline)        |
| Berkan_Peak_v1       | DeepLabV3 ResNet50                      |
| Berkan_Peak_v2       | DeepLabV3 ResNet101 (Highest Performance) |
| Berkan_Peak_v3       | DeepLabV3Plus ResNet101                 |

## Efficiency Benchmark
http://131.155.126.249:5003/

| Team Name               | Model                          |
|------------------------|--------------------------------|
| Berkan_Efficiency_v0   | UNet (efficiency baseline)     |
| Berkan_Efficiency_v1   | DeeplabV3Plus MobileNetV3      |
| Berkan_Efficiency_v2   | ENet (Highest Efficiency)      |
| Berkan_Efficiency_v3   | BisENetV2                      |


## Trained Model Weights

Due to file size limitations, the trained model weights (.pt files) are not included directly in this repository. They are available via Google Drive:

[Download all trained models](https://drive.google.com/drive/folders/1Xo7UmpkR7ZRilUFocdES78TjcQ51rKRq?usp=drive_link)

The folder contains the following models:

| Model | Description |
|------|------------|
| UNet-Benchmark-best-model.pt | Baseline model |
| DeepLabV3-ResNet101-PeakPerformance-best-model.pt | Best peak-performance model |
| DeepLabV3-ResNet50-PeakPerformance-best-model.pt | Additional peak-performance submission |
| DeepLabV3Plus-ResNet101-PeakPerformance-best-model.pt | Additional peak-performance submission |
| ENet-Efficiency-best-model.pt | Most efficient model |
| BisENetV2-Efficiency-best-model.pt | Additional efficiency submission |
| DeeplabV3Plus-MobileNetV3-Efficiency-best-model.pt | Additional efficiency submission |


To use a trained model:

1. Download the desired `.pt` file  
2. Place it inside the `HPC` folder along with the other files of the model (`main.sh`, `train.py`, `model.py`, and `predict.py`)
3. Rename it to:

```
model.pt
```

4. Run inference or create submission as described above