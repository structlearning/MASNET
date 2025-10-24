# 🧠 Graph Model Training Framework
A modular framework for training and evaluating **set- and graph-based neural models** — including `MasNet`, `ReLUMasNet`, `DeepSets`, and `SetTransformer` — on benchmark tasks such as `FacilityLocation` and `SubSet`. Built on **PyTorch Lightning**, the framework provides reproducible training, automatic checkpointing, and optional **out-of-distribution (OOD)** evaluation.

## 🚀 Features
- Unified interface for multiple architectures  
- Simple configuration via command-line arguments  
- PyTorch Lightning integration for clean training loops  
- Automatic OOD evaluation by doubling the input dimension  
- Reproducible experiments with fixed random seeds  

## 📦 Installation
```bash
git clone https://github.com/yonatansverdlov/Monotone-Embedding.git
cd <your_repo>
pip install -r requirements.txt
```
*(Make sure `torch`, `pytorch-lightning`, and `easydict` are included in your `requirements.txt`.)*

## 🧩 Usage
### Create Real Datasets
#### MSWEB or MSNBC
```bash
cd data
python msweb_data_generator.py --DATASET_NAME MSWEB
python msweb_data_generator.py --DATASET_NAME MSWEB
```
#### Amazon
```bash
cd data
python amazon.py --dataset_name bedding/feeding
```
#### PointCloud
```bash
cd data
python pointcloud.py --s1_size 128 & python pointcloud.py --s1_size 256 & python pointcloud.py --s1_size 128
```
### Train and evaluate on Text datasets
```bash
./run_datasets.sh
```
### Train and evaluate on PointCloud Datasets
```bash
./run_pointcloud.sh
```
### Train a model
```bash
python main.py --task_type FacilityLocation --model_type MasNet
```
### Use a different model or task
```bash
python main.py --task_type SubSet --model_type SetTransformer
```
### Run out-of-distribution (OOD) test
```bash
python main.py --model_type ReLUMasNet --test_dist
```

## ⚙️ Command-Line Arguments
| Argument | Type | Default | Description |
|-----------|------|----------|--------------|
| `--task_type` | str | `FacilityLocation` | Task name (`FacilityLocation` or `SubSet`) |
| `--model_type` | str | `MasNet` | Model architecture (`ReLUMasNet`, `DeepSets`, `MasNet`, `SetTransformer`) |

## 🧠 Example Output
```
Starting training...
Model MasNet | Task: FacilityLocation | m: 10, n: 20, d: 4 | Accuracy: 91.24 | OOD Accuracy: 84.67
Final Results Summary:
m: 10, n: 20, d: 4 -> Test Accuracy: 91.24 | Out-of-distribution: 84.67
```

## 📁 Project Structure
```
.
├── main.py
├── lightning_model.py
├── utils.py
├── requirements.txt
├── README.md
└── data/
```

## 🧬 Citation
If you use this repository in your research, please cite:
```text
@misc{graphmodels2025,
  author = {Sverdlov, Yonatan},
  title  = {Graph Model Training Framework},
  year   = {2025},
  note   = {Technion – Israel Institute of Technology}
}
```

## 🧩 Contributing
Pull requests are welcome!  
To add a new model:
1. Implement it in `lightning_model.py` or create a new file under `models/`.
2. Register it inside `get_args()` in `utils.py`.
3. Run and test with `python main.py --model_type YourModelName`.

## 📜 License
MIT License © 2025 Yonatan Sverdlov
