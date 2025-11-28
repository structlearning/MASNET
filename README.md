# Neural Monotone Set Functions
A framework for training and evaluating **set neural models** — including `MasNet`, `ReLUMasNet`, `DeepSets`, and `SetTransformer` — on benchmark tasks such as `FacilityLocation` and `SubSet`. Built on **PyTorch Lightning**, the framework provides reproducible training, automatic checkpointing, and optional **out-of-distribution (OOD)** evaluation. 

## Installation
```bash
git clone https://github.com/yonatansverdlov/Monotone-Embedding.git
cd <your_repo>
conda env create -f dependencies.yml
conda activate <environment_name_from_yml>
```
*(Make sure `torch`, `pytorch-lightning`, and `easydict` are included in your `requirements.txt`.)*

## 
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

## Command-Line Arguments
| Argument | Type | Default | Description |
|-----------|------|----------|--------------|
| `--task_type` | str | `FacilityLocation` | Task name (`FacilityLocation` or `SubSet`) |
| `--model_type` | str | `MasNet` | Model architecture (`ReLUMasNet`, `DeepSets`, `MasNet`, `SetTransformer`) |



## 📁 Project Structure
```
.
├── MonotoneModel_real
├── MonotoneModel_synthetic
├── ddependencies.yaml
├── README.md
```

## Citation
If you use this repository in your research, please cite:
```text
@misc{sarangi2025monotoneseparablesetfunctions,
      title={Monotone and Separable Set Functions: Characterizations and Neural Models}, 
      author={Soutrik Sarangi and Yonatan Sverdlov and Nadav Dym and Abir De},
      year={2025},
      eprint={2510.23634},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23634}, 
}
```


## License
MIT License © 2025 Yonatan Sverdlov
