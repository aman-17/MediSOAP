# MediSOAP: MediSOAP: Enhanced Clinical Note Generation with Fine-Tuned Llama2

## Project Overview

This project involves fine-tuning the Llama2-7B model from scratch using LoRA and QLoRA techniques. The goal is to generate structured SOAP (Subjective, Objective, Assessment, Plan) notes from patient-doctor conversations. The dataset used for training comprises transcribed medical dialogues that follow the SOAP note format.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Fine-Tuning Process](#fine-tuning-process)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [License](#license)

## Introduction
SOAP notes are a method of documentation employed by healthcare providers to write out notes in a patient's chart, along with other common formats. This project automates the generation of SOAP notes from patient-doctor conversations using a fine-tuned Llama2-7B model. The model leverages Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) for efficient training.

## Prerequisites
- Python 3.11 or higher
- PyTorch 1.10.0 or higher
- CUDA 10.2 or higher (for GPU support)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aman-17/MediSOAP.git
   cd MediSOAP
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset used for this project is a collection of patient-doctor conversation transcripts formatted into SOAP notes. The dataset must be preprocessed into the required format before training.

To preprocess your custom dataset, follow the format of train.jsonl, then:
1. Place your raw data files in the `data/` directory.
2. Run the preprocessing script:
   ```bash
   python data_preprocessing.py
   ```

## Fine-Tuning Process

Fine-tuning involves adapting the pre-trained Llama2-7B and phi2 model to our specific task using LoRA technique.

### Steps:

1. **Data Preparation**:
   Ensure your preprocessed data is in the `data/` directory.

2. **Training**:
   Run the training script:
   ```bash
   python train_phi2.py
   ```

## Evaluation

Evaluate the model's performance on a test dataset:
```bash
python evaluate.py --model-path path/to/fine-tuned-model --test-data path/to/test-data
```

Metrics such as BLEU, ROUGE, and accuracy can be used to assess the model's performance.

## Usage

To generate SOAP notes from new patient-doctor conversations, use the inference script:
```bash
python generate.py --model-path path/to/fine-tuned-model --input path/to/conversation.txt
```

The output will be a structured SOAP note based on the input conversation.

## Results

Summarize the results obtained from the model's performance on the test dataset, including key metrics and example outputs.

## Contributing

We welcome contributions from the community. To contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to update this README with additional details as needed.