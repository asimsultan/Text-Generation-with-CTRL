
# Text Generation with CTRL

Welcome to the Text Generation with CTRL project! This project focuses on generating text using the CTRL model.

## Introduction

Text generation involves creating text based on input prompts. In this project, we leverage CTRL to generate text using a dataset of text examples.

## Dataset

For this project, we will use a custom dataset of text examples. You can create your own dataset and place it in the `data/text_generation_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/text_generation_ctrl.git
cd text_generation_ctrl

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes text examples. Place these files in the data/ directory.
# The data should be in a CSV file with one column: text.

# To fine-tune the CTRL model for text generation, run the following command:
python scripts/train.py --data_path data/text_generation_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/text_generation_data.csv
