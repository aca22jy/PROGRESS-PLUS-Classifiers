# PPlus Multilabel Classifier

This project implements a multilabel text classifier using RoBERTa-large (or other BERT-based models) to categorize text data based on multiple predefined labels. The script is designed to train a model on a dataset and then evaluate its performance.

## Prerequisites

Ensure you have Python installed, along with the following libraries:
- transformers
- torch
- pandas
- scikit-learn
- numpy

You can typically install these using pip:
```sh
pip install transformers torch pandas scikit-learn numpy
```

## Data

The script expects a TSV file named `ProgressTrainingCombined.tsv` located in a directory `../sources/` relative to the script's location. This file should contain columns like `PaperTitle`, `Abstract`, `JN` (Journal Name), and the label columns (`Place`, `Race`, `Occupation`, `Gender`, `Religion`, `Education`, `Socioeconomic`, `Social`, `Plus`).

## Running the Script

**Important:** Before running the script, navigate to the `Plus_classifiers` directory in your terminal. The script uses relative paths (e.g., `../sources/ProgressTrainingCombined.tsv`) to access the dataset. If you run the script from a different directory, it will not be able to find the data file and will result in an error.

```sh
cd path/to/your/project/Plus_classifiers
python PPlus_multilabel_RoBERTa_large.py [OPTIONS]
```

If you prefer to run the script from a different location, you will need to modify the file paths for loading the dataset (e.g., `../sources/ProgressTrainingCombined.tsv` in [`PPlus_multilabel_RoBERTa_large.py`](PPlus_multilabel_RoBERTa_large.py)) to be absolute paths or relative paths from your chosen execution directory.

### Command-Line Options

The script accepts several command-line arguments to customize its behavior:

-   `--test`: (Optional) If specified, runs the script in a test mode. This typically uses a smaller subset of data (300 samples) and a smaller `MAX_LEN` (20) for quick testing.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --test
    ```
-   `--epoch` / `-e`: (Optional) Specifies the number of training epochs. Default is `30`.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --epoch 50
    ```
-   `--max_len` / `-m`: (Optional) Sets the maximum sequence length for the tokenizer. Default is `512`.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --max_len 256
    ```
-   `--learning_rate` / `-l`: (Optional) Sets the learning rate for the optimizer. Default is `5e-06`.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --learning_rate 1e-5
    ```
-   `--train_batch_size` / `-t`: (Optional) Defines the batch size for training. Default is `6`.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --train_batch_size 8
    ```
-   `--journal_name` / `-j`: (Optional) If specified, the journal name (`JN` column) will be concatenated with the paper title and abstract as input text to the model.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --journal_name
    ```
-   `--bert_model` / `-b`: (Optional) Specifies the BERT-based model to use from the Hugging Face model hub. Default is `roberta-large`. You can change this to other compatible models like `allenai/scibert_scivocab_uncased`, etc.
    ```sh
    python PPlus_multilabel_RoBERTa_large.py --bert_model allenai/scibert_scivocab_uncased
    ```

### Example Usage

Assuming you are in the `Plus_classifiers` directory:
To train the model with default RoBERTa-large for 20 epochs, a max length of 256, a learning rate of 3e-5, a training batch size of 8, and including the journal name:
```sh
python PPlus_multilabel_RoBERTa_large.py --epoch 20 --max_len 256 --learning_rate 3e-5 --train_batch_size 8 --journal_name --bert_model roberta-large
```

## Output

-   **Trained Model**: The best performing model (based on micro F1-score on the validation set) will be saved in the `../results/` directory (relative to `Plus_classifiers`). The model filename is dynamically generated based on the model type, variant (base/large), batch size, learning rate, and number of epochs (e.g., `../results/roberta_large_bs6_lr5.0e-06_ep30.pt`).
-   **Results CSV**: A CSV file containing the text, ground truth labels, predicted labels, and prediction probabilities for the test set will be saved in the `../results/` directory. The filename is also dynamically generated (e.g., `../results/roberta_512len_6b_30e_multilabel_results.csv` or `../results/JN_roberta_512len_6b_30e_multilabel_results.csv` if `--journal_name` is used).
-   **Console Output**: The script will print training progress, average loss per epoch, F1 scores, recall, precision, and Brier score for each label, as well as micro and macro F1 scores, and ROC AUC score.

## Key Features

-   **Multilabel Classification**: Handles tasks where each input can belong to multiple categories.
-   **BERT-based Models**: Utilizes transformer models like RoBERTa for text encoding.
-   **Customizable Training**: Offers various command-line arguments for hyperparameter tuning.
-   **Weighted Loss**: Implements a custom focal loss with class-specific weights and gammas to handle imbalanced datasets.
-   **Early Stopping**: Includes an early stopping mechanism to prevent overfitting and save the best model.
-   **Gradient Accumulation**: Uses gradient accumulation to simulate larger batch sizes with limited GPU memory.