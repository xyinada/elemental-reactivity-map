# Elemental Reactivity Map

## Overview

**Elemental Reactivity Map** is a machine learning-based tool that predicts the probability of three different elements reacting to form ternary compounds.

- Generate training data from a list of known chemical compositions.
  - By changing the training data, the model can predict not only elemental reactivity but also elemental combinations that produce compounds with specific properties.
- Train a machine learning model using the generated training data.
- The predicted results are visualized in a heatmap.

This repository contains the code used in the following publication (**details to be added after publication**).

## Dependencies

The following libraries are required to run the project:

- TensorFlow (2.16 or later)
- NumPy
- Pandas
- scikit-learn (sklearn)
- Pymatgen
- Matplotlib
- Seaborn
- Bokeh
- tqdm

## Usage

### Execution

Prepare a training dataset and a configuration file (in JSON format), then run:

```bash
python3 main.py /path/to/your_config_file.json
```

### Training Data

The training data must be a CSV file with headers. At least one column should contain chemical compositions parsable by **Pymatgen**.
Additional columns can include physical properties or crystal structures, which will be used to generate training data.

### Configuration File

All settings related to data preprocessing, model training, and result visualization are defined in a `.json` configuration file.
Refer to `example/config/mp_example.json`.

Example configuration file structure:

```json
{
  "files_and_directories": {
    "data_source_file": "./example/data/mp.csv", // CSV file used as training data
    "source_name": "mp", // Custom name for the data source
    "save_dir": "./example/result_mp" // Directory to save output results
  },
  "condition": {
    "composition_column_name": "composition", // Column header for compositions in data file
    "property_column_name": "composition", // Column header for property data in data file
    "property_selection_method": "all", // Selection method: all, name, range, high, low.
                                        // all: use all composition,
                                        // name: use composition only the prpoerty_column value matchs "property_filter/name"
                                        // range: use composition `"property_filter/value_min" value <= prpoerty_column value <= "property_filter/value_max" value`
                                        // high: use compositions where the property_column value fall within the top fraction specified by property_filter/ratio.
                                        // low: use compositions where the property_column value fall within the bottom fraction specified by property_filter/ratio.
    "property_filter": {
      "value_max": -1,
      "value_min": -1,
      "name": "",
      "ratio": 0.5
    }
  },
  "data": {
    "data_num": {
      "positive_test_data": 2000, // Number (>=1) or ratio (0<value<1) of data used as positive test data
      "positive_train_data": 10000, // Number (>=1) or ratio (0<value<1) of data used as positive training data
      "unlabeled_test_data": 2000, // Number of data used as unlabeled test data
      "reliable_negative_train_data": 10000, // Number of data used as reliable negative training data
      "positive_train_data_dup": "auto" // Oversampling of positive data if needed.
                                        // When the value is "auto", positive_train_data are oversampled to be same the number of reliable_neagative_train_data.
                                        // When the value is number, positive_train_data are oversampled specified times.
    },
    "parameters": {
      "knn_k_list": [1, 2, 3, 4, 5], // k-NN parameters for data selection. Training data will be generated for all these values.
      "rn_lower_ratio_list": [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1], // Reliable negative data ratio. Training data will be generated for all these values.
      "element_parameter": [
        // Specify the file path of the input parameters.
        // These files must have the first line as a header, the first column as the element name, and the second and subsequent columns as the element parameters.
        // Of the parameters, the first d (“dim” value) parameters will be used for input.
        // If multiple files are specified, the parameters will be concatenated and used for input.
        {"source_file": "./example/parameter/CMP.csv", "dim": 80, "name": "CMP"},
        {"source_file": "./example/parameter/CRD.csv", "dim": 80, "name": "CRD"},
        {"source_file": "./example/parameter/TPL.csv", "dim": 80, "name": "TPL"}
      ]
    }
  },
  "learning": {
    "learning_rate": [0.0001], // Learning rate. Machine learning model will be generated for all these values.
    "batch_size": [100], // Batch size. Machine learning model will be generated for all these values.
    "epoch": [100], // Number of epochs. Machine learning model will be generated for all these values.
    "node_nums": [[128, 128, 128, 1]], // Neural network layer configuration. Machine learning model will be generated for all these values.
    "model_num": 10 // Number of models trained. The average predicted value of all models are used as predicted result.
  },
  "heatmaps": [
    {
      "name": "heatmap_mp", // Name of the heatmap
      "mark_sources": [
        {
          "file": "./example/data/mp.csv",
          "composition_column": "composition",
          "property_column": "property",
          "property_selection_method": "all",
          "property_filter": {
            "value_max": -1,
            "value_min": -1,
            "name": "",
            "ratio": 0.5
          },
          "mark": "★", // Mark symbol for matched compositions
          "label": "MP", // Custom label
          "square_color": "r" // Matplotlib color name
        }
        {
          "file":"./example/data/icsd_hq.csv",
          "composition_column": "composition",
          "property_column": "property",
          "propety_selection_method": "all",
          "property_filter": {
            "value_max": -1,
            "value_min": -1,
            "name": "",
            "ratio": 0.5
          },
          "mark": "◆",
          "label": "ICSD (HQ)",
          "square_color": "b"
        },
        {
          "file":"./example/data/icsd_lq.csv",
          "composition_column_name": "composition",
          "property_column": "property",
          "propety_selection_method": "all",
          "property_filter": {
            "value_max": -1,
            "value_min": -1,
            "name": "",
            "ratio": 0.5
          },
          "mark": "▲",
          "label": "ICSD (LQ)",
          "square_color": "b"
        }
      ]
    }
    {
        "name": "heatmap_mp_blue",
        "mark_sources": [] // default square color is blue
    }
  ]
}
```

## Output

The program generates the following outputs:

- Training data files in `.pkl` format.
- Machine learning model files and training results using TensorFlow.
- Elemental reactivity maps as PNG and interactive HTML files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

