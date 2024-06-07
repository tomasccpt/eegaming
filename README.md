# EEGaming

Welcome to the EEGaming repository, a research-based project designed to explore the intersection of electroencephodelic (EEG) data collection and gaming. This project facilitates the acquisition, processing, and application of EEG data in real-time simulations, leveraging machine learning models for classification within the scope of a custom-built game environment.

## Repository Structure

- **Acquisition/**: This folder contains all the data collected during the EEG sessions.
- **Classification/**: Notebooks within this directory are used for training classification models using the datasets available in the Acquisition folder.
- **dataset_processing/**: Scripts for processing the clinical datasets are stored in this folder.
- **dataset/**: Clinial data., found at https://physionet.org/content/eegmmidb/1.0.0/
- **game/**: Here you will find the scripts and executables of the custom game developed as a part of this project.
- **process_collected/**: Contains scripts that process the data acquired by the acquisition script.
- **protocol/**: Includes the protocols and routines for EEG data acquisition.

- **real_time.py**: This script runs the project in a real-time simulation setup, integrating the game with live EEG data processing.

- **requirements.txt**: Lists all the necessary libraries and their versions required to run the scripts within this repository effectively.

## Getting Started

To set up and start using this project, please follow the steps below:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/tomasccpt/eegaming.git
    cd eegaming
    ```

2. **Install dependencies**:
    Ensure you have Python installed on your system, and then run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Datasets**:
    The clinical dataset used can be found at https://physionet.org/content/eegmmidb/1.0.0/
    Our collected dataset was uploaded to kaggle, and can be found at https://www.kaggle.com/datasets/miguelrcborges/motor-imagery-dataset

4. **Process collected dataset with the scripts in /process_collected**

5. **Run real_time.py using the installed dependecies**

## Credits

This project was developed by:
Tomás Cruz, up202008274
Miguel Borges, up202004481
Simão Francisco, up201907198
Francisco Magalhães, 202005141

For the course of Neuroengineering, at the Faculty of Engineering of the University of Porto, in the academic year 2023/2024, under the supervision of Professor João Paulo Cunha.
