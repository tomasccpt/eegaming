# EEGaming

EEGaming was a project that aimed to build a Brain-Computer Interface (BCI) technology using motor imagery to interact with video games. We designed a system capable of recognizing four distinct motor imagery actions - each mapped to a corresponding directional input in the game - in a slowed down real time simmulation This project not only utilized custom data collected with the OpenBCI platform but also used a public clinical dataset comprising more than 100 subjects to enhance the robustness and accuracy of our methods. We created a pipeline that builds a personalized model for each subject, and uses a linear classifier on SWDCSP filtered data.

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
    - The clinical dataset used can be found at https://physionet.org/content/eegmmidb/1.0.0/
    - Our collected dataset was uploaded to kaggle, and can be found at https://www.kaggle.com/datasets/miguelrcborges/motor-imagery-dataset. Download it and place it in a folder in the root called "data"

4. **Process collected dataset**
   1. Run Process_Collected/segmentation.ipynb
   2. Run Process_Collected/complete.ipynb

6. **Run Simmulation**
    Run real_time.py using the installed dependecies

## Credits

This project was developed by:
Tomás Cruz, up202008274
Miguel Borges, up202004481
Simão Francisco, up201907198
Francisco Magalhães, 202005141

For the course of Neuroengineering, at the Faculty of Engineering of the University of Porto, in the academic year 2023/2024, under the supervision of Professor João Paulo Cunha.
