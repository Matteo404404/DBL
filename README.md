# DataChallenge JBG030

This repository contains the complete implementation for the JBG030 course data challenge, focused on airline customer service analysis through social media data mining, sentiment analysis, and influencer identification.

**Note**: This project is distributed as a zip file. Extract all contents to your desired location before proceeding with the installation.

## Table of Contents
- [Requirements](#requirements)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Sprint Overview](#sprint-overview)

## Requirements

### System Requirements
- **Python Version**: 3.11+
  - **Operating System**: Windows, macOS, or Linux
  - **Memory**: Minimum 8GB RAM recommended

### External Tools
- **Jupyter Notebook**: For interactive data analysis
  - Installation guide: [https://jupyter.org/install](https://jupyter.org/install)
  - **Label Studio**: For manual data labeling
    - Installation guide: [https://labelstud.io/guide/install.html](https://labelstud.io/guide/install.html)
  - **MongoDB**: For data storage
    - Installation guide: [https://docs.mongodb.com/manual/installation/](https://docs.mongodb.com/manual/installation/)

## Installation

### 1. Extract the Project Files
```bash
# Extract the zip file to your desired location
# Navigate to the extracted folder
cd DataChallenge-JBG030
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create requirements.txt
Create a `requirements.txt` file with the following dependencies:
```
# External libraries (need to be installed)
pandas>=2.2.3
numpy>=1.26.4
matplotlib>=3.10.3
seaborn>=0.13.2
plotly>=6.1.2
scikit-learn>=1.7.0
torch>=2.7.1
transformers>=4.52.4
datasets>=3.6.0
evaluate>=0.4.1
pymongo>=4.13.2
networkx>=3.5
tqdm>=4.67.1
ftfy>=6.3.1
jupyter>=1.0.0
ipython>=9.2.0

# Built-in Python libraries (included for reference)
# These come with Python installation - no need to install separately:
# sys
# os
# re
# json
# time
# csv
# argparse
# traceback
# html
# unicodedata
# collections
# pathlib
# datetime
# typing
# warnings
# bson (comes with pymongo)
```

## Project Structure

This repository is organized into three main sprint folders, each corresponding to a project phase:

```
DataChallenge-JBG030/
├── .git/               # Contains Git version control history and metadata.
├── .idea/              # Stores project-specific settings for JetBrains IDEs (e.g., PyCharm).
├── .venv/              # Holds the isolated Python virtual environment and its dependencies.
├── Archive/            # A place for old or backed-up files not in active use.
├── CSVs/               # Storage for data files in CSV format, including generated metrics.
├── ExtraAnalysis/      # Scripts for supplementary analysis, such as hypothesis testing.
├── images/             # Contains static image files, plots, and visualizations.
├── JSONs/              # Stores raw data exports and other data in JSON format.
├── __pycache__/        # Automatically generated by Python to store compiled bytecode.
├── SPRINT_1/           # Code and notebooks for data cleaning, database setup, and initial EDA.
│   ├── ...
├── SPRINT_2/           # Scripts and notebooks for conversation mining and core sentiment analysis.
│   ├── ...
└── SPRINT_3/           # Advanced analytics including multilingual models and influencer analysis.
    ├── ...
```
## Folder Overview

> A quick guide to what lives where in this repository.

- **`.git/`**  
  tracks all changes and version history via Git.

- **`.idea/`**  
  IDE-specific settings for JetBrains tools like PyCharm. Contains configs like interpreter paths, run settings, and code style preferences.

- **`.venv/`**  
  This virtual environment holds a dedicated interpreter and all the project’s dependencies.

- **`Archive/`**  
  Where old files go to retire. 
- **`CSVs/`**  
  The csv data includes input datasets and output results — like `metrics_<airline>.csv` — ready for analysis and visualization.

- **`ExtraAnalysis/`**  
  Contains scripts outside the main sprint workflow, hypothesis testing (`HYP_test_airlines.py`) and exploratory visualizations.

- **`images/`**  
  Some visual assets in one place, charts, graphs, and figures used in docs, reports, or notebooks.

- **`JSONs/`**  
  Raw data lives here. This folder stores unprocessed JSON files.

- **`__pycache__/`**  
  Auto-generated by Python to store compiled `.pyc` files. Improves performance.

- **`SPRINT_1/`, `SPRINT_2/`, `SPRINT_3/`**  
  The core of the project. Each folder contains code, notebooks, and outputs specific to the respective sprint phase.

---

## 🚀 Usage


### Prerequisites
Before running any scripts, ensure you have:
1. **Python IDE**: 
   - **VS Code**: Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - **PyCharm**: Download from [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)
   - Install Python extension for VS Code or ensure PyCharm has Python support enabled
   2. **Virtual Environment**: Set up and activated as described in the installation section
   3. **MongoDB Running**: Start your MongoDB service

### Step-by-Step Execution

#### 1. Data Cleaning
Clean raw tweet exports and normalize text data.

**In VS Code:**
- Open the file `SPRINT_1/Data_Cleaning_SPRINT_1/Data_Cleaning.py`
  - Right-click in the editor and select "Run Python File in Terminal"
  - Or use Ctrl+F5 to run without debugging

**In PyCharm:**
- Open the file `SPRINT_1/Data_Cleaning_SPRINT_1/Data_Cleaning.py`
  - Right-click in the editor and select "Run 'Data_Cleaning'"
  - Or press Ctrl+Shift+F10

#### 2. Database Setup & Data Upload
Initialize MongoDB collections and upload cleaned data.

**Step 2.1 - Create Database Structure:**
- **VS Code**: Open and run `SPRINT_1/Data_Upload_SPRINT_1/DatabaseCreation.py`
  - **PyCharm**: Open and run `SPRINT_1/Data_Upload_SPRINT_1/DatabaseCreation.py`

**Step 2.2 - Upload Data:**
- **VS Code**: Open and run `SPRINT_1/Data_Upload_SPRINT_1/upload.py`
  - **PyCharm**: Open and run `SPRINT_1/Data_Upload_SPRINT_1/upload.py`

#### 3. Conversation Mining
Extract per-thread metrics (reply rates, sentiment changes, etc.) for each airline.

**In VS Code:**
- Open the file `SPRINT_2/Convo_Mining_Sprint_2/Mining.py`
  - Right-click in the editor and select "Run Python File in Terminal"
  - Or use Ctrl+F5 to run without debugging

**In PyCharm:**
- Open the file `SPRINT_2/Convo_Mining_Sprint_2/Mining.py`
  - Right-click in the editor and select "Run 'Mining'"
  - Or press Ctrl+Shift+F10

**Output**: Generates `metrics_<airline>.csv` files

#### 4. Multilingual Sentiment Analysis

##### 4.1 Prepare Fine-tuning Data
- **VS Code/PyCharm**: Open and run `SPRINT_3/Sentiment_Analysis_Sprint_3/adding_prefixes_for_better_fine_tuning.py`

##### 4.2 Manual Labeling
1. Use Label Studio for manual annotation
   2. Export labeled data
   3. Convert to CSV format:
      - **VS Code/PyCharm**: Open and run `SPRINT_3/Sentiment_Analysis_Sprint_3/manual_labeling_multilingual.py`

##### 4.3 Model Training
**Train model from scratch:**
- **VS Code/PyCharm**: Open and run `SPRINT_3/Sentiment_Analysis_Sprint_3/multilingual_fine_tuning_threads.py`

**Resume/refine existing training:**
- **VS Code/PyCharm**: Open and run `SPRINT_3/Sentiment_Analysis_Sprint_3/fine_tuning_threads_multilingual.py`

##### 4.4 Thread Analysis
- **VS Code/PyCharm**: Open and run `SPRINT_3/Sentiment_Analysis_Sprint_3/threads_analysis.py`

#### 5. Additional Analysis Scripts (Sprint 3)

**EDA and Analysis:**
- **VS Code/PyCharm**: Open and run `SPRINT_3/EDA_Sprint_3/analyze_results.py`
  - **VS Code/PyCharm**: Open and run `SPRINT_3/EDA_Sprint_3/main_analysis.py`
  - **VS Code/PyCharm**: Open and run `SPRINT_3/EDA_Sprint_3/threads_analysis.py`
  - **VS Code/PyCharm**: Open and run `SPRINT_3/EDA_Sprint_3/zero_shot_approach.py`

#### 6. Customer Service & Influencer Analysis

##### 6.1 Compute Metrics
- **VS Code/PyCharm**: Open and run `SPRINT_3/Influencer_User_Business_Idea_Sprint_3/influencer_number.py`

**Output**: Generates `metrics_<airline>.csv`

##### 6.2 Statistical Analysis
**Run hypothesis tests:**
- **VS Code/PyCharm**: Open and run `ExtraAnalysis/HYP_test_airlines.py`

**Generate comparison visualizations:**
- **VS Code/PyCharm**: Open and run `ExtraAnalysis/airline_comparison_vis.py`

### Working with Jupyter Notebooks

The project includes several Jupyter notebooks for exploratory data analysis and visualization:

**Sprint 1 EDA:**
- `SPRINT_1/SPRINT_1_EDA/*.ipynb` - Initial exploratory data analysis

**Sprint 2 EDA:**
- `SPRINT_2/EDA_Sprint_2/*.ipynb` - Sprint 2 analysis notebooks
  - `SPRINT_2/EDA_MINING/*.ipynb` - Mining-specific EDA notebooks
  - `SPRINT_2/Sentiment_Analysis_Sprint_2/Airline_Sentiment_Comparison_EDA.ipynb` - Airline sentiment comparison
  - `SPRINT_2/Sentiment_Analysis_Sprint_2/sentiment_analysis.ipynb` - General sentiment analysis

**Sprint 3 EDA:**
- `SPRINT_3/EDA_Sprint_3/vis.ipynb` - Visualization notebook

**Running Notebooks in VS Code:**
1. Install the Jupyter extension for VS Code
   2. Open any `.ipynb` file directly in VS Code
   3. Select your Python interpreter (the one from your virtual environment)
   4. Run cells individually or run all

**Running Notebooks in PyCharm Professional:**
1. Open any `.ipynb` file directly in PyCharm
   2. PyCharm will automatically recognize them as Jupyter notebooks
   3. Run cells individually or run all

**Alternative - Jupyter in Browser:**
If you prefer the traditional Jupyter interface:
1. Open the integrated terminal in your IDE
   2. Run: `jupyter notebook`
   3. Navigate to the notebook files in your browser

## Sprint Overview

### SPRINT 1: Data Foundation
- **Data Cleaning**: Raw data preprocessing and normalization
  - **Database Setup**: MongoDB collection creation and data upload
  - **Output**: Clean, structured data ready for analysis

### SPRINT 2: Core Analysis
- **Conversation Mining**: Extract thread-level metrics and patterns
  - **Exploratory Data Analysis**: Initial data insights and visualizations
  - **Sentiment Analysis**: Basic sentiment classification and comparison
  - **Output**: Thread metrics, sentiment labels, and initial findings

### SPRINT 3: Advanced Analytics
- **Multilingual Processing**: Enhanced sentiment analysis for multiple languages
  - **Influencer Identification**: Business intelligence for top influencers
  - **Advanced Mining**: Metadata enrichment and complex queries
  - **Output**: Comprehensive analytics, influencer insights, and business recommendations

## Troubleshooting

### Common Issues
1. **Python Version**: Ensure you're using Python 3.11
   2. **Dependencies**: If packages fail to install, try upgrading pip: `pip install --upgrade pip`
   3. **MongoDB Connection**: Verify MongoDB is running and accessible
   4. **Memory Issues**: Close other applications if running out of memory during model training

### Getting Help
- Check the individual script files for specific error messages
  - Ensure all file paths are correct relative to the project root
  - Verify input data files exist before running scripts
  - email us at m.melis@student.tue.nl or y.dandriyal@student.tue.nl

## License
This project is for educational purposes as part of the JBG030 course.