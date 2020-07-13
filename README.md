
# L1c3nc3 t0 C0d3  
**DCSI_522_group-415**  
Authors: Keanna Knebel, Cari Gostic, Furqan Khan

## Project Summary  
This project attempts to identify character strings that predict whether or not a vanity liscence plate submitted to the New York State Department of Motor Vehicles will be rejected. The dataset includes all accepted vanity license plate applications submitted between October, 2010 and September, 2014, and all license plates that passed an initial automatic screen against a red-list, but were ultimately rejected upon inspection by clerical staff. If strong predictors are identified, these may be added to the red list to make the initial screening of applications more effective, and therefore, reduce the time the clerical staff spends on inspecting re-submissions from applicants whose initial submissions were rejected in secondary screening.

## Report
The final report can be found [here.](https://cgostic.github.io/license_2_code/docs/05_generate_report.html)

## Scipt Flow chart

![](script_flowchart.png)

## Usage

### 1. Using Docker
*note - the instructions in this section also depends on running this in a unix shell (e.g., terminal or Git Bash), if you are using Windows Command Prompt, replace `/$(pwd)` with PATH_ON_YOUR_COMPUTER.*

1. Install [Docker](https://www.docker.com/get-started)
2. Download/clone this repository
3. Use the command line to navigate to the root of this downloaded/cloned repo
4. Type the following to run the analysis:

```
docker run --rm -v /$(pwd):/home/522_project fkhan72/522_proj:v1.0 make -C /home/522_project all
```

5. Type the following to clean up the analysis  

```
docker run --rm -v /$(pwd):/home/522_project fkhan72/522_proj:v1.0 make -C /home/522_project clean
```

### 2. Using Bash/Terminal 

To replicate the analysis performed in this project, clone this GitHub repository, install the required [dependencies](#package-dependencies) listed below, and run the following commands in your command line/terminal from the root directory of this project:

1. 01_download_data.R
```
Rscript scripts/01_download_data.R --file_path="data/raw" --filename_1="accepted_plates.csv" --url1="https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv" --filename_2="rejected_plates.csv" --url2="https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv"
```

2. 02_data_processing.py
```
python scripts/02_data_processing.py --file_path_read="data/raw/" --file_path_write="data/processed/" --accepted_plates_csv="accepted_plates.csv" --rejected_plates_csv="rejected_plates.csv" 
```

3. 03_EDA.py
```
python scripts/03_EDA.py --file_path_raw="data/raw/" --file_path_pro="data/processed/" --accepted_plates_csv="accepted_plates.csv" --rejected_plates_csv="rejected_plates.csv" --X_train_csv="X_train.csv" --y_train_csv="y_train.csv" --file_path_img="results/"
```

4. 04_data_model.py
```
python scripts/04_data_model.py --file_path_read="data/processed/" --filename_x_train="X_train.csv" --filename_x_validate="X_validate.csv" --filename_x_test="X_test.csv" --filename_y_train="y_train.csv" --filename_y_validate="y_validate.csv" --filename_y_test="y_test.csv" --filename_path_write="results/"
```

5. 05_general_report.rmd
```
Rscript -e "rmarkdown::render('docs/05_generate_report.rmd')"
```

#### Running complete project

To run the entire project, run the following commands in your command line/terminal from the root directory of this project:

```
make all
```

To clear the generated outputs from the scripts, run the following commands in your command line/terminal from the root directory of this project:

```
make clean
```

#### Make file graph

![](Makefile_graph.png)

  
## Package Dependencies

### Python 3.7.3 and Python packages:

- pandas --0.25.3
- numpy --1.18.1
- scikit-learn --0.21.3
- altair --3.2.0
- docopt -- 0.6.2
- imgkit -- 1.0.2
- selenium --3.141.0
- subprocess.run --0.0.8

### R version 3.6.1 and R packages:

- tidyverse --1.2.1
- docopt --0.6.1
- knitr --1.27.2
- testit --0.11

### Other dependencies:

- wkhtmltopdf --0.12.4

