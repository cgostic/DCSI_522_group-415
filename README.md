# L1c3nc3 t0 C0d3  
**DCSI_522_group-415**  
Authors: Keanna Knebel, Cari Gostic, Furqan Khan

## Project Summary  
This project attempts to identify character strings that predict whether or not a vanity liscence plate submitted to the New York State Department of Motor Vehicles will be rejected. The dataset includes all accepted vanity license plate applications submitted between October, 2010 and September, 2014, and all license plates that passed an initial automatic screen against a red-list, but were ultimately rejected upon inspection by clerical staff. If strong predictors are identified, these may be added to the red list to make the initial screening of applications more effective, and therefore, reduce the time the clerical staff spends on inspecting re-submissions from applicants whose initial submissions were rejected in secondary screening.

## Scipt Flow chart

![](script_flowchart.png)

## Usage

1. 01_download_data.R
```
Rscript scripts/01_download_data.R "data/" "accepted_plates.csv" "https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv" "rejected_plates.csv" "https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv"
```

2. 02_data_preprocessing.py


3. 03_EDA.py
```
python scripts/03_EDA.py --file_path_data="data/" --accepted_plates_csv="accepted_plates.csv" --rejected_plates_csv="rejected_plates.csv" --reduced_plate_csv="full_vanity_plate_data.csv" --X_train_csv="X_train.csv" --y_train_csv="y_train.csv" --file_path_img="docs/imgs/"
```

4. 
```
python scripts/04_data_model.py --file_path_read="data/" --filename_x_train="X_train.csv" --filename_x_validate="X_validate.csv" --filename_x_test="X_test.csv" --filename_y_train="y_train.csv" --filename_y_validate="y_validate.csv" --filename_y_test="y_test.csv" --filename_path_write="docs/imgs/" --filename_cl_table="classification_report" --filename_predictor_table="best_predictors"
```

5. 05_final_report.rmd
```
Rscript -e "rmarkdown::render('scripts/05_final_report.rmd', output_file = 'docs/final_report.html')"
```