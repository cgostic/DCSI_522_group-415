# License plate analysis pipe
# author: Keanna
# date: 2020-01-29

# run entire analysis from scratch
all : docs/05_generate_report.html 

# load in data 
data/raw/accepted_plates.csv data/raw/rejected_plates.csv : scripts/01_download_data.R 
	Rscript scripts/01_download_data.R --file_path=data/raw/ --filename_1=accepted_plates.csv --url1=https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv --filename_2=rejected_plates.csv --url2=https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv
 

# data pre-processing (e.g., undersample and split into train, validate & test)
data/processed/X_train.csv data/processed/X_validate.csv data/processed/X_test.csv data/processed/y_train.csv data/processed/y_validate.csv data/processed/y_test.csv data/processed/full_vanity_plate_data.csv : scripts/02_data_processing.py 
	python scripts/02_data_processing.py --file_path_read=data/raw/ --file_path_write=data/processed/ --accepted_plates_csv=accepted_plates.csv --rejected_plates_csv=rejected_plates.csv --reduced_plates_csv=full_vanity_plate_data.csv 

# exploratory data analysis - visualize propotions of classes and n-gram features
results/examples_per_classification.png results/class_proportion_bl.png ngram_length_counts.png : scripts/03_EDA.py data/raw/accepted_plates.csv data/raw/rejected_plates.csv data/processed/full_vanity_plate_data.csv data/processed/X_train.csv data/processed/y_train.csv
	python scripts/03_EDA.py --file_path_raw=data/raw/ --file_path_pro=data/processed/ --accepted_plates_csv=accepted_plates.csv --rejected_plates_csv=rejected_plates.csv --reduced_plate_csv=full_vanity_plate_data.csv --X_train_csv=X_train.csv --y_train_csv=y_train.csv --file_path_img=results/

# create and tune data model 
results/train_val_error.png results/clf_val_report.png results/clf_test_report.png results/predictors_2_2.png results/predictors_3_3.png results/predictors_4_4.png : scripts/04_data_model.py data/processed/X_train.csv data/processed/X_validate.csv data/processed/X_test.csv data/processed/y_train.csv data/processed/y_validate.csv data/processed/y_test.csv
	python scripts/04_data_model.py --file_path_read=data/processed/ --filename_x_train=X_train.csv --filename_x_validate=X_validate.csv --filename_x_test=X_test.csv --filename_y_train=y_train.csv --filename_y_validate=y_validate.csv --filename_y_test=y_test.csv --filename_path_write=results/ 
	rm results/*.html

# render final report
docs/05_generate_report.html : docs/05_generate_report.Rmd docs/vanity_plates.bib results/examples_per_classification.png results/class_proportion_bl.png ngram_length_counts.png results/train_val_error.png results/clf_val_report.png results/clf_test_report.png results/predictors_2_2.png results/predictors_3_3.png results/predictors_4_4.png 
	Rscript -e "rmarkdown::render('docs/05_generate_report.Rmd')"


clean :
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results/*
	rm -rf docs/05_generate_report.html

	
