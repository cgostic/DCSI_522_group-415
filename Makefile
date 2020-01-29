# License plate analysis pipe
# author: Keanna
# date: 2020-01-29

# run entire analysis from scratch
all : docs/final_report.md docs/final_report.html

# load in data 
data/accepted_plates.csv data/rejected_plates.csv : scripts/01_download_data.R 
  Rscript scripts/01_download_data.R "data/" "accepted_plates.csv" "https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv" "rejected_plates.csv" "https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv"

# data pre-processing (e.g., undersample and split into train, validate & test)

python scripts/02_data_preprocessing.py --file_path="data/" --accepted_plates_csv="accepted_plates.csv" --rejected_plates_csv="rejected_plates.csv" --reduced_plate_csv= "full_vanity_plate_data.csv"" --X_test_csv="X_test.csv" --X_train_csv="X_train.csv" --X_validate_csv="X_validate.csv" --y_test_csv="y_test.csv" --y_train_csv="y_train.csv" --y_validate_csv="y_validate.csv"


results/isles.dat : data/isles.txt src/wordcount.py
	python src/wordcount.py data/isles.txt results/isles.dat
	
results/abyss.dat : data/abyss.txt src/wordcount.py
	python src/wordcount.py data/abyss.txt results/abyss.dat


# plot 
docs/img/best_predictors.png : results/isles.dat src/plotcount.py
	python src/plotcount.py results/isles.dat results/figure/isles.png

results/figure/abyss.png : results/abyss.dat src/plotcount.py
	python src/plotcount.py results/abyss.dat results/figure/abyss.png


# render final report
doc/final_report.md doc/final_report.html : doc/final_report.Rmd results/figure/isles.png results/figure/abyss.png results/figure/sierra.png results/figure/last.png
	Rscript -e "rmarkdown::render('doc/final_report.Rmd')"


clean :
	rm -f docs/*.png
	rm -f docs/final_report.md final_report.html
	# remove train/validate data csv's
	
	#should we add results folder
	# do we need both png and html of figures