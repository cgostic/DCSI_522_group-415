
#
"This script downloads 2 .csv files with the same columns,  
combines them into a single dataframe, and exports this dataframe 
as a .csv to the file_path provided. This script takes file_name, 
url1, url2, and file_path as arguments.

Usage: scripts/01_download_data.R --file_path=<file_path> --filename_1=<filename_1> --url1=<url1> --filename_2=<filename_2> --url2=<url2>
" -> doc

# file_path = './data/raw'
# file1_name = 'accepted_plates.csv'
# file2_name = 'rejected_plates.csv'
# url1 = "https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv"
# url2 = "https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv"

library(tidyverse)
library(docopt)
library(testit)

opt <- docopt(doc)

main <- function(file_path, filename_1, url1, filename_2, url2){
    # open csvs from urls
    csv1 <- read.csv(url(url1))
    csv2 <- read.csv(url(url2))
    
    # Test that csvs are not empty
    testit::assert("CSV 1 has observations", dim(csv1)[1] != 0)
    testit::assert("CSV 2 is not empty", dim(csv2)[1] != 0)
    
    # write csv files to filepath
    write.csv(csv1,paste0(file_path,filename_1))
    write.csv(csv2, paste0(file_path,filename_2))
}

main(opt$file_path, opt$filename_1, opt$url1, opt$filename_2, opt$url2)


