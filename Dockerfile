# Docker file for license plate analysis 
# Author: Keanna, Cari, Furquan
# Date: 2020-02-05

# use rocker/tidyverse as the base image
FROM rocker/tidyverse

# Install required R packages
RUN Rscript -e "install.packages('testit')"
RUN Rscript -e "install.packages('docopt')"
RUN Rscript -e "install.packages('knitr')"

# install the anaconda distribution of python (includes: pandas, numpy, scikit-learn)
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda update -n base -c defaults conda

# install docopt python package
RUN /opt/conda/bin/conda install -y -c anaconda docopt

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"

# Install chromium
RUN apt-get update && apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip

# Install chromedriver
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /usr/bin/ \
    && rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# Install altair, selenium, and vega-datasets
RUN conda install -y -c conda-forge altair && conda install -y vega_datasets && conda install -y selenium

# Install imgkit and whtnltopdf
RUN pip install imgkit && \
  apt-get install -y wkhtmltopdf

# install subprocess.run python package
RUN pip install subprocess.run

# Launch bash shell
CMD ["/bin/bash"]