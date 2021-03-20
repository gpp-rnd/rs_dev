#!/bin/zsh

for nb in \
 00_processing_Aguirre2017.ipynb \
 00_processing_Chari2015.ipynb \
 00_processing_DeWeirdt2020.ipynb \
 00_processing_Doench2014.ipynb \
 00_processing_Doench2016.ipynb \
 00_processing_Kim2019.ipynb \
 00_processing_Koike-Yusa2014.ipynb \
 00_processing_Shalem2014.ipynb \
 00_processing_Wang2014.ipynb
do
  jupyter nbconvert --to notebook --execute --inplace notebooks/$nb
done
