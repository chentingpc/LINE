#!/bin/sh

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate

data_folder=data/
result_folder=result/

#./reconstruct -train $data_folder/net_youtube.txt -output $data_folder/net_youtube_dense.txt -depth 2 -k-max 1000
./line -train $data_folder/net_youtube_dense.txt -output $result_folder/vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 6
./line -train $data_folder/net_youtube_dense.txt -output $result_folder/vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 6
./normalize -input $result_folder/vec_1st_wo_norm.txt -output $result_folder/vec_1st.txt -binary 1
./normalize -input $result_folder/vec_2nd_wo_norm.txt -output $result_folder/vec_2nd.txt -binary 1
./concatenate -input1 $result_folder/vec_1st.txt -input2 $result_folder/vec_2nd.txt -output $result_folder/vec_all.txt -binary 1

