#!/usr/bin/env bash
#python reader.py 0 ../data/raw_data/0822/cy_acc_relax_new.txt --output_dir ../data/clean_data/0822/cy_relax.json;
#echo "Finish cy_relax";
#python reader.py 0 ../data/raw_data/0822/gr_acc_relax_new.txt --output_dir ../data/clean_data/0822/gr_relax.json;
#echo "Finish gr_relax";
#python reader.py 0 ../data/raw_data/0822/swn_acc_relax_new.txt --output_dir ../data/clean_data/0822/swn_relax.json;
#echo "Finish swn_relax";
#python reader.py 0 ../data/raw_data/0822/wjh_acc_relax_new.txt --output_dir ../data/clean_data/0822/wjh_relax.json;
#echo "Finish wjh_relax";
#python reader.py 0 ../data/raw_data/0822/wrl_acc_relax_new.txt --output_dir ../data/clean_data/0822/wrl_relax.json;
#echo "Finish wrl_relax";
#python reader.py 1 ../data/raw_data/0822/cy_acc_tense.txt --csv ../data/raw_data/0822/log/cy_log.csv --output_dir ../data/clean_data/0822/cy_tense.json;
#echo "Finish cy_tense";
#python reader.py 1 ../data/raw_data/0822/swn_acc_tense.txt --csv ../data/raw_data/0822/log/swn_log.csv --output_dir ../data/clean_data/0822/swn_tense.json;
#echo "Finish swn_tense";
#python reader.py 1 ../data/raw_data/0822/wjh_acc_tense.txt --csv ../data/raw_data/0822/log/wjh_log.csv --output_dir ../data/clean_data/0822/wjh_tense.json;
#echo "Finish wjh_tense";
#python reader.py 2 ../data/raw_data/0822/cy_acc_extra.txt --csv ../data/raw_data/0822/log/cy_extra_log.csv --output_dir ../data/clean_data/0822/cy_extra.json;
#echo "Finish cy_extra";
#python reader.py 2 ../data/raw_data/0822/swn_acc_extra.txt --csv ../data/raw_data/0822/log/swn_extra_log.csv --output_dir ../data/clean_data/0822/swn_extra.json;
#echo "Finish swn_extra";
#python reader.py 2 ../data/raw_data/0822/wjh_acc_extra.txt --csv ../data/raw_data/0822/log/wjh_extra_log.csv --output_dir ../data/clean_data/0822/wjh_extra.json;
#echo "Finish wjh_extra";

#
#python fft.py 0 ../data/clean_data/0822/cy_relax.json --output_dir ../data/fft_data/0822/cy_relax
#echo "Finish cy_relax"
#python fft.py 0 ../data/clean_data/0822/gr_relax.json --output_dir ../data/fft_data/0822/gr_relax
#echo "Finish gr_relax"
#python fft.py 0 ../data/clean_data/0822/swn_relax.json --output_dir ../data/fft_data/0822/swn_relax
#echo "Finish swn_relax"
#python fft.py 0 ../data/clean_data/0822/wjh_relax.json --output_dir ../data/fft_data/0822/wjh_relax
#echo "Finish wjh_relax"
#python fft.py 0 ../data/clean_data/0822/wrl_relax.json --output_dir ../data/fft_data/0822/wrl_relax
#echo "Finish wrl_relax"
#python fft.py 1 ../data/clean_data/0822/cy_tense.json --output_dir ../data/fft_data/0822/cy_tense
#echo "Finish cy_tense"
#python fft.py 1 ../data/clean_data/0822/swn_tense.json --output_dir ../data/fft_data/0822/swn_tense
#echo "Finish swn_tense"
#python fft.py 1 ../data/clean_data/0822/wjh_tense.json --output_dir ../data/fft_data/0822/wjh_tense
#echo "Finish wjh_tense"
#python fft.py 2 ../data/clean_data/0822/cy_extra.json --output_dir ../data/fft_data/0822/cy_extra
#echo "Finish cy_extra"
#python fft.py 2 ../data/clean_data/0822/swn_extra.json --output_dir ../data/fft_data/0822/swn_extra
#echo "Finish swn_extra"
#python fft.py 2 ../data/clean_data/0822/wjh_extra.json --output_dir ../data/fft_data/0822/wjh_extra
#echo "Finish wjh_extra"

#python reader.py 0 ../data/raw_data/0816/pxy_relax_clipped.txt --output_dir ../data/clean_data/0816/pxy_relax.json;
#echo "Finish pxy_relax";
#python reader.py 0 ../data/raw_data/0816/zmy_relax_clipped.txt --output_dir ../data/clean_data/0816/zmy_relax.json;
#echo "Finish zmy_relax";
python fft.py 0 ../data/clean_data/0816/pxy_relax.json --output_dir ../data/fft_data/0822/pxy_relax
echo "Finish pxy_relax"
python fft.py 0 ../data/clean_data/0816/zmy_relax.json --output_dir ../data/fft_data/0822/zmy_relax
echo "Finish zmy_relax"
