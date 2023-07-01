Linearly interpolated data file - interpolated_n2_1_xs_fund_feateng.csv (36 MB) : https://drive.google.com/file/d/1AUeGnzfO5BqdUDftE5lYGQVHNH0M8mkT/view?usp=share_link

Original data - 1_xs_fund_feateng.csv (15 MB): https://drive.google.com/file/d/1a1VaVLPuFGAqF8EOutKTETSM5FN4KN37/view?usp=drive_link
-----------------------------------------------------------------------------------------------------------------------------
Linearly interpolated 0+ data (42.6 MB): https://drive.google.com/file/d/1PqMRNOkVa4ejFMlQ0dFmiLB5pjaaZPfX/view?usp=drive_link

Original data with 0+ (21.1 MB): https://drive.google.com/file/d/1huTXxWFD6Vj-d5p5YzQChUqSdV0Z_kAO/view?usp=drive_link

XGBoost_zeroed.py is the main script in use for training and testing.
Benchmarker_zeroed.py evaluates a given model's performance over all the data using manual k-fold cross validation.

Requirements for running the script:
Numpy 1.24.3
pandas 2.0.1
matplotlib 3.7.1
periodictable 1.5.2
