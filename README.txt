Linearly interpolated data file - interpolated_n2_1_xs_fund_feateng.csv (36 MB) : https://drive.google.com/file/d/1AUeGnzfO5BqdUDftE5lYGQVHNH0M8mkT/view?usp=share_link

Original data - 1_xs_fund_feateng.csv (15 MB): https://drive.google.com/file/d/1a1VaVLPuFGAqF8EOutKTETSM5FN4KN37/view?usp=drive_link
-----------------------------------------------------------------------------------------------------------------------------
Linearly interpolated 0+ data (42.6 MB): https://drive.google.com/file/d/1PqMRNOkVa4ejFMlQ0dFmiLB5pjaaZPfX/view?usp=drive_link

Original data with 0+ (20.4 MB): https://drive.google.com/file/d/1QFyBTPAOsGAoTnXtgt0tON9ExmwCASmx/view?usp=drive_link

XGBoost_interpolation.py is the script used for training and testing, with training done on the interpolated data, and testing done on the original data.
XGBoost_int_opt.py is used for hyperparameter optimisation, using the linearly interpolated data as training data, and the original uninterpolated data as validation data.

Requirements for running the script:
Numpy 1.24.3
pandas 2.0.1
matplotlib 3.7.1
periodictable 1.5.2
