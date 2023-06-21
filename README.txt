Linearly interpolated data file - interpolated_n2_1_xs_fund_feateng.csv (36 MB) : https://drive.google.com/file/d/1AUeGnzfO5BqdUDftE5lYGQVHNH0M8mkT/view?usp=share_link

Original data - 1_xs_fund_feateng.csv (214 MB): https://drive.google.com/file/d/1qbpQh2tjkvumejsoT7FlxYMMU6968hn6/view?usp=share_link
-------------------------------------------------------------------------------------------------------------------------------------------------------------------


XGBoost_interpolation.py is the script used for training and testing, with training done on the interpolated data, and testing done on the original data.
XGBoost_int_opt.py is used for hyperparameter optimisation, using the linearly interpolated data as training data, and the original uninterpolated data as validation data.

Requirements for running the script:
Numpy 1.24.3
pandas 2.0.1
matplotlib 3.7.1
periodictable 1.5.2
