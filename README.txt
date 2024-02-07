Any questions/concerns please email rnt26@cam.ac.uk

CURRENTLY IN USE - 1 < A < 255
ENDF/B-VIII data with features (19.2 MB): https://drive.google.com/file/d/1QNpYC0647buCRA3PrY3MGJ0iRAaXFbWb/view?usp=drive_link
ENDF/B-VIII data with features and uncertainties (Mk. 1) (22.6 MB): https://drive.google.com/file/d/1Mjt9IQStzmSAP075NftQ4-4oGgbzFWJQ/view?usp=drive_link

JENDL-5 with features (34 MB): https://drive.google.com/file/d/1ekuekIWEKcv3qA9rhaVBDDewIMf3ETBl/view?usp=drive_link
CENDL-3.2 with features (17.3 MB): https://drive.google.com/file/d/1dgAFojVzRxSpJWNf2ojF3SYuT2TTQSSW/view?usp=drive_link
TENDL-2021 with features (115.1 MB): https://drive.google.com/file/d/1uIkaZcdnbdpIfWZkHK7G-evDYmvwUT2C/view?usp=share_link
JEFF-3.3 with features (34.7 MB): https://drive.google.com/file/d/1qohTLi1tDvQXWqPALhyEK-xRJsfJztjq/view?usp=drive_link


Level Density Parameters: https://drive.google.com/file/d/1X_g66YiBr2ifDbQag7Gk3GLj4TzShGoP/view?usp=drive_link

Requirements for running the scripts:
Numpy 1.24.3
pandas 2.0.1
matplotlib 3.7.1
periodictable 1.5.2

Note that hyperparameters and data splits are not always up-to-date, as I continuously trial different combinations to optimise results.

Script shortlist:

To make uncertainty plots for central mass nuclides, use ReLU_uncertainty_producer.py. Use Comparator.py to both produce predictions using the rectifier method, and to do feature importance analysis. Benchmarking (cross validation) is mainly done using Multi_run_benchmarker.py.

"matrix_functions.py" contains a variety of useful functions employed throughout the project.
