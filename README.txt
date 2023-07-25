CURRENTLY IN USE - 1 < A < 255
ENDF/B-VIII data with features (19.2 MB): https://drive.google.com/file/d/1jG6yP5L9XUsA3KXYVl3jfXoITtOxTJYb/view?usp=drive_link
JENDL5 with features (39.3 MB): https://drive.google.com/file/d/1gx1tGePpG61Y7ykC73r8uO-3NaPNg-lq/view?usp=drive_link
CENDL3.2 with features (16.8 MB): https://drive.google.com/file/d/1hXg8GL-GLiGKVD2M8cJebGo7UPHoJajU/view?usp=drive_link
TENDL21 with features (154.1 MB): https://drive.google.com/file/d/1V-qARaUFJ3UG-IMjvzxLzNFH306A5OfP/view?usp=drive_link
JEFF3.3 with features (33.7 MB): https://drive.google.com/file/d/1eHnWfEl0uAWXg8rA9wbisTZf3hv6VfhV/view?usp=drive_link

ReLU_ENDF_comparator.py
 is the main script in use for training and testing.
Benchmarker_zeroed.py evaluates a given model's performance over all the data using manual k-fold cross validation.

Requirements for running the script:
Numpy 1.24.3
pandas 2.0.1
matplotlib 3.7.1
periodictable 1.5.2
