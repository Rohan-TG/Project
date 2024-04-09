from Data_handling import ENDF6
import os
import pandas as pd





ENDF6_path = "C:/Users/TG300/Project/Data_handling/ENDF6_reading"

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])