# Rohan - Read JSON example

# This is an example code that reads cross section data from a JSON file and plots it

# NOTE: CHANGE LINE 40 TO SUIT YOUR DIRECTORY/FILE LAYOUT


import sys
import json
import matplotlib.pyplot as plt

# comment the next line out if not installed
import periodictable     #  https://periodictable.readthedocs.io/en/latest/api/core.html#periodictable.core.Element


# this is the start of the json file being read in. It only contains cross sections for MT = 16, for this example, else the file will be huge if I include all reactions

#{
#"Nuclear_Info":{
#"Source":"/home/lee/code/FISPACT/NEACD1/nuclear_data/ENDFB80data/endfb80-n/gxs-709/",
#"MTs":[
     #16
#],
#"Nuclear_Data":
#[
   #{
      #"A": 36,
      #"Z": 17,
      #"MASS": 3.565932E+01,
      #"LISO": 0,
      #"LIS": 0,
      #"ELIS": 0.000000E+00,
      #"REACTION": [
         #{
            #"MT": 16,
            #"QI": -8.579000E+06,
            #"QM": -8.579000E+06,
            #"Energy": [
                     #8.709600E+06,


XS_data_path = r'C:\Users\TG300\Desktop\Research Project - AWE\xs.json'                #   CHANGE THIS PATH TO SUIT WHERE YOUR JSON FILE IS

def Get_XS_Data(Z,A,M, MT):
    
    with open(XS_data_path) as f:                   
        all_data = json.load(f)


    #print("len all_data = ", len(all_data))
    
    Reaction_Data = all_data["Nuclear_Info"]["Nuclear_Data"]
    
    Data_Found = 0
    xs_erg = []
    xs_xs = []
    
    for j in range(0,len(Reaction_Data)):
        xs_z = int(Reaction_Data[j]["Z"])
        xs_a = int(Reaction_Data[j]["A"])
        xs_m = int(Reaction_Data[j]["LISO"])
        
        #print(">>>>> ", xs_z, "-",xs_a,"-",xs_m,"      ",Z, "-",A,"-",M)
        
        if( (xs_z == Z) and (xs_a == A) and (xs_m == M)):
            print("ZAM match")
            Data_Found = 1
            xs_reaction = Reaction_Data[j]["REACTION"]

            #print("Z = ", xs_z)
            #print("A = ", xs_a)
            #print("REACTIONS = ", len(xs_reaction))
            
            for i in range(0,len(xs_reaction)):
                xs_reaction_i = xs_reaction[i]
                xs_mt = xs_reaction[i]["MT"]
                
                if(xs_mt == MT):
                    print("ZAM-MT match")
                    Data_Found = 2
                    xs_erg = xs_reaction[i]["Energy"]
                    xs_xs = xs_reaction[i]["XS"]
                    
                    break
            if(Data_Found == 2):
                break
            
    return xs_erg,xs_xs,Data_Found




if __name__ == "__main__":
    
    Z = 26       # proton number
    A = 56       # mass number
    LISO = 0     # LISO is isomer state number
    MT = 16      # MT is reaction identifier. 16 = (n,2n)
    
    xs_energy, xs_values, found = Get_XS_Data(26,56,0,16)
    print(len(xs_energy), len(xs_values))
    
    if(found):                                  # retruns found = true if cross section has been found in json file
        plt.plot(xs_energy,xs_values)
        plt.grid()
        if "periodictable" in sys.modules:
            plt.title("Cross section of " + str(periodictable.elements[Z]) + "-" + str(A) + "-" + str(LISO) + " for MT = " + str(MT), fontsize=18)
        else:
            print("INFO:: \'periodictable\' library not loaded. Defaulting to proton number instead of element symbol")
            plt.title("Cross section of " + str(Z) + "-" + str(A) + "-" + str(LISO) + " for MT = " + str(MT), fontsize=18)
            
        plt.xlabel("Energy (MeV)", fontsize=18)
        plt.ylabel("XS (Barns)", fontsize=18)
        plt.xlim([0, 30E6])
        plt.show()
    else:
        print("ERROR :: Cross section of " + str(Z) + "-" + str(A) + "-" + str(LISO) + " for MT = " + str(MT))
