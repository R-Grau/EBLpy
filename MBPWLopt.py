import numpy as np
import os
import yaml

with open("MBPWLopt_config.yml", "r") as f:
    inp_config = yaml.safe_load(f)


#will have to put this into a configuration file when everything works
redshift = inp_config["redshift"]
# filepath = "~/Desktop/EBL_Gamma2016/EBL_samples/1ES1011_Feb2014/data/nominal/flute/Output_flute.root"
filepath = inp_config["filepath"]
n_breaks = 1
deltalogE = inp_config["deltalogE"]
EBLscale = inp_config["EBLscale"]
#find the optimal position of the middle node (change this depending on the source)
Efirst_arr = np.geomspace(inp_config["minEfirst"], inp_config["maxEfirst"], inp_config["n_steps"])  #start, stop, number of points
Chi2_arr = []
for i in Efirst_arr:
    Efirst = i
    Elast = 10**(deltalogE+np.log10(Efirst))
    os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE -b > temp.txt".format(filepath, redshift, n_breaks, Efirst, Elast, EBLscale))

    file = open("temp.txt").readlines()
    Chi2s = ""
    Chi2_temp = ""
    for line in file:
        if "Chi2" in line:
            Chi2_temp = line.split(": ")[-1].strip()
            Chi2s = Chi2_temp.split(" /")[0].strip()
    Chi2 = float(Chi2s)
    print("Chi2:", Chi2)
    Chi2_arr.append(Chi2)

Efirst = Efirst_arr[np.argmin(Chi2_arr)]
Elast = 10**(deltalogE + np.log10(Efirst))
#here we get a more fine tunning
print("Again but around the min value found before")
Efirst_arr = np.geomspace(Efirst-10, Efirst+10, 20)
Chi2_arr = []
for i in Efirst_arr:
    Efirst = i
    Elast = 10**(deltalogE+np.log10(Efirst))
    os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE -b > temp.txt".format(filepath, redshift, n_breaks, Efirst, Elast, EBLscale))

    file = open("temp.txt").readlines()
    Chi2s = ""
    Chi2_temp = ""
    for line in file:
        if "Chi2" in line:
            Chi2_temp = line.split(": ")[-1].strip()
            Chi2s = Chi2_temp.split(" /")[0].strip()
    Chi2 = float(Chi2s)
    print("Chi2:", Chi2)
    Chi2_arr.append(Chi2)


Efirst = Efirst_arr[np.argmin(Chi2_arr)]
print("Efirst:", Efirst)
Elast = 10**(deltalogE + np.log10(Efirst))

#now that we have the ideal position of the central node we will use 3 nodes and check for different deltalogE values that have the same center:
print("now with 3 nodes and different distances between Efirst and Elast")
Emid = np.sqrt(Efirst * Elast)
Chi2delt = []
for i in range(15):
    deltalogE = (i + 1) * 0.1
    Efirst = 10**(np.log10(Emid) - deltalogE/2)
    Elast = 10**(np.log10(Emid) + deltalogE/2)
    print("Efirst and Elast", Efirst, Elast)
    n_breaks = 3
    os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE -b > temp.txt".format(filepath, redshift, n_breaks, Efirst, Elast, EBLscale))

    file = open("temp.txt").readlines()
    Chi2s = ""
    Chi2_temp = ""
    for line in file:
        if "Chi2" in line:
            Chi2_temp = line.split(": ")[-1].strip()
            Chi2s = Chi2_temp.split(" /")[0].strip()
    Chi2 = float(Chi2s)
    print("Chi2:", Chi2)
    Chi2delt.append(Chi2)


deltalogE = (np.argmin(Chi2delt) + 1) * 0.1
Efirst = 10**(np.log10(Emid) - deltalogE/2)
Elast = 10**(np.log10(Emid) + deltalogE/2)
print("This is the data of the best fit for 3 knots:")
print("Efirst: ", Efirst)
print("Elast: ", Elast)
print("deltalogE: ", deltalogE)
print("Chi2: ", Chi2delt[np.argmin(Chi2delt)])

os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE -b > temp.txt".format(filepath, redshift, n_breaks, Efirst, Elast, EBLscale))

#fix Emin and Emax and change the number of nodes:
print("now with different nodes")
nodes = [1, 3, 5, 9, 17]
Chi2_fin = []
for i in nodes:
    n_breaks = i
    os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE -b > temp.txt".format(filepath, redshift, n_breaks, Efirst, Elast, EBLscale))

    file = open("temp.txt").readlines()
    Chi2s = ""
    Chi2_temp = ""
    for line in file:
        if "Chi2" in line:
            Chi2_temp = line.split(": ")[-1].strip()
            Chi2s = Chi2_temp.split(" /")[0].strip()
    Chi2 = float(Chi2s)
    print("n_breaks: ", i)
    print("Chi2: ", Chi2)
    Chi2_fin.append(Chi2)

for i in range(5):
    print("# nodes: ", nodes[i], " Chi2: ", Chi2_fin[i])

print("Best fit parameters: \nEfirst = ", Efirst, "\nElast", Elast)
os.system("fold --inputfile={} --function=MBPWL --redshift={} --MBPWLn_breaks={} --Efirst={} --Elast={} --EBLscale={}. --logdFdE > temp.txt".format(filepath, redshift, 3, Efirst, Elast, EBLscale))