import numpy as np
import json

def main():
    from matplotlib import pyplot as plt
    fname = "data/pair_corrfuncs_tempdependent_gsAl7200Mg800.json"
    with open(fname,'r') as infile:
        data = json.load(infile)

    print ("here")
    T = data["temperature"]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cfs = data["cfs"]
    cf_dict = {key:[] for key in cfs[0].keys()}
    for i in range(len(T)):
        for key,value in cfs[i].items():
            cf_dict[key].append( value )

    #del T[6]
    sort_indx = np.argsort(T)
    T = [T[indx] for indx in sort_indx]
    for key in cf_dict.keys():
        cf_dict[key] = [cf_dict[key][indx] for indx in sort_indx]

    for key,value in cf_dict.items():
        value = np.array(value)
        ax.plot( T, value, "-o", label="{}".format(key))

    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "Correlation function" )
    ax.legend( loc="best", frameon=False )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    E = data["energy"]
    Cv = data["heat_capacity"]
    E = [E[indx] for indx in sort_indx]
    Cv = [Cv[indx] for indx in sort_indx]
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot( T, E, color="#1b9e77",marker="o")
    ax3 = ax2.twinx()
    ax3.plot( T, Cv, color="#d95f02",marker="o")
    ax2.set_xlabel( "Temeprature (K)" )
    ax2.set_ylabel( "Internal energy (eV)" )
    ax3.set_ylabel( "Heat capacity (eV/K)")
    plt.show()

if __name__ == "__main__":
    main()
