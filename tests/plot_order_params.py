import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import json

def main():
    fname = "data/pair_corrfuncsAl6400Mg1600.json"
    with open(fname,'r') as infile:
        data = json.load(infile)

    print ("here")
    T = data["temperature"]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cfs = data["cfs"]
    cf_dict = {key:[] for key in cfs[0].keys()}
    for i in range(len(T)):
        for key,value in cfs[i].iteritems():
            cf_dict[key].append( value )

    sort_indx = np.argsort(T)
    T = np.sort(T)
    for key in cf_dict.keys():
        cf_dict[key] = [cf_dict[key][indx] for indx in sort_indx]

    for key,value in cf_dict.iteritems():
        value = np.array(value)
        ax.plot( T, value, "-o", label="{}".format(key))

    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "Correlation function" )
    ax.legend( loc="best", frameon=False )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
