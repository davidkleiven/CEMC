import numpy as np
from scipy.stats import linregress
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

lnf = [1.3550,0.6775,0.3387,0.1693,0.0846,0.04930001,0.02110001,0.00529296875,0.002646484375,0.0013232421875,0.00066162109375,0.000330810546875,0.000165405273437,
8.27026367187e-05,4.13513183594e-05,2.06756591797e-05]
N = 10

def var_scaling():
    base_name = "data/HistFluct/histogram"

    flucts = []
    for mod_f in lnf:
        avg_std = 0.0
        for i in range(N):
            if ( int(mod_f*10000) == 0 ):
                fname = base_name+"%d_%d.txt"%(mod_f*1E10,i)
            else:
                fname = base_name+"%d_%d.txt"%(mod_f*10000,i)
            hist = np.loadtxt(fname)
            #hist = hist/np.sum(hist)
            std = np.max(hist)-np.min(hist)
            avg_std += std
        flucts.append(avg_std/N)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(lnf,flucts)
    ax.set_xscale("log")
    plt.show()

def plot_ex_hist():
    data = np.loadtxt( "data/examplehist.txt" )
    fig = plt.figure()
    x = np.arange(len(data))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,data,ls="steps")
    #ax.fill_between(x,0,data)
    ax.set_xlabel( "Bin number" )
    ax.set_ylabel( "Number of visists" )
    mean = np.mean(data)
    std = np.std(data)
    ax.axhline(mean, lw=2,color="#e41a1c")
    ax.axhline(mean-std,ls="--",color="#e41a1c")
    ax.axhline(mean+std,ls="--",color="#e41a1c")
    plt.show()

def conv_time_scaling():
    lnf = [2.71,1.3550,0.6775,0.3387,0.1693,0.0846,0.04930001,0.02110001,0.0105859,0.00529296875,0.002646484375,0.0013232421875,0.00066162109375,0.000330810546875,0.000165405273437,
    8.27026367187e-05,4.13513183594e-05,2.06756591797e-05,1.03378e-05]
    fname = "data/convergence_times.csv"
    data = np.loadtxt(fname,delimiter=",")
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)/np.sqrt(data.shape[0])

    start = 8
    stop = -1
    slope, interscept, rvalue, pvalue,stderr = linregress(np.log(lnf[start:stop]), np.log(mean[start:stop]) )
    print ("Slope: {}".format(slope))
    x = np.linspace(0.5*lnf[stop], 2*lnf[start],100 )
    y = np.exp(interscept)*x**(slope)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar( lnf, mean, yerr=std, capsize=2, fmt="o" )
    ax.plot(x,y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel( "Log. modification factor")
    ax.set_ylabel( "Time to reach flat histogram" )

    mxtime = np.max(data,axis=0)
    slope, interscept, rvalue, pvalue,stderr = linregress(np.log(lnf[start:stop]), np.log(mxtime[start:stop]) )
    print ("Maximum times slope: {}".format(slope))
    est_max_time = np.exp(interscept)*x**slope
    figmax = plt.figure()
    axmax = figmax.add_subplot(1,1,1)
    axmax.plot(lnf,mxtime,"o")
    axmax.plot(x,est_max_time)
    axmax.set_xscale("log")
    axmax.set_yscale("log")
    axmax.set_xlabel( "Log. modification factor" )
    axmax.set_ylabel( "Max. time to reach flat histogram" )
    corr_plot(data)
    plt.show()

def corr_plot( data ):
    """
    Create a figure of the correlation
    """
    shifted = np.roll(data,1,axis=1)
    data = data[:,1:]
    shifted = shifted[:,1:]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( shifted, data, 'o', color="#8da0cb", mfc="none" )
    x = np.linspace( data.min(), data.max(), 100 )
    y = np.linspace( shifted.min(), shifted.max(), 100 )
    ax.plot(y,x,color="#ff7f00")
    ax.set_xlabel( "Time to convergence iteration \$N\$" )
    ax.set_ylabel( "Time to convergence iteration \$N+1\$")


def main():
    conv_time_scaling()
    #plot_ex_hist()

if __name__ == "__main__":
    main()
