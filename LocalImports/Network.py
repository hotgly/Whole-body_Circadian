# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2017

@author: john abel
Module set up to perform analysis for Yongli Shan.
"""


from concurrent import futures
import itertools
import numpy  as np
from scipy import sparse, signal, optimize
from scipy.interpolate import UnivariateSpline
import statsmodels.tsa.filters as filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minepy as mp
import numpy.random as random
import networkx as nx
import fastdtw
import pywt
import collections

from . import PlotOptions as plo
from . import DecayingSinusoid as ds
from . import Bioluminescence as bl

if pywt.__version__ != '0.3.0':
    # 0.5.2 is used for firing patterns paper
    print("Some features involving wavelet analysis may not work if PyWavelets version is not 0.3.0")


class Network(object):
    """
    Class to analyze time series data and infer network connections within data
    from the suprachiasmatic nucleus.
    """

    def __init__(self, data, sph=None, t=None, loc=None, green_red=None,
                    name=None):
        """
        Parameters
        ----------
        data : np.ndarray
            should contain multiple time series for analysis. column is a
            time series for a single node, horizontal is contant time different
            nodes.
        sph : float
            samples per hour
        t   : np.ndarray
            time points of the trajectory. provide only one.
        loc : np.ndarray
            (x,y) coordinates for each neuron
        greenred : np.ndarray
            this tells whether neurons are green or red in the image. 0s indicate
            green, 1s represent red
        name : str
            just a name for the network

        """
        self.sph = {'raw' : sph}
        if t is not None:
            if len(t) != len(data[:,0]):
                print(('ERROR: '
                    +'Time series data and time array do not match in size.'))
            else:
                self.t = {'raw': t}
                self.sph = {'raw': 1/(t[1]-t[0])}

        if (t is None and sph is not None):
            # defining t from sph
            t = np.arange(0,1/sph*len(data),1/sph).astype(float)
            self.t = {'raw': t}


        self.data = {'raw' : data}
        self.nodecount = len(data[0,:])

        if loc is not None:
            # is it a dict?
            if type(loc) is dict:
                self.locations = loc
            # or is it a numpy array
            elif type(loc) is np.ndarray:
                loc_dict = {}
                for i in range(len(loc)):
                    loc_dict[i] = (loc[i,-2], loc[i,-1])
                self.locations = loc_dict
            else:
                print("Location data type not recognized.")

        if green_red is not None:
            self.green_red = green_red
            self.division = np.argmax(green_red)
        if name is not None:
            self.name = name


    def interpret_nans(self):
        """
        Interpolates missing values to replace NaNs. Also removes
        leading/trailing rows where any piece of data is a NaN.
        """
        # make a copy of the data
        data = np.copy(self.data['raw'])
        # first, interpolate the intermediate data
        h1, h2 = nan_helper(data)


        # interpolate the intermediate points
        for i in range(len(self.data['raw'][0,:])):
            try:
                init_goodval = np.min(np.where(~h1[:,i])[0])
                end_goodval  = np.max(np.where(~h1[:,i])[0])
                # only interpolate where we have good values
                y = data[init_goodval:end_goodval, i]
                # find nans
                nans, x= nan_helper(y)
                # interpolate
                y[nans]= np.interp(x(nans), x(~nans), y[~nans])
                # replace old values with new
                data[init_goodval:end_goodval, i] = y
            except ValueError:
                # if there are no nans
                pass

        h1, h2 = nan_helper(data)

        # NOW trim the data
        # find the first row where nothing is a nan
        init_goodval = 0
        for i in range(len(h1)):
            if any(h1[i,:]==True):
                pass
            else:
                init_goodval = i
                break

        # find the last row where nothing is a nan
        end_goodval = 0
        for i in range(1,len(h1)):
            if any(h1[-i,:]==True):
                pass
            else:
                end_goodval = -i
                break

        #save the new data and times
        self.data['nan_restricted'] = \
                data[init_goodval:end_goodval,:]
        self.t['nan_restricted'] = self.t['raw'][init_goodval:end_goodval]
        self.sph['nan_restricted'] = self.sph['raw']

    def parallel_MIC(self, use_data='raw', num_cpus=10):
        """
        Parallel calculation of MIC for an array of trajectories.

        Parameters
        ----------
        use_data : str (default='raw')
            Key of the Network.data dictionary from which the data is pulled.
        num_cpus : int (default=10)
            Number of CPUs to allocate to the parallel process.
        """
        # pick which data we are using
        data = self.data[use_data]

        # get the index combinations
        inds = list(itertools.combinations_with_replacement(
                    list(range(self.nodecount)),2))

        data_inputs = []
        # set up the input file to the parallelization
        for ind in inds:
            # input looks like: index1, index2, data1, data2
            data_inputs.append([ind[0],ind[1], data[:,ind[0]],data[:,ind[1]]])

        with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            result = executor.map(single_MIC, data_inputs)

        # process the result object that is output
        mic = np.zeros([self.nodecount,self.nodecount])
        for r in result:
            mic[r[0],r[1]] =r[2]
            mic[r[1],r[0]] =r[2]

        np.fill_diagonal(mic,0)
        try:
            self.mic[use_data] = mic
        except AttributeError:
            self.mic = {use_data : mic}

    def full_DTW(self, use_data='raw'):
        """
        Calculation of DTW for an array of trajectories.

        Parameters
        ----------
        use_data : str (default='raw')
            Key of the Network.data dictionary from which the data is pulled.
        """
        # pick which data we are using
        data = self.data[use_data]

        # get the index combinations
        inds = list(itertools.combinations_with_replacement(
                    list(range(self.nodecount)),2))

        data_inputs = []
        # set up the input file to the parallelization
        for ind in inds:
            # input looks like: index1, index2, data1, data2
            data_inputs.append([ind[0],ind[1], data[:,ind[0]],data[:,ind[1]]])

        # PARALLELIZATION NOT WORKING
        #with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        #    result = executor.map(single_DTW, data_inputs, chunk_size=1000)
        result = []
        for inputi in data_inputs:
            result.append(single_DTW(inputi))

        # process the result object that is output
        dtw = np.zeros([self.nodecount,self.nodecount])
        for r in result:
            dtw[r[0],r[1]] =r[2]
            dtw[r[1],r[0]] =r[2]

        np.fill_diagonal(dtw,0)
        self.dtw = {use_data : dtw}

    def MIC(self, use_data='raw'):
        """
        Serial calculation of MIC for an array of trajectories.

        Parameters
        ----------
        use_data : str (default='raw')
            Key of the Network.data dictionary from which the data is pulled.
        """
        miner = mp.MINE(alpha=0.6, c=15, est="mic_approx")
        c1 = list(range(self.nodecount))
        c2 = list(range(self.nodecount))

        mic = np.zeros([self.nodecount,self.nodecount])
        for i in c1:
            x1 = self.data[use_data][:,i]
            for j in c2:
                if i<=j:
                    x2 = self.data[use_data][:,j]
                    miner.compute_score(x1,x2)
                    mic[i,j] = miner.mic()
                else:
                    mic[i,j] = mic[j,i]

        if hasattr(self,'mic'):
            self.mic[use_data] = mic
        else: self.mic = {use_data : mic}


    def create_adjacency(self, data='raw', method='def', thresh=0.90, adj=None):
        """
        Creates an adjacency matrix using one of several methods.

        Parameters
        ----------
        data : str (default='raw')
            Key of the Network.data dictionary from which the data is pulled.
        method : str (default='def')
            Choose the method for defining the adjacency matrix. 'mic' uses a
            previously calculated MIC confusion matrix, 'dtw' uses a dynamic
            time warping confusion matrix. In both cases, the resulting adj is
            named for the specified threshold.
        thresh : float \in [0,1] (default=0.90)
            Correlation threshold above which a connection exists. Used only for
            'mic' or 'dtw' methods.
        """

        if method=='mic':
            adj = 1*self.mic[data]>=thresh #floors values below thresh
            np.fill_diagonal(adj,0)
            if hasattr(self,'adj'):
                self.adj['%.2f' % (thresh)] = adj
            else: self.adj = {'%.2f' % (thresh) : adj}

        elif method=='dtw':
            adj = np.floor(-self.dtw[data]+thresh) #floors values below thresh
            np.fill_diagonal(adj,0)
            if hasattr(self,'adj'):
                self.adj['%.2f' % (thresh)] = adj
            else: self.adj = {'%.2f' % (thresh) : adj}
        #defining one
        else:
            if hasattr(self,'adj'):
                self.adj[method] = adj
            else: self.adj = {method : adj}

    def ls_periodogram(self, cells='all', data = 'raw',
                       period_low=None, period_high=None, res=None, norm=True):
        """
        Calculates the lomb-scargle normalized periodogram for
        bioluminescence of individual cells. creates and attaches array of
        frequencies, periodogram magnitudes, and significance.

        Parameters
        ----------
        cells : np.ndarray (default = 'all')
            Which cells to include in the graph, by index. Defaults to all.
        data : str (default='raw')
            Key of the Network.data dictionary from which the data is pulled.
        period_high, period_low : float (defaults=None)
            High or low period allowed. If unspecified, uses nyquist frequency
            to determine low and 64 for high.
        res : int (default = 10*(high-low))
            Resolution of periodogram.
        norm : bool (default=True)
            Normalize the pgram.

        Returns
        ----------
        Network.periodogram : dict
            keys: cells (the cells used), 'periods' (LS peak periods), 'cell_data'
            (the data used in calcuation of the periodogram)
        """

        if period_low is None:
            nyquist_freq = self.sph[data]/2
            period_low = 1/nyquist_freq
        if period_high is None:
            period_high = 64
        if res is None:
            res = (period_high-period_low)*10

        # select cell traces
        if cells is 'all':
            cells = np.arange(self.nodecount)
        lsdata = self.data[data][:,cells]

        pgrams = np.zeros([int(res),int(cells.size)])
        sigs = np.zeros([int(res),int(cells.size)])

        for i in range(len(cells)):
            cell_data = np.copy(lsdata[:,i])
            cell_times = self.t[data][~np.isnan(cell_data)]
            cell_data = cell_data[~np.isnan(cell_data)]
            cell_pers, cell_pgram, cell_sig = \
                                periodogram(cell_times, cell_data,
                                            period_low = period_low,
                                            period_high = period_high,
                                            res = res, norm = True)
            pgrams[:,i] = cell_pgram
            sigs[:,i] = cell_sig

        self.periodogram = {}
        self.periodogram['cells'] = cells
        self.periodogram['periods'] = cell_pers
        self.periodogram['cell_data'] = lsdata
        self.periodogram['pgrams'] = pgrams
        self.periodogram['sigs'] = sigs

    def ls_rhythmic_cells2(self, alpha = 0.05, circ_low=18, circ_high=30):
        """TEST FUNCTION determines which cells are rhythmic from the lomb-scargle
        periodogram, by comaparing significance to our p-value

        Parameters
        ----------
        alpha : float \in [0,1] (default=0.05)
            Significance threshold.
        circ_low, circ_high : floats (defaults=18,30)
            Low and high range for a signficant oscillation being circadian.
        """
        print("NOT forcing circadian period to have largest peak")
        # get info from the periodograms
        try:
            cells = self.periodogram['cells']
            periods = self.periodogram['periods']
            pgrams = self.periodogram['pgrams']
            cell_data = self.periodogram['cell_data']
        except:
            print("ERROR: ls_periodogram must be calculated before testing.")
            return

        period_inds = np.where(
                        np.logical_and(circ_low <= self.periodogram['periods'],
                                    self.periodogram['periods'] <= circ_high))[0]

        #set up empty lists of rhythmic cells and max power periods
        rhythmic_cells = np.zeros(len(cells))
        cell_periods = np.zeros(len(cells))
        zscores = np.zeros(len(cells))

        # loop through cells
        for i in range(len(cells)):
            # pull the periodogram
            pgrams_all = pgrams[:,i]
            pgrams_in_range = pgrams_all[period_inds].flatten()

            # find the critical z-score (= pgram score)
            z = -np.log(1-(1-alpha)**(1/len(cell_data[:,i])))
            zscores[i] = z
            # now test if there is a period in range
            if (pgrams_in_range > z).any():

                # enforces that highest peak must be in circadian range
                # find all peaks
                all_peak_locs = signal.argrelmax(pgrams_all, order = 14)
                all_peaks = pgrams_all[all_peak_locs]
                # find peaks in range
                range_peak_locs = signal.argrelmax(pgrams_in_range, order=2)
                range_peaks = pgrams_in_range[range_peak_locs]

                # if the max peak is the peak in range
                if np.max(all_peaks) == np.max(range_peaks):
                    # cell is rhythmic
                    rhythmic_cells[i] = 1
                    # cell period
                    cell_periods[i] = periods[period_inds[np.argmax(pgrams_in_range)]]
                else:
                    # the peak is significant, but another is more significant
                    rhythmic_cells[i] = 0
                    cell_periods[i] = np.nan
                    # cell is rhythmic
                    #rhythmic_cells[i] = 1
                    # cell period
                    #cell_periods[i] = periods[period_inds[np.argmax(pgrams_in_range)]]

            else:
                # yield a false period
                cell_periods[i] = np.nan

        #period_of_oscillatory_cells = np.argma
        self.periodogram['rhythmic_'+str(alpha)] = rhythmic_cells
        self.periodogram['zcrit_alpha'+str(alpha)] = z
        self.periodogram['cell_period_peaks'] = cell_periods
        self.rc = np.where(rhythmic_cells > 0.1)[0]
        self.nrc = np.where(rhythmic_cells < 0.1)[0]

    def networkx_graph(self,adj=None,cells='all'):
        """
        Creates a networkx graph object for further analysis. Adds connections
        from a given adj if desired, and can exclude cells if not 'all' are
        desired.

        Parameters
        ----------
        adj : np.ndarray (default = None)
            An adjacency matrix from which to construct graph connections.
        cells : np.ndarray (default = 'all')
            Which cells to include in the graph, by index. Defaults to all.
            Choosing 'green_red' will create two
        """

        # initialize the object
        G = nx.Graph()
        if cells=='all':
            cells=list(range(self.nodecount))

        G.add_nodes_from(cells)

        if adj is not None:
            connections = self.adj[adj]
            edge_list = np.vstack(np.where(connections!=0)).T
            G.add_edges_from(edge_list)

        self.nx_graph = G

    def networkx_plot(self,ax=None, invert_y=False, **kwargs):
        """
        Plots a networkx graph.

        Parameters
        ----------
        ax : plt.subplot (default=None)
            The subplot where to plot.
        invert_y : bool (default=False)
            Inverts the y-axis if True.
        **kwargs for networkx.draw_networkx_edges
        """

        try:
            G = self.nx_graph
        except:
            print("ERROR: Networkx graph must be generated before it is plotted.")
            return

        if ax is None:
            ax = plt.subplot()


        #Turn off the ticks
        ax.tick_params(\
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off',
                left='off',
                right='off',
                labelleft='off')

        if invert_y==True:
            plt.gca().invert_yaxis()

        nx.draw_networkx_nodes(G,
                               pos=self.locations,
                               ax=ax, zorder=-6, **kwargs)
        nx.draw_networkx_edges(G,pos=self.locations,width=0.5,alpha=0.3,ax=ax,
            zorder=-5,**kwargs)

    def split_scn(self, xdivision, inverted=False):
        """
        Splits the SCN into left and right lobes, using xdivision to divide.
        Creates self.mic['valid'] (where only within-half connections are
        allowed) and self.lobe_id where -1 is left and 1 is right.

        Note that if the SCN is inverted, this will be flipped.
        """
        lr = np.zeros(self.nodecount)
        for node in range(self.nodecount):
            if self.locations[node][0]<=xdivision:
                lr[node] =-1
            else:
                lr[node] =1

        if inverted:
            # flip lr
            lr *= -1

        # tells us if connections are valid
        lobes1 = np.multiply(lr,np.ones([self.nodecount, self.nodecount]))
        lobes2 =np.transpose(lobes1)
        valid = np.multiply(lobes1,lobes2) > 0

        # save it
        if hasattr(self,'adj'):
            self.adj['valid'] = valid
        else: self.adj = {'valid' : valid}

    def split_classes_mic(self, mic_data='raw'):
        """
        Splits the MIC of the SCN into three classes, using redgreen to divide.
        Creates self.gir where -1 is left and 1 is right.

        Note that if the SCN is inverted, this will be flipped.
        """
        mic = self.mic[mic_data]
        reds = []
        greens = []
        inters = []
        for node1 in range(self.nodecount):
            for node2 in range(self.nodecount):
                # only do a triangle of the MIC matrix
                if node1>node2:
                    class1 = self.green_red[node1]
                    class2 = self.green_red[node2]
                    if class1 != class2:
                        inters.append(mic[node1,node2])
                    else:
                        if class1==0:
                            greens.append(mic[node1,node2])
                        else:
                            reds.append(mic[node1,node2])

        self.classes_mic = {'r': reds,
                            'g': greens,
                            'i': inters
                            }

    def cwt(self, datatype='raw', shortestperiod=18, longestperiod=30):
        """Uses the bioluminescence package to do a CWT analysis of the cells. Returns
        instantaneous periods and phases for each cell."""

        times = self.t[datatype]
        data = self.data[datatype]

        phase_data = [np.nan]*len(data[0])
        period_data = [np.nan]*len(data[0])
        times_data = [np.nan]*len(data[0])
        for idx, di in enumerate(data.T):
            if self.rhythmic[idx]==1:
                ti = np.copy(times[~np.isnan(di)])
                di = di[~np.isnan(di)]
                cwt = bl.continuous_wavelet_transform(ti, di, shortestperiod=shortestperiod,
                                    longestperiod=longestperiod)
                phase_data[idx]=cwt['phase']
                period_data[idx]=cwt['period']
                times_data[idx]=cwt['x']

        self.cwt_phase = phase_data
        self.cwt_period = period_data
        self.cwt_times = times_data

# component functions not explicitly in the class
def single_MIC(data_t1t2):
    """
    MIC calculation for a single pair of trajectories. Eliminates NaNs.
    """
    t1, t2, d1, d2 = data_t1t2
    d1nan = np.isnan(d1)
    d2nan = np.isnan(d2)
    d1 = d1[~d1nan & ~d2nan]
    d2 = d2[~d1nan & ~d2nan]
    miner = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    miner.compute_score(d1, d2)
    return t1,t2,miner.mic()

def single_DTW(data_t1t2):
    """
    DTW calculation for a single pair of trajectories
    """
    t1, t2, d1, d2 = data_t1t2
    dtw = fastdtw.fastdtw(d1,d2)[0]
    return t1,t2,dtw

def periodogram(x, y, period_low=2, period_high=35, res=200, norm=True):
    """ calculate the periodogram at the specified frequencies, return
    periods, pgram. if norm = True, normalized pgram is returned """

    periods = np.linspace(period_low, period_high, res)
    # periods = np.logspace(np.log10(period_low), np.log10(period_high),
    #                       res)
    freqs = 2*np.pi/periods
    try: pgram = signal.lombscargle(x, y, freqs)
    # Scipy bug, will be fixed in 1.5.0
    except ZeroDivisionError: pgram = signal.lombscargle(x+1, y, freqs)

    # significance (see press 1994 numerical recipes, p576)
    if norm == True:
        var = np.var(y)
        pgram_norm = pgram/var
        significance =  1-(1-np.exp(-pgram_norm))**len(x)
        return periods, pgram_norm, significance
    else:
        return periods, pgram


def estimate_period(x, y, period_low=1, period_high=100, res=200):
    """ Find the most likely period using a periodogram """
    periods, pgram = periodogram(x, y, period_low=period_low,
                                 period_high=period_high, res=res)
    return periods[pgram.argmax()]

def radius_phase(phases):
    """ returns radius, phase, number of cells"""

    if len(phases)==0:
        return 0, 0, 0

    m = (1/len(phases))*np.sum(np.exp(1j*phases))
    length = np.abs(m)
    ang = np.angle(m)

    return length, ang, len(phases)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def plot_polygon(txt_left, txt_right, ftype='txt', ax=plt.subplot(), zorder=-10):
    """Plots the polygon of the SCN from one of Yongli's txt files.
    The zorder argument sets the level the plot is at, often so that it's
    under everything else."""

    if ftype=='txt':
        poly_left_raw = np.loadtxt(txt_left, delimiter='\r\n', dtype='str')[2:]
        poly_right_raw = np.loadtxt(txt_right, delimiter='\r\n', dtype='str')[2:]
        poly_left = []
        poly_right= []
        for item in poly_left_raw:
            poly_left.append(item.split(' '))
        for item in poly_right_raw:
            poly_right.append(item.split(' '))
        poly_left = np.array(poly_left, dtype='float')
        poly_right = np.array(poly_right, dtype='float')
        ax.fill(poly_left[:,0], poly_left[:,1], color='gray', alpha=0.3, zorder=zorder)
        ax.fill(poly_right[:,0], poly_right[:,1], color='gray', alpha=0.3,  zorder=zorder)
    elif ftype=='csv':
        poly_left = np.genfromtxt(txt_left, delimiter=',')[1:,1:]
        poly_right = np.genfromtxt(txt_right, delimiter=',')[1:,1:]
        ax.fill(poly_left[:,0], poly_left[:,1], color='gray', alpha=0.3,  zorder=zorder)
        ax.fill(poly_right[:,0], poly_right[:,1], color='gray', alpha=0.3,  zorder=zorder)
