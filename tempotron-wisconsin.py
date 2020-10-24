# coding=utf-8
###################################################################################
## File Name: tempotron-wisconsin.py
## Description: functions and classes for tempotron on wisconsin
## Reference: https://github.com/dieuwkehupkes/Tempotron
###################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def frequency_incode(num, num_min , num_max , frequency_min,frequency_max):
    '''
    encode the data into frequency domain
    :param num: the data
    :param num_min: minimum of all the data
    :param num_max: maximum of all the data
    :param frequency_min: target minimum frequency, unit: Hz
    :param frequency_max: target maximum frequency, unit: Hz
    :return: frequency
    '''
    return frequency_min + (num-num_min)/(num_max - num_min)*(frequency_max - frequency_min)

def generate_pulse_sequence_rate(time_window,freqency):
    '''
    generate time sequence with the target frequency in a certain time_window
    :param time_window: time window, unit:ms
    :param freqency: frequency
    :return: generated time sequence, unit:ms
    '''
    pulse_sequence = []
    pulse_timing = time_window/(freqency+1)
    while(pulse_timing < time_window):
        pulse_sequence.append(int(pulse_timing))
        pulse_timing += time_window/(freqency+1)
    return pulse_sequence

def convert_to_pulse_rate(inputarray,label,samplelength,frequency_min,frequency_max,num_min, num_max ,time_window):
    '''
    encode the data into a time sequence with corresponding frequency
    :param inputarray: input data
    :param label: label
    :param samplelength: number of variables in a sample
    :param frequency_min: target minimum frequency, unit: Hz
    :param frequency_max: target maximum frequency, unit: Hz
    :param num_min: minimum of all the data
    :param num_max: maximum of all the data
    :param time_window: time window, unit:ms
    :return: pulse sequences: [[[sequence],[sequence],...,[sequence], target],...]
    '''
    pulse_sequences = []
    for i in range(len(inputarray)):
        output_pulse = []
        for j in range(samplelength):
            frequency = frequency_incode(inputarray[i][j], num_min, num_max, frequency_min,
                                         frequency_max)
            generated_pulse_sequence = generate_pulse_sequence_rate(time_window, frequency)
            output_pulse.append(generated_pulse_sequence)
        pulse_sequences.append((output_pulse,(label[i]==4)))
    return pulse_sequences

column_names = ['Sample code number', 'Clump Thickness',
                'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

def load_data_from_wisconsin():
    '''
    load data form wisconsin
    :return: x_train,x_test,y_train,y_test
    '''
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin//breast-cancer-wisconsin.data',
        names=column_names)
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                        data[column_names[10]], test_size=0.5,
                                                        random_state=1234)
    x_train = np.array(x_train).astype(int)
    x_test = np.array(x_test).astype(int)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)
    return x_train,x_test,y_train,y_test


### Tempotron Author: Dieuwke Hupkes @ https://github.com/dieuwkehupkes/Tempotron
class Tempotron:
    """
    A class representing a tempotron, as described in
    Gutig & Sompolinsky (2006).
    The (subthreshold) membrane voltage of the tempotron
    is a weighted sum from all incoming spikes and the
    resting potential of the neuron. The contribution of
    each spike decays exponentiall with time, the speed of
    this decay is determined by two parameters tau and tau_s,
    denoting the decay time constants of membrane integration
    and synaptic currents, respectively.
    """

    def __init__(self, V_rest, tau, tau_s, synaptic_efficacies, threshold=1.0):
        # set parameters as attributes
        self.V_rest = V_rest
        self.tau = float(tau)
        self.tau_s = float(tau_s)
        self.log_tts = np.log(self.tau / self.tau_s)
        self.threshold = threshold
        self.efficacies = synaptic_efficacies  ## weight
        self.t_spi = 10  # spike integration time, compute this with formula

        # compute normalisation factor V_0
        self.V_norm = self.compute_norm_factor(tau, tau_s)

    def compute_norm_factor(self, tau, tau_s):
        """
        Compute and return the normalisation factor:

        V_0 = (tau * tau_s * log(tau/tau_s)) / (tau - tau_s)

        That normalises the function:

        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)

        Such that it amplitude is 1 and the unitary PSP
        amplitudes are given by the synaptic efficacies.
        """
        tmax = (tau * tau_s * np.log(tau / tau_s)) / (tau - tau_s)
        v_max = self.K(1, tmax, 0)
        V_0 = 1 / v_max
        return V_0

    def K(self, V_0, t, t_i):
        """
        Compute the function

        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)
        """
        if t < t_i:
            value = 0
        else:
            value = V_0 * (np.exp(-(t - t_i) / self.tau) - np.exp(-(t - t_i) / self.tau_s))
        return value

    def compute_membrane_potential(self, t, spike_times):
        """
        Compute the membrane potential of the neuron given
        by the function:

        V(t) = sum_i w_i sum_{t_i} K(t-t_i) + V_rest

        Where w_i denote the synaptic efficacies and t_i denote
        ith afferent.

        :param spike_times: an array with at position i the spike times of
                            the ith afferent
        :type spike_times: numpy.ndarray
        """
        # create an array with the contributions of the
        # spikes for each synaps
        spike_contribs = self.compute_spike_contributions(t, spike_times)

        # multiply with the synaptic efficacies
        total_incoming = spike_contribs * self.efficacies

        # add sum and add V_rest to get membrane potential
        V = total_incoming.sum() + self.V_rest

        return V

    def compute_derivative(self, t, spike_times):
        """
        Compute the derivative of the membrane potential
        of the neuron at time t.
        This derivative is given by:

        V'(t) = V_0 sum_i w_i sum_{t_n} (exp(-(t-t_n)/tau_s)/tau_s - exp(-(t-t_n)/tau)/tau)

        for t_n < t
        """
        # sort spikes in chronological order
        spikes_chron = [(time, synapse) for synapse in range(len(spike_times)) for time in spike_times[synapse]]
        spikes_chron.sort()

        # Make a list of spike times and their corresponding weights
        spikes = [(s[0], self.efficacies[s[1]]) for s in spikes_chron]

        # At time t we want to incorporate all the spikes for which
        # t_spike < t
        sum_tau = np.array([spike[1] * np.exp(spike[0] / self.tau) for spike in spikes if spike[0] <= t]).sum()
        sum_tau_s = np.array([spike[1] * np.exp(spike[0] / self.tau_s) for spike in spikes if spike[0] <= t]).sum()

        factor_tau = np.exp(-t / self.tau) / self.tau
        factor_tau_s = np.exp(-t / self.tau_s) / self.tau_s

        deriv = self.V_norm * (factor_tau_s * sum_tau_s - factor_tau * sum_tau)

        return deriv

    def compute_spike_contributions(self, t, spike_times):
        """
        Compute the decayed contribution of the incoming spikes.
        """
        # nr of synapses
        N_synapse = len(spike_times)
        # loop over spike times to compute the contributions
        # of individual spikes
        spike_contribs = np.zeros(N_synapse)
        for neuron_pos in range(N_synapse):
            for spike_time in spike_times[neuron_pos]:
                # print self.K(self.V_rest, t, spike_time)
                spike_contribs[neuron_pos] += self.K(self.V_norm, t, spike_time)
        return spike_contribs

    def train(self, io_pairs, steps, learning_rate):
        """
        Train the tempotron on the given input-output pairs,
        applying gradient decscend to adapt the weights.

        :param steps: the maximum number of training steps
        :param io_pairs: a list with tuples of spike times and the
                         desired response on them
        :param learning_rate: the learning rate of the gradient descend
        """
        # Run until maximum number of steps is reached or
        # no weight updates occur anymore
        for i in range(steps):
            # go through io-pairs in random order
            for spike_times, target in np.random.permutation(io_pairs):
                #print(spike_times,target)
                #print("Target in train",target)
                self.adapt_weights(spike_times, target, learning_rate)
        return
    def test(self, input):
        """
        test the tempotron on the given input

        :param input: a list with tuples of spike times
        """
        # Run until maximum number of steps is reached or
        # no weight updates occur anymore
        try:
            tmax = self.compute_tmax(input)
        except:
            tmax = 0

        vmax = self.compute_membrane_potential(tmax, input)


        # if target output is correct, don't adapt weights
        if (vmax >= self.threshold):
            return True
        else:
            return False


    def get_membrane_potentials(self, t_start, t_end, spike_times, interval=0.1):
        """
        Get a list of membrane potentials from t_start to t_end
        as a result of the inputted spike times.
        """
        # create vectorised version of membrane potential function
        potential_vect = np.vectorize(self.compute_membrane_potential)
        # exclude spike times from being vectorised
        potential_vect.excluded.add(1)

        # compute membrane potentials
        t = np.arange(t_start, t_end, interval)
        membrane_potentials = potential_vect(t, spike_times)

        return t, membrane_potentials

    def get_derivatives(self, t_start, t_end, spike_times, interval=0.1):
        """
        Get a list of the derivative of the membrane potentials from
        t_start to t_end as a result of the inputted spike times.
        """
        # create a vectorised version of derivative function
        deriv_vect = np.vectorize(self.compute_derivative)
        # exclude spike times from being vectorised
        deriv_vect.excluded.add(1)

        # compute derivatives
        t = np.arange(t_start, t_end, interval)
        derivatives = deriv_vect(t, spike_times)

        return t, derivatives

    def plot_membrane_potential(self, t_start, t_end, spike_times, interval=0.1):
        """
        Plot the membrane potential between t_start and t_end as
        a result of the input spike times.
        :param t_start: start time in ms
        :param t_end: end time in ms
        :param interval: time step at which membrane potential is computed
        """
        # compute membrane_potential
        t, membrane_potentials = self.get_membrane_potentials(t_start, t_end, spike_times, interval)

        # format axes   #plot here for time vs V
        plt.xlabel('Time (ms)')
        plt.ylabel('V(t)')

        ymax = max(membrane_potentials.max() + 0.1, self.threshold + 0.1)
        ymin = min(membrane_potentials.min() - 0.1, -self.threshold - 0.1)
        plt.ylim(ymax=ymax, ymin=ymin)
        plt.axhline(y=self.threshold, linestyle='--', color='k')

        # plot membrane potential
        plot = plt.plot(t, membrane_potentials)
        # return plot
        # plt.show()

    def plot_potential_and_derivative(self, t_start, t_end, spike_times, interval=0.1):
        """
        Plot the membrane potential and the derivative of the membrane
        potential as a result of the input spikes between t_start and
        t_end.
        :param t_start: start time in ms
        :param t_end: end time in ms
        """
        # compute membrane potentials
        t, membrane_potentials = self.get_membrane_potentials(t_start, t_end, spike_times, interval)

        # compute derivatives
        t, derivatives = self.get_derivatives(t_start, t_end, spike_times, interval)

        # format axes
        plt.xlabel('Time(ms)')
        # ylabel???

        ymax = max(membrane_potentials.max() + 0.1, self.threshold + 0.1)
        ymin = min(membrane_potentials.min() - 0.1, -self.threshold - 0.1)
        plt.ylim(ymax=ymax, ymin=ymin)

        plt.axhline(y=self.threshold, linestyle='--', color='k')
        plt.axhline(y=0.0, linestyle='--', color='r')
        plt.axvline(x=16.5, color='b')

        # plot
        plt.plot(t, membrane_potentials, label='Membrane potential')
        plt.plot(t, derivatives, label='Derivative')
        plt.show()

    def compute_tmax(self, spike_times):
        """
        Compute the maximum mebrane potential of the tempotron as
        a result of the input spikes.
        The maxima of the function can be computed analytically, but as
        there are as many maxima and minima as their are number of spikes,
        we still need to sort through them to find the highest one.

        The maxima are given by:

        t = (log(tau/tau_s) + log(sum w_n exp(t_n/tau_s)) - log(sum w_n exp(t_n/tau)))*tau_s*tau/ (tau-tau_s)

        for n = 1, 2, ..., len(spike_times)

        The time at which the membrane potential is maximal is given by
        Check if the input spikes result produce the desired
        output. Return tmax. (maybe I should return something else)
        """

        # sort spikes in chronological order
        spikes_chron = [(time, synapse) for synapse in range(len(spike_times)) for time in spike_times[synapse]]
        spikes_chron.sort()

        # Make a list of spike times and their corresponding weights
        spikes = [(s[0], self.efficacies[s[1]]) for s in spikes_chron]
        times = np.array([spike[0] for spike in spikes])
        weights = np.array([spike[1] for spike in spikes])

        sum_tau = (weights * np.exp(times / self.tau)).cumsum()
        sum_tau_s = (weights * np.exp(times / self.tau_s)).cumsum()

        # when an inhibitive spike is generated when the membrane potential
        # is still growing, the derivative does not exist in the maximum
        # In such cases, thus when sum_tau/sum_tau_s is negative,
        # manually set tmax to the spike time of the second spike
        div = sum_tau_s / sum_tau
        boundary_cases = div < 0
        div[boundary_cases] = 10

        tmax_list = self.tau * self.tau_s * (self.log_tts + np.log(div)) / (self.tau - self.tau_s)
        tmax_list[boundary_cases] = times[boundary_cases]

        vmax_list = np.array([self.compute_membrane_potential(t, spike_times) for t in tmax_list])

        tmax = tmax_list[vmax_list.argmax()]

        return tmax

    def adapt_weights(self, spike_times, target, learning_rate):
        """
        Modify the synaptic efficacies such that the learns
        to classify the input pattern correctly.
        Whenever an error occurs, the following update is
        computed:

        dw = lambda sum_{ti} K(t_max, ti)

        The synaptic efficacies are increased by this weight
        if the tempotron did erroneously not elecit an output
        spike, and decreased if it erroneously did.
        :param spike_times: an array with lists of spike times
                            for every afferent
        :param output_spike: the classification of the input pattern
        :type output_spike: Boolean
        """
        #print("Target in adapt:", target)
        # compute tmax
        try:
            tmax = self.compute_tmax(spike_times)
        except:
            tmax = 0
        vmax = self.compute_membrane_potential(tmax, spike_times)

        # print "vmax = ", vmax
        # print "target = ", target

        # if target output is correct, don't adapt weights
        if (vmax >= self.threshold) == target:  ## get results
            # print "no weight update necessary"
            return

        # compute weight updates
        dw = self.dw(learning_rate, tmax, spike_times)
        # print "update =", dw


        if target == True: ## repair
            self.efficacies += dw
        else:
            self.efficacies -= dw

    def dw(self, learning_rate, tmax, spike_times):
        """
        Compute the update for synaptic efficacies wi,
        according to the following learning rule
        (implementing gradient descend dynamics):

        dwi = lambda sum_{ti} K(t_max, ti)

        where lambda is the learning rate and t_max denotes
        the time at which the postsynaptic potential V(t)
        reached its maximal value.
        """
        # compute the contributions of the individual spikes at
        # time tmax
        spike_contribs = self.compute_spike_contributions(tmax, spike_times)

        # multiply with learning rate to get updates
        update = learning_rate * spike_contribs

        return update



