"""
Performance analysis of M-ary QAM over AWGN channel by BER vs Eb/N0 chart.
Author: Asif Rahman Rumee
Date: 05/01/2022

"""
import numpy as np
import matplotlib.pyplot as plt
from ModulationPy import QAMModem
    
EbN0dBs = np.arange(start=-2,stop = 26, step = 2)
number_symbols = 10**4
modulations_array = [4, 16, 64, 256]
colors = ['k', 'b', 'r', 'm']
fig, ax = plt.subplots(nrows=1,ncols=1)

for idx_modulation, modulation in enumerate(modulations_array):
    BER = np.zeros(len(EbN0dBs))
    number_bits_per_symbol = np.log2(modulation)
    modem = QAMModem(modulation, soft_decision=False)
    
    for idx_EbN0dB, EbN0dB in enumerate(EbN0dBs):
        EsN0dB = EbN0dB + 10*np.log10(number_bits_per_symbol)
        noise_variance = 10**(EsN0dB/10)
        input_symbol_bits = np.random.randint(0, 2, int(number_symbols*number_bits_per_symbol))
        modulated_symbol_bits = modem.modulate(input_symbol_bits)
        Es = np.mean(np.abs(modulated_symbol_bits)**2) 
        No = Es/((10**(EbN0dB/10))*np.log2(modulation))

        noisy = modulated_symbol_bits + np.sqrt(No/2) * (np.random.randn(modulated_symbol_bits.shape[0])+ 1j*np.random.randn(modulated_symbol_bits.shape[0])) # AWGN

        demodulated_symbol_bits = modem.demodulate(noisy, noise_var=noise_variance)
        BER[idx_EbN0dB] = np.mean(np.abs(input_symbol_bits - demodulated_symbol_bits))
        
    ax.semilogy(EbN0dBs,BER,color=colors[idx_modulation],marker='o',linestyle='-',label='M = '+str(modulation))
    
plt.title('Performance analysis of M-QAM over AWGN')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend()
plt.show()