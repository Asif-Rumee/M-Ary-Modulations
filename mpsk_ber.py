"""
Performance analysis of M-ary PSK over AWGN channel by BER vs Eb/N0 chart.
Author: Asif Rahman Rumee
Date: 05/01/2022

"""
import numpy as np
import matplotlib.pyplot as plt
from ModulationPy import PSKModem
    
EbN0dBs = np.arange(start=-2,stop = 26, step = 2)
number_symbols = 10**4
modulations_array = [2, 4, 8, 16, 32]
colors = ['k', 'b', 'r', 'm', 'y']
fig, ax = plt.subplots(nrows=1,ncols=1)

for idx_modulation, modulation in enumerate(modulations_array):
    BER = np.zeros(len(EbN0dBs))
    number_bits_per_symbol = np.log2(modulation)
    modem = PSKModem(modulation, soft_decision=False)
    
    for idx_EbN0dB, EbN0dB in enumerate(EbN0dBs):
        EsN0dB = EbN0dB + 10*np.log10(number_bits_per_symbol)
        input_symbol_bits = np.random.randint(0, 2, int(number_symbols*number_bits_per_symbol))
        modulated_symbol_bits = modem.modulate(input_symbol_bits)
        Es = np.mean(np.abs(modulated_symbol_bits)**2) 
        No = Es/((10**(EbN0dB/10))*np.log2(modulation))

        noisy = modulated_symbol_bits + np.sqrt(No/2) * (np.random.randn(modulated_symbol_bits.shape[0])+ 1j*np.random.randn(modulated_symbol_bits.shape[0])) # AWGN

        demodulated_symbol_bits = modem.demodulate(noisy)
        BER[idx_EbN0dB] = np.mean(np.abs(input_symbol_bits - demodulated_symbol_bits))
        
    ax.semilogy(EbN0dBs,BER,color=colors[idx_modulation],marker='.',linestyle='-',label='M = '+str(modulation))
    
ax.set_title('Performance analysis of M-PSK over AWGN')
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('BER')
ax.grid()
ax.legend()
plt.show()
