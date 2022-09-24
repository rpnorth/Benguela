import scipy as sp
import scipy.stats as ss
from scipy.stats import chi2
from scipy import signal
import pycwt as wavelet
from pycwt.helpers import find
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import pyplot
from cycler import cycler
import src.ssa_core as ssa

def detrend(h):
    n = len(h)
    t = np.arange(n)
    p = np.polyfit(t, h, 1)
    h_detrended = h - np.polyval(p, t)
    return h_detrended

def quadwin(n):
    """
    Quadratic (or "Welch") window
    """
    t = np.arange(n)
    win = 1 - ((t - 0.5 * n) / (0.5 * n)) ** 2
    return win

def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw

def spectrum2(h, dt=1, nsmooth=5):
    """
    Add simple boxcar smoothing to the raw periodogram.
    
    Chop off the ends to avoid end effects.
    """
    freqs, ps, psd = spectrum1(h, dt=dt)
    weights = np.ones(nsmooth, dtype=float) / nsmooth
    ps_s = np.convolve(ps, weights, mode='valid')
    psd_s = np.convolve(psd, weights, mode='valid')
    freqs_s = np.convolve(freqs, weights, mode='valid')
    return freqs_s, ps_s, psd_s
    
def spectrum3(h, dt=1, nsmooth=5):
    """
    Detrend and apply a quadratic window.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    n = len(h)

    h_detrended = detrend(h)
    
    winweights = quadwin(n)
    h_win = h_detrended * winweights
    
    freqs, ps, psd = spectrum2(h_win, dt=dt, nsmooth=nsmooth)
    
    # Compensate for the energy suppressed by the window.
    psd *= n / (winweights**2).sum()
    ps *= n**2 / winweights.sum()**2
    
    return freqs, ps, psd

def win_dtr(h):
    n = len(h)
    h_detrended = detrend(h)    
    winweights = quadwin(n)
    h_win = h_detrended * winweights
    return h_win

def plot_loglog_slope(ax_in,slope_power,ff,line_color='#cccccc',factor=1):
    xlim1,xlim2 = ax_in.get_xlim() # messes up axes
    ylim1,ylim2 = ax_in.get_ylim()
    for si in range(slope_power.shape[0]):
        ax_in.plot(ff, factor[si]/ff**(slope_power[si]),'--','lw',.5,color=line_color,label='_nolegend_') 
    ax_in.set_xlim(xlim1,xlim2)
    ax_in.set_ylim(ylim1,ylim2)
    
def get_fft_conf_interval(psd3,probability,dof):
    #Calculates the Confidence interval
    alfa = 1 - probability
    c = chi2.ppf([1 - alfa / 2, alfa / 2], dof)
    c = dof / c
    psd3_lower = psd3 * c[0]
    psd3_upper = psd3 * c[1]
    return psd3_lower, psd3_upper

def run_wavelet(data_in, avg1=4, avg2=6):
    #data_in = rcm.U[100:-100].load()
    dat = data_in  # remove start and end as may be affected by deployment 

    #avg1, avg2 = (4,6)                  # Range of periods to average - days
    slevel = 0.95                        # Significance level

    std = dat.std()                      # Standard deviation
    std2 = std ** 2                      # Variance
    dat = (dat - dat.mean()) / std       # Calculating anomaly and normalizing

    N = dat.shape[0]                          # Number of measurements
    N
    time = dat.time #np.arange(0, N) * rcm.dt + dat.time[0]  # Time array in days
    dt = np.diff(data_in.time).mean()

    dj = 1 / 12                          # Twelve sub-octaves per octaves
    s0 = 2 * dt # -1  # 2 * dt                    # Starting scale, here 6 months
    J = -1  # 7 / dj                     # Seven powers of two with dj sub-octaves
    #  alpha = 0.0                       # Lag-1 autocorrelation for white noise
    try:
        alpha, _, _ = wavelet.ar1(dat.values)   # Lag-1 autocorrelation for red noise
    except Warning:
        # When the dataset is too short, or there is a strong trend, ar1 raises a
        # warning. In this case, we assume a white noise background spectrum.
        alpha = 1.0

    mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6

    # The following routines perform the wavelet transform and siginificance
    # analysis for the chosen data set.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat.values, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother)

    # Normalized wavelet and Fourier power spectra
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # Significance test. Where ratio power/sig95 > 1, power is significant.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=slevel,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Power rectification as of Liu et al. (2007). 
    # !!!! TODO: confirm if significance test ratio should be calculated first.
    power_uncorrected = power
    glbl_power_uncorrected = power_uncorrected.mean(axis=1)
    power =  power/scales[:, None] # rpn removed /= because it changes power_uncorrected as well
    power = power*N*dt # rpn !!! from matlab code, arg.MaxScale, which seems to be N*dt in wavelet.cwt

    # Calculates the global wavelet spectrum and determines its significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales                     # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(std2.values, dt, scales, 1, alpha,
                                            significance_level=slevel, dof=dof,
                                            wavelet=mother)
    glbl_signif = N*dt*glbl_signif/scales # rpn !!! N*dt*   /scales to match glbl_power correction

    # Scale average between avg1 and avg2 periods and significance level
    sel = find((period >= avg1) & (period < avg2))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    scale_avg = power_uncorrected / scale_avg  # rpn !! because I can't figure out what the correct signif should be
    scale_avg = std2.values * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(std2.values, dt, scales, 2, alpha,
                                                 significance_level=slevel,
                                                 dof=[scales[sel[0]],
                                                      scales[sel[-1]]],
                                                 wavelet=mother)

    return scale_avg_signif, scale_avg, glbl_signif, glbl_power, power, sig95, fft_theor, period, fft_power, fftfreqs, iwave, scales,coi, std2, dt, avg1, avg2, N, dat, time

def calc_wavelet_slopes(power, period, window=15, pc_s=0):
    # first smooth the power data using SSA
    power_run = np.nan*power
    #window = 15
    pc_s = 0 # np.arange(0,1,1)
    for ri in range(power.shape[1]):
        pc_out, _, v_out = ssa.ssa(power[:,ri], window)
        power_run[:,ri] = ssa.inv_ssa(pc_out, v_out, pc_s)

    period_array = np.array(np.tile(period,(power_run.shape[1],1))).T # T to transpose
    
    # get slope or change and related periods
    slope = ((np.diff(power_run,axis=0)))#/(np.log10(np.diff(period_array,axis=0)))
    period_shift = np.diff(period)/2 + period[0:-1]
    
    return slope, period_shift

    #fig, axs = plt.subplots(nrows=3,figsize=(10,4))
    #axs[0].loglog(period,power[:,3000])
    #axs[0].loglog(period,power_run[:,3000])
    #axs[0].loglog(period_shift,slope[:,3000])

    #axs[1].plot(np.log2(period),(power[:,3000]))
    #axs[1].plot(np.log2(period),(power_run[:,3000]))
    #axs[1].set_xlim([-6,6])#

    #axs[2].plot(np.log2(period_shift),((slope[:,3000])))
    ##axs[2].set_ylim([-50,50])
    #axs[2].set_xlim([-6,6])
    #time[100]

    ## first smooth the power data
    #power_dat = xr.DataArray(power, coords=[period,rcm.time[0:power.shape[1]]],dims=['period','time'])
    #window = 50 
    #power_run = power_dat.rolling(period=window, center=True).mean().dropna('period')
    #power.shape, power_run.shape, period.shape
