import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sp
import pandas as pd
import Denmark_Strait.src.ssa_core as ssa
import Denmark_Strait.src.spectra_and_wavelet_functions as sw
import pycwt as wavelet
from pyspec import helmholtz as helm
from pyspec import spectrum as spec    

def fftwaveplt_KE(scan_sadcp_in, depth_range_in=None,ylim=[4*10**-2, 10**1],xlim=[10**3, 6*10**4],nbins_spec_av=10,wavelet_scale_av=[10**3,10**4],psd_only=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import Denmark_Strait.src.spectra_and_wavelet_functions as sw
    import scipy as sp
    import scipy.stats as ss
    import pycwt as wavelet
    from pyspec import spectrum as spec    
    
    # based on plotting in SubEx_Paper4_Analysis.ipynb
    
    # Loop through each depth and get spectra of each; then average together
      
    # Second setup values for Welch windowing
    # N sample series, divided into P segments of D samples with a shift S between adjacent segments
    probability = 0.95
    
    if psd_only is None:
        nrows = 3
    else: 
        nrows = 1
            
    fig, axs = plt.subplots(nrows=3,ncols=depth_range_in.size, sharex=True, figsize=(18,12)) #
    
    for di in (range(depth_range_in.size)):
        color_idx = np.linspace(0, 1, len(scan_sadcp_in)+1)
        for ci, ti in zip(color_idx, range(len(scan_sadcp_in))): # loop through transects
            # get range of depths to be used
            if isinstance(depth_range_in[di],slice): 
                depth_range = scan_sadcp_in[ti].z.sel(z=depth_range_in[di]).values # depth_range_in must be negative 
            else: # use full depth range 
                depth_range = scan_sadcp_in[ti].z.values

            N = scan_sadcp_in[ti].xy.shape[0]
            D = round(N/3)
            S = D*0.5 # 50% shift
            # Number of estimations
            P = int((N - D) / S) + 1
            # degrees of freedom
            dof = 2*P

            #color_idx = np.linspace(0, 1, scan_sadcp_in[ti].z.shape[0])
            for zi in depth_range: # loop through depths

                dx = scan_sadcp_in[ti].x_m.diff('xy').mean().values
                # PSD 
                wavenumKE, psdKE = sp.signal.welch(scan_sadcp_in[ti].ke.sel(z=zi).dropna('xy'), 
                                               fs=1/dx, window="hanning", nperseg=D, noverlap=S, detrend="linear")
                # Averaging with 10 bins decade; from pyspec; removing k=0
                wavenumKE, psdKE =  spec.avg_per_decade(wavenumKE[1:],psdKE[1:],nbins=nbins_spec_av)

                if zi==depth_range[0]:
                    axs[0,di].set_title('a) KE - PSD')
                if zi==depth_range[0]:
                    psdKE_transect = psdKE
                if psdKE.size > 0:
                    psdKE_transect = np.vstack((psdKE_transect, psdKE)) 
                    wavenumKE_transect = wavenumKE

                # global wavelet spectrum
                if psd_only is None:
                    # getting coords right for wavelet
                    dat_in = scan_sadcp_in[ti].ke.sel(z=zi).dropna('xy').reset_index('xy').swap_dims({'xy': 'x_m'})
                    if dat_in.size > 0:
                        waveKE = sw.run_wavelet(dat_in,wavelet_scale_av,time_name='x_m',period_name='wavelength')

                        waveKE['glbl_power_var'] = waveKE.std2.values * waveKE.glbl_power
                    axs[2,di].set_xscale('log')
                    axs[2,di].set_yscale('log')

                    if zi==depth_range[0]:
                        axs[1,di].set_title('b) KE - Variance-preserving plot')
                        axs[2,di].set_title('c) KE -  GWS')

                    if zi==depth_range[0]:
                        waveKE_transect = waveKE['glbl_power_var'].values

                    if psdKE.size > 0:
                        waveKE_transect = np.vstack((waveKE_transect, waveKE['glbl_power_var'].values))
                        waveKE_wavelength = waveKE.wavelength.values
                        waveKE_last = waveKE
                    if zi==depth_range[0]:
                        dat_last = dat_in
                        

            num_spectra_transect = psdKE_transect.shape[0]
            # get mean spectra
            psdKE_transect = psdKE_transect.mean(axis = 0)
            dof_transect = dof+num_spectra_transect

            # get psd value for different locations of confidence limit bar
            conf_x = 1/np.array((3*10**4,10**4,5*10**3,10**3,6*10**2))
            conf_y1, conf1 = np.zeros(conf_x.shape), np.zeros((2,conf_x.shape[0]))
            for tti in range(conf_x.shape[0]):
                conf_y1[tti] = psdKE_transect[np.abs(wavenumKE_transect-conf_x[tti]).argmin()]
                conf1[:,tti] = conf_y1[tti] * dof_transect / ss.chi2.ppf([1-probability, probability], dof_transect)

            # confidence interval from pyspec
            El,Eu = spec.spec_error(psdKE_transect, sn=num_spectra_transect, ci=0.95) 
            axs[0,di].fill_between(1/(wavenumKE_transect),El,Eu, color=plt.cm.viridis(ci), alpha=0.25)
            axs[0,di].loglog(1/(wavenumKE_transect),psdKE_transect, alpha=0.85,color=plt.cm.viridis(ci),lw=2,label=('Transect ',str(ti+1)))
            axs[0,di].plot([1/conf_x, 1/conf_x], conf1, color=plt.cm.viridis(ci), lw=4.5,alpha=0.5)
            axs[0,di].plot(1/conf_x, conf_y1, color=plt.cm.viridis(ci), linestyle='none', lw=4.5, 
                    marker='_', ms=8, mew=2,alpha=0.5)
            sw.plot_loglog_slope(axs[0,di],np.array((5/3,2,3)),xlim,ylim)
            
            if psd_only is None:
                 # variance preserving
                axs[1,di].semilogx(1/(wavenumKE_transect), psdKE_transect * wavenumKE_transect, color=plt.cm.viridis(ci), alpha=0.85,lw=2)
                axs[1,di].plot([1/conf_x, 1/conf_x], conf1*conf_x, color=plt.cm.viridis(ci), lw=4.5,alpha=0.5)
                axs[1,di].plot(1/conf_x, conf_y1*conf_x, color=plt.cm.viridis(ci), linestyle='none', lw=4.5, 
                        marker='_', ms=8, mew=2,alpha=0.5)
                # GWS
                waveKE_transect = waveKE_transect.mean(axis = 0)
                axs[2,di].loglog(waveKE_wavelength,waveKE_transect, alpha=0.85,color=plt.cm.viridis(ci),lw=2)
                # Calculates the global wavelet spectrum and determines its significance level.
                std = dat_last.std()                      # Standard deviation
                std2 = std ** 2                      # Variance
                dx = np.diff(dat_last.x_m).mean() #dat_last.time_secs.diff('time_secs').mean('time_secs').values*10**(-9)/(60*60*24) # using time_secs which is an integer
                N = dat_last.shape[0]                          # Number of measurements
                mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6
                slevel = 0.95                        # Significance level
                alpha, _, _ = wavelet.ar1(dat_last.values)
                # !!! IS THIS CORRECT
                dof_transect = dof_transect - waveKE_last.scales #
                waveKE_glbl_signif, tmp = wavelet.significance(std2.values, dx, waveKE_last.scales, 1, alpha,
                                                        significance_level=slevel, dof=dof_transect,
                                                        wavelet=mother)
                waveKE_glbl_signif = N*dx*waveKE_glbl_signif/waveKE_last.scales # rpn !!! N*dt*   /scales to match glbl_power correction
                axs[2,di].loglog(waveKE_wavelength,waveKE_glbl_signif, alpha=0.85,color=plt.cm.viridis(ci),lw=2,linestyle='--')


        if psd_only is None:
            axs[2,di].set_xlabel('Wavelength [m]')
        else:
            axs[0,di].set_xlabel('Wavelength [m]')
        if di==0:
            axs[0,di].set_ylabel('Power Spectral Density [m$^4$ s$^{-4}$]')
            axs[0,di].legend()
        axs[0,di].set_ylim(ylim)
        axs[0,di].set_xlim(xlim)
        axs[0,di].invert_xaxis()

def plot_spec_pyspec(sadcp_transect_all,title,depth_range,nbins_spec_av,spec_variables,ax,left_flag=None,line_color_start=0,
                     var_preserve_plt=False,gws_plt=False,wave_calc=False,ssa_filter=False,xlim_in=None,ylim_in=None):
    
    sadcp_transect_in = sadcp_transect_all[title]
    
    col_ind = np.arange(line_color_start,1,1/len(spec_variables))
    
    for ii in range(len(spec_variables)):
        if len(spec_variables[ii])>0:
            #print(sadcp_transect_in.z.sel(z=depth_range).values)
            K_u, psd_u, nspec_u = sw.pspecd_depth_av(sadcp_transect_in, spec_variable=spec_variables[ii], depth_range_in=depth_range,nbins_spec_av=nbins_spec_av)
            if len(spec_variables)==2 and len(spec_variables[1])==0: 
                line_label = title
            else:
                line_label = spec_variables[ii]

            if var_preserve_plt:
                uEl,uEu = spec.spec_error(K_u*psd_u/2, sn=nspec_u, ci=0.95) 
                ax.fill_between((K_u),uEl,uEu, color=plt.cm.tab20(col_ind[ii]), alpha=0.1)
                ax.loglog((K_u),K_u*psd_u/2, alpha=0.85,color=plt.cm.tab20(col_ind[ii]),lw=2,label=line_label)#('U: along-front'))
            elif gws_plt:
                wave_out = sw.wavelet_depth_av(sadcp_transect_in,spec_variable=spec_variables[ii], depth_range_in=depth_range,ssa_filter=ssa_filter)
                ax.loglog(1/wave_out.scales,wave_out.glbl_power, alpha=0.85,color=plt.cm.tab20(col_ind[ii]),lw=2,label=line_label)
            else:
                uEl,uEu = spec.spec_error(psd_u/2, sn=nspec_u, ci=0.95) 
                ax.fill_between((K_u),uEl,uEu, color=plt.cm.tab20(col_ind[ii]), alpha=0.1)
                ax.loglog((K_u),psd_u/2, alpha=0.85,color=plt.cm.tab20(col_ind[ii]),lw=2,label=line_label)#('U: along-front'))

                if spec_variables[ii] is 'across_eddy_ke':
                    psd_across = psd_u
                if spec_variables[ii] is 'along_eddy_ke':
                    psd_along = psd_u
                    psi_rot, phi_div = helm.spec_helm_decomp(K_u,psd_across, psd_along)
                    rotEl,rotEu = spec.spec_error(psi_rot/2, sn=nspec_u, ci=0.95) 
                    divEl,divEu = spec.spec_error(phi_div/2, sn=nspec_u, ci=0.95) 
                    ax.fill_between((K_u),rotEl,rotEu, color=plt.cm.autumn(0.6), alpha=0.1)
                    ax.loglog((K_u),psi_rot/2, alpha=0.85,color=plt.cm.autumn(0.6),lw=2,label=('Psi: rotational'))
                    ax.fill_between((K_u),divEl,divEu, color=plt.cm.autumn(0.9), alpha=0.1)
                    ax.loglog((K_u),phi_div/2, alpha=0.85,color=plt.cm.autumn(0.9),lw=2,label=('Phi: divergent'))
            
            if wave_calc:
                wave_out = sw.wavelet_depth_av(sadcp_transect_in,spec_variable=spec_variables[ii], depth_range_in=depth_range,ssa_filter=ssa_filter)                
                
    if var_preserve_plt:
        xlim, ylim = [3*10**-6,10**-3], [0, 1e-3]
    elif gws_plt:
        xlim, ylim = [3*10**-6,10**-3], [1e-6, 1e2]
    else:
        xlim, ylim = [3*10**-6,10**-3], [10**-3, 2*10**2]
    if xlim_in is not None: xlim = xlim_in
    if ylim_in is not None: ylim = ylim_in        
    sw.plot_loglog_slope(ax,np.array((-5/3,-2,-3)),xlim,ylim)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("Along-track wavenumber [cpm]")
    if len(spec_variables)>2 and len(spec_variables[1])>0: 
        lg = ax.legend(loc=3)
        ax.text(0.98,0.95,title,
                    transform=ax.transAxes,horizontalalignment='right', verticalalignment='top')
    if left_flag: 
        if var_preserve_plt:
            ax.set_ylabel(r"KE Variance-preserving [1/cpm]")
        elif gws_plt:
            ax.set_ylabel(r"KE GWS [..]")
        else:
            ax.set_ylabel(r"KE spectral density [m$^2$ s$^{-2}$/cpm]")
        ax.text(0.98,0.90,'Depths: %i ' %sadcp_transect_in.z.sel(z=depth_range)[0] + 'to %i m' %sadcp_transect_in.z.sel(z=depth_range)[-1],
                    transform=ax.transAxes,horizontalalignment='right', verticalalignment='top')

    # !!! Minor Ticks are still not quite right
    sw.add_second_axis(ax)
    ax.set_xlabel("Along-track wavenumber [cpm]")
    
    if gws_plt or wave_calc:
        return ax, wave_out
    else:
        return ax
    
def plot_line_at_one_depth(scan_sadcp, var_name, z_value=None,z_index=None, Gradient=False, Filter=False, Factor=False, 
                           ax=None, lcolor=None, llinestyle=None, x_lim=[0,180],y_lim=[-150,0], last_row_flag=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    # due to multi-indexing...
    if z_index: 
        if isinstance(z_index,slice):
            scan_sadcp_z_mean = scan_sadcp[var_name].reset_index('xy').swap_dims({'xy': 'x_km'}).isel(z=z_index).mean(dim='z')
            label_var = (var_name+' over %i ' %scan_sadcp.z.isel(z=z_index)[0] + 'to %i m' %scan_sadcp.z.isel(z=z_index)[-1])
        else:
            scan_sadcp_z_mean = scan_sadcp[var_name].reset_index('xy').swap_dims({'xy': 'x_km'}).isel(z=z_index)
            label_var = (var_name+' at %i m' %scan_sadcp.z.isel(z=z_index).values)
    
    elif z_value:
        if isinstance(z_value,slice):
            scan_sadcp_z_mean = scan_sadcp[var_name].reset_index('xy').swap_dims({'xy': 'x_km'}).sel(z=z_value).mean(dim='z')
            label_var = (var_name+' over %i ' %scan_sadcp.z.sel(z=z_value)[0] + 'to %i m' %scan_sadcp.z.sel(z=z_value)[-1])
        else:
            scan_sadcp_z_mean = scan_sadcp[var_name].reset_index('xy').swap_dims({'xy': 'x_km'}).sel(z=z_value,method='nearest')
            label_var = (var_name+' at %i m' %scan_sadcp.z.sel(z=z_value,method='nearest').values)

    if var_name is 'sigma_0':
        scan_sadcp_z_mean = 10*(scan_sadcp_z_mean/scan_sadcp_z_mean.mean('x_km')-np.mean(scan_sadcp_z_mean/scan_sadcp_z_mean.mean('x_km')))
        label_var = ('normalized('+label_var+')x10')
    if Filter:
        y = ssa.ssafilter(scan_sadcp_z_mean,np.int(scan_sadcp_z_mean.size/3),[0,1])
        label_var = ('filtered('+label_var+')')
    else:
        y=scan_sadcp_z_mean.values
    if Gradient:
        y = np.diff(y)
        x = scan_sadcp_z_mean.x_km[0:-1].values+scan_sadcp_z_mean.x_km.diff(dim='x_km').mean(dim='x_km').values
        label_var = ('grad('+label_var+')')
    else:
        x = scan_sadcp_z_mean.x_km.values
    if Factor:
        y = y*Factor
        label_var = (label_var+'x10')
        
    ax.text(0.98,0.95,scan_sadcp.TransectName,
                transform=ax.transAxes,horizontalalignment='right', verticalalignment='top')

    ax.plot(x,y,label=label_var,color=lcolor,linestyle=llinestyle)

    if last_row_flag is not None:
        ax.set_xlabel('Distance [km]')
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False) 
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim) 

    if scan_sadcp.TransectDirection in ['N-S','W-E']: 
        # then flip it so it is always S-N or E-W
        ax.invert_xaxis()

    if scan_sadcp.TransectDirection in ['N-S', 'S-N']: 
        ax.text(5/80,-0.12,'South',transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom')
        ax.text(7.75/8,-0.12,'North',transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom')
    else:
        ax.text(5/80,-0.12,'East',transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom')
        ax.text(7.75/8,-0.12,'West',transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom')
    return ax

def plot_profile_view(scan_sadcp, ctd_ladcp, var_names, x_lim=[0,180],y_lim=[-150,0],pcolormesh_flag=None,last_row_flag=None,axs_in=None,ctd_only=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    import pandas as pd

    # ================        
    # Plot sections of different variables
    # ================        
    U_range = np.array((-0.4,0.4))
    T_range = np.array((15,18)) #sst_range
    S_range = np.array((35.,35.6)) #sst_range
    Rho_range = np.array((1025.6,1026.6)) #((ctd_data.RHO.min(),ctd_data.RHO.max()))
    sigma_range = np.array((25.6,26.2)) #Rho_range-1000
    N2_range = np.array((-5,-3)) #np.array((0.0001,0.1)) #Rho_range-1000
    M2_range = np.array((-7,-6)) #np.array((0.0001,0.1)) #Rho_range-1000
    Ri_range = np.array((0,1)) #Rho_range-1000
    PV_range = np.array((-6,6))
    Instab_range = np.array((0,5))
    #x_lim = [0,180]
    sigma_levels = np.arange(sigma_range[0],sigma_range[1]+.2,0.1)

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    from matplotlib.colors import BoundaryNorm
    cmap = plt.cm.RdBu_r
    levels = np.arange(-0.5,0.5,0.001)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    shading_type = 'flat' #'gouraud' #'flat'

    ncols, nrows = 8, 1
    if scan_sadcp is None: ncols, nrows = len(ctd_ladcp)*2, 1
    if ctd_ladcp is None:  ncols, nrows = len(scan_sadcp)*2, 1
    if ctd_ladcp is None and var_names[1] == '': ncols, nrows = len(scan_sadcp), 1
    elif var_names[1] == '': ncols, nrows = len(scan_sadcp) + len(ctd_ladcp), 1
    
    #ncols, nrows = 8, 1
    if axs_in is None:
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey=True, figsize=(20,5))
    
    cik = 0
    for ti in range(2): # loop through scan_sadcp and ctd_ladcp
        # skip empty values
        if (ti == 0 and scan_sadcp is None) or (ti==1 and ctd_ladcp is None): continue

        if ti == 0: platform, subscrpt = scan_sadcp, 'SADCP'
        if ti == 1: platform, subscrpt = ctd_ladcp, 'LADCP'
        
        if isinstance(platform, dict): # if OrderedDict
            ind_in = platform
        else:
            ind_in = range(len(platform))  

        rik = 0
        for ri in ind_in: # loop through the different sections
            # 
            ci_next = range(len(var_names)) 
            for ci in range(len(var_names)): # loop through the different variables
                if var_names[ci] == '': continue # blank
                
                if axs_in is None: 
                    axs = ax[cik]
                elif isinstance(axs_in,list): 
                    axs=axs_in[cik]
                else:
                    axs=axs_in

                if var_names[ci] == 'u' or var_names[ci] == 'u_prime': v_range, cmap_in, title = U_range, plt.cm.RdBu_r, 'u$_{'+subscrpt+'}$ [ms$^{-1}$]'
                if var_names[ci] == 'v' or var_names[ci] == 'v_prime': v_range, cmap_in, title = U_range, plt.cm.RdBu_r, 'v$_{'+subscrpt+'}$ [ms$^{-1}$]'
                if var_names[ci] == 'across' or var_names[ci] == 'across_prime': v_range, cmap_in, title = U_range, plt.cm.RdBu_r, 'across$_{'+subscrpt+'}$ [ms$^{-1}$]'
                if var_names[ci] == 'along' or var_names[ci] == 'along_prime': v_range, cmap_in, title = U_range, plt.cm.RdBu_r, 'along$_{'+subscrpt+'}$ [ms$^{-1}$]'
                if var_names[ci] == 'CT': v_range, cmap_in, title = T_range, plt.cm.viridis, 'T$_C$ [$^\circ$]' #RdBu_r
                if var_names[ci] == 'SA': v_range, cmap_in, title = S_range, plt.cm.viridis, 'S$_A$ [g kg$^{-1}$]'
                if var_names[ci] == 'sigma_0': v_range, cmap_in, title = sigma_range, plt.cm.viridis, r'$\rho_{\theta}$ [$kg m^{-3}$]'
                if var_names[ci] == 'db_dx_log10': v_range, cmap_in, title = M2_range, plt.cm.Blues,'M$^2_{'+subscrpt+'}$ [s$^{-2}$]'
                if var_names[ci] == 'db_dz_log10': v_range, cmap_in, title = N2_range, plt.cm.Blues,'N$^2_{'+subscrpt+'}$ [s$^{-2}$]'
                if var_names[ci] == 'Rib': v_range, cmap_in, title = Ri_range, plt.cm.Blues_r ,'Ri$^B_{'+subscrpt+'}$ [-]'
                if var_names[ci] == 'Rig': v_range, cmap_in, title = Ri_range, plt.cm.Blues_r ,'Ri$^G_{'+subscrpt+'}$ [-]'
                if var_names[ci] == 'EPV_plot': v_range, cmap_in, title = PV_range, plt.cm.coolwarm ,'EPV$_{'+subscrpt+'}$ [-]'
                if var_names[ci] == 'relative_vorticity': v_range, cmap_in, title = PV_range, plt.cm.coolwarm ,'RV$_{'+subscrpt+'}$ [-]'
                if var_names[ci] == 'Instability_GravMixSymInertStab': v_range, cmap_in, title = Instab_range, plt.cm.magma_r ,'Instability$_{'+subscrpt+'}$'
                if var_names[ci] == 'geo_velocity': v_range, cmap_in, title = U_range, plt.cm.RdBu_r ,'u$^Geo_{'+subscrpt+'}$ [-]'
                    
                if var_names[ci] == 'db_dx_log10': platform[ri]['db_dx_log10']  = np.log10(abs(platform[ri]['db_dx']))
                if var_names[ci] == 'db_dz_log10': platform[ri]['db_dz_log10']  = np.log10(abs(platform[ri]['db_dz']))
                if var_names[ci] == 'EPV_plot': platform[ri]['EPV_plot']  = platform[ri]['Ertel_Potential_Vorticity']*10**9
                if var_names[ci] == 'relative_vorticity': platform[ri]['relative_vorticity']  = platform[ri]['relative_vorticity']*10**6

                if pcolormesh_flag is None:
                    # due to multi-indexing, problems with x-axis; workaround: reset_index('xy')
                    conmap = platform[ri][var_names[ci]].reset_index('xy').plot.contourf(x='x_km',y='z',ax=axs,vmin=v_range[0],vmax=v_range[1],
                                                                    cmap = cmap_in, levels = np.linspace(v_range[0],v_range[1],21),
                                                                    cbar_kwargs={'ticks': np.linspace(v_range[0],v_range[1],4), 
                                                                                'orientation':"horizontal",'pad': -0.2,
                                                                                'label':'','shrink':0.5,'aspect':10})
                elif var_names[ci] == 'Instability_GravMixSymInertStab':
                    conmap = platform[ri][var_names[ci]].reset_index('xy').plot.pcolormesh(x='x_km',y='z',ax=axs,vmin=v_range[0],vmax=v_range[1],
                                                                    cmap = cmap_in, levels = np.arange(0,6,1), shading = shading_type,
                                                                    cbar_kwargs={'label': ([' ','GI','MI','SI','II']), 
                                                                                'orientation':"horizontal",'pad': -0.2,
                                                                                'shrink':0.5,'aspect':10})
                else:    
                    conmap = platform[ri][var_names[ci]].reset_index('xy').plot.pcolormesh(x='x_km',y='z',ax=axs,vmin=v_range[0],vmax=v_range[1],
                                                                        cmap = cmap_in, shading = shading_type, levels = np.linspace(v_range[0],v_range[1],21),
                                                                        cbar_kwargs={'ticks': np.linspace(v_range[0],v_range[1],3), 
                                                                                    'orientation':"horizontal",'pad': -0.2,
                                                                                    'label':'','shrink':0.5,'aspect':10})
                if var_names[ci] == 'Instability_GravMixSymInertStab':
                    # Mixed layer depth
                    platform[ri]['MLD'] = platform[ri].sigma_0 - platform[ri].sigma_0.sel(z=platform[ri].z.max(),method='nearest') # density nearest to surface
                    conmap3 = platform[ri].MLD.T.reset_index('xy').plot.contour(x='x_km',y='z',ax=axs,levels=[0.01,0.1],colors='0.05',linewidths=1.5)
                elif hasattr(platform[ri], 'sigma_0'):
                    # contour of density on top !!! Note. xarray seems to account for different plotting locations of contour and pcolormesh
                    conmap2 = platform[ri].sigma_0.reset_index('xy').plot.contour(x='x_km',y='z',ax=axs,levels = sigma_levels,colors='0.25',linewidths=1)
                elif ctd_only:
                    conmap2 = ctd_only[ri].sigma_0.reset_index('xy').plot.contour(x='x_km',y='z',ax=axs,levels = sigma_levels,colors='0.25',linewidths=1)
                    

                if rik > 0: # remove colorbar
                    conmap.colorbar.remove()     
                else:
                    conmap.colorbar.ax.tick_params(labelsize=10) #

                axs.set_title(title)

                if cik == 0:
                    axs.set_ylabel('Depth [m]')
                elif axs_in is not None:
                    axs.set_ylabel('Depth [m]')
                else:
                    axs.set_ylabel('')
                    axs.tick_params(labelleft=False) 
                if last_row_flag is not None:
                    axs.set_xlabel('Distance [km]')
                else:
                    axs.set_xlabel('')
                    axs.tick_params(labelbottom=False) 
                axs.set_xlim(x_lim)
                axs.set_ylim(y_lim)

                if ci == 0:
                    dates = pd.Series(platform[ri].time.values)
                    axs.text(0,0,(str(dates[0].month)+'-'+str(dates[0].day)+' '+str(dates[0].hour)+':'+str(dates[0].minute)),
                                   transform=axs.transAxes,horizontalalignment='left',verticalalignment='top',fontsize=10)        
                    axs.text(1,0,(str(dates[dates.size-1].month)+'-'+str(dates[dates.size-1].day)+' '+str(dates[dates.size-1].hour)+':'+str(dates[dates.size-1].minute)),
                                   transform=axs.transAxes,horizontalalignment='right',verticalalignment='top',fontsize=10)        
                #else:
                #    axs.text(0,0.9,' S',transform=axs.transAxes,horizontalalignment='left',verticalalignment='top',fontsize=10)      
                #    axs.text(1,0.9,' N',transform=axs.transAxes,horizontalalignment='right',verticalalignment='top',fontsize=10)      
                
                #if platform[ri].lat.values[0] > platform[ri].lat.values[-1]:
                #    axs.invert_xaxis()
                    
                if platform[ri].TransectDirection in ['N-S','W-E']: 
                    # then flip it so it is always S-N or E-W
                    axs.invert_xaxis()

                if platform[ri].TransectDirection in ['N-S', 'S-N'] and ci == 0: 
                    axs.text(5/80,-0.12,'South',transform=axs.transAxes,horizontalalignment='right',verticalalignment='bottom')
                    axs.text(7.75/8,-0.12,'North',transform=axs.transAxes,horizontalalignment='right',verticalalignment='bottom')
                elif ci == 0:
                    axs.text(5/80,-0.12,'East',transform=axs.transAxes,horizontalalignment='right',verticalalignment='bottom')
                    axs.text(7.75/8,-0.12,'West',transform=axs.transAxes,horizontalalignment='right',verticalalignment='bottom')

                if axs_in is None: 
                    ax[cik] = axs
                elif isinstance(axs_in,list): 
                    axs_in[cik]=axs
                else:
                    axs_in=axs
                
                cik = cik +1
            rik = rik+1

        
    if axs_in: return axs_in
    else:   return ax
    
def plot_map_view(sadcp=None, ctd_data=None, glider_track=None, ladcp_data=None, scanfish_data=None, scan_sadcp=None, ctd_ladcp=None,
                              topo=None,sst_map=None,sst_map1=None,ssh_name='sst',x_lim=[0,180],axs=None):
    
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from oceans.datasets import etopo_subset
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    from matplotlib import animation, rc
    from IPython.display import HTML
    import pandas as pd

    # Stations map.

    if axs is None:
        fig, ax = plt.subplots(ncols = 2, sharey=True, figsize=(15,8))
        num_subplots = range(2)
    else:
        num_subplots = range(1)
    sst_range = np.array((15,19))
    ssh_range = np.array((0.25,0.4))
    Ro_range = np.array((-0.4,0.4))
    lon_0, lat_0, bwidth, bheight =13,-26,450000,300000
    if isinstance(scan_sadcp, dict):
        lon_0, lat_0, bwidth, bheight =12.5,-27,550000,500000
    if axs:
        lon_0, lat_0, bwidth, bheight =axs[3], axs[4], axs[1], axs[2]
    print(bwidth,bheight)    
    
    for si in num_subplots: 
        # setup map
        m = Basemap(width=bwidth,height=bheight,resolution='i',projection='aea',lon_0=lon_0,lat_0=lat_0)
        if axs: 
            m.ax = axs[0]
        else:
            m.ax = ax[si] 
            
        m.drawcoastlines(), m.fillcontinents(color='0.85')
        m.drawparallels(np.arange(-90.,91.,1.),labels=[True,False,False,False],dashes=[2,2]), m.drawmeridians(np.arange(-180.,181.,1.),labels=[False,False,False,True],dashes=[2,2])

        if topo:
            # add contour lines of bathymetry
            lon2, lat2 = np.meshgrid(topo.lon.values,topo.lat.values)
            #m.contourf(lon2, lat2,topo.Band1,40,cmap=plt.cm.Blues_r,latlon=True)
            m.contour(lon2, lat2,topo.Band1,5,linestyles='solid',linewidths=1.,colors='0.35',latlon=True)

        if si == 0:
            # === SST MAP ===
            lon2, lat2 = np.meshgrid(sst_map.lon.values,sst_map.lat.values)
            #m.contourf(lon2, lat2, sst_map.analysed_sst[0,:,:],40,cmap=plt.cm.coolwarm,latlon=True)
            sst_plt = m.pcolormesh(lon2, lat2, sst_map.sst,vmin=sst_range[0],vmax=sst_range[1],cmap=plt.cm.coolwarm,latlon=True)
            plt.text(1,1,sst_map.time_coverage_start[:-8],transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        
        else:
            # === SSH MAP ===
            #m.contourf(lon2, lat2, sst_map1.analysed_sst[0,:,:],40,cmap=plt.cm.coolwarm,latlon=True)
            lon2, lat2 = np.meshgrid(sst_map1.lon_left.values,sst_map1.lat_left.values)
            Ro_plt = m.pcolormesh(lon2, lat2, sst_map1.Ro,vmin=Ro_range[0],vmax=Ro_range[1],cmap=plt.cm.coolwarm,latlon=True)
            lon2, lat2 = np.meshgrid(sst_map1.lon.values,sst_map1.lat.values)
            ssh_plt = m.contour(lon2, lat2, sst_map1.adt,5,linestyles='solid',linewidths=2.,colors='0.01',latlon=True)
            ssh_date = pd.Series(sst_map1.time.values)
            plt.text(1,1,(str(ssh_date[0].year)+'-'+str(ssh_date[0].month)+'-'+str(ssh_date[0].day)),
                     transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        
            # === SSH MAP ===
            #lon2, lat2 = np.meshgrid(ssh_map.lon.values,ssh_map.lat.values)
            #ssh_plt = m.contour(lon2, lat2, ssh_map.sla[0,:,:],vmin=ssh_range[0],vmax=ssh_range[1],colors='0.5',latlon=True) # 
            #plt.text(0,1,ssh_map.time_coverage_start[:-4],transform=m.ax.transAxes,horizontalalignment='left',verticalalignment='bottom')  
            #if hasattr(sst_map1, 'ugos'):
            
            # add Geostrophic current vectors
            gos_plt = m.quiver(lon2, lat2, sst_map1.ugos,sst_map1.vgos,latlon=True)#,scale=700)
            # make quiver key.
            qk = plt.quiverkey(gos_plt, 0.8, 0.8, 0.1, '0.1 m/s', labelpos='W')


        if sadcp is not None:
            # plot ship track from SADCP data
            m.plot(sadcp.lon.values, sadcp.lat.values,'-',color='0.55', latlon=True)

        if ctd_data is not None:
            # plot ladcp/ctd stations
            m.plot(ctd_data.lon.values, ctd_data.lat.values, 'ko', latlon=True)
            #m.plot(ladcp_data.lon.values, ladcp_data.lat.values, 'b.', latlon=True)

        if glider_track is not None:
            # plot glider track
            m.plot(glider_track[0,:].values, glider_track[1,:].values, color='#3cb371',lw=2, latlon=True)

        # plot sections that are used below
        if ctd_ladcp is not None:
            if isinstance(ctd_ladcp, dict): # if OrderedDict
                ind_in = ctd_ladcp
            else:
                ind_in = range(len(ctd_ladcp))  

            rik = 1
            for ri in ind_in:
                m.plot(ctd_ladcp[ri].lon.dropna('xy').values,ctd_ladcp[ri].lat.dropna('xy').values, '.', color= 'chartreuse', latlon=True)
                if si == 0:
                    skip=1
                    ladcp_plt = m.quiver(ctd_ladcp[ri].lon.dropna('xy')[::skip].values, ctd_ladcp[ri].lat.dropna('xy')[::skip].values, 
                                              ctd_ladcp[ri].u.sel(z=ctd_ladcp[ri].z[-5]).dropna('xy')[::skip].values,
                                              ctd_ladcp[ri].v.sel(z=ctd_ladcp[ri].z[-5]).dropna('xy')[::skip].values,
                                              latlon=True)
                    if rik==1: time1 = ctd_ladcp[ri].time[0].values
                    # make quiver key.
                    qk = plt.quiverkey(ladcp_plt, 0.9, 0.9, 0.1, '0.1 m/s', labelpos='W')
                rik = rik+1
            
            start_end_date = pd.Series([time1,ctd_ladcp[ri].time[-1].values])
        
        if scan_sadcp is not None:
            if isinstance(scan_sadcp, dict): # if OrderedDict
                ind_in = scan_sadcp
            elif 'TransectName' in scan_sadcp:
                ind_in = range(len(scan_sadcp.TransectName))
            else:
                ind_in = range(len(scan_sadcp))  
                 
                
            color_idx = np.linspace(0, 1, len(scan_sadcp)+1)
            rik = 1
            for ri in ind_in:
                if 'TransectName' in scan_sadcp:
                    m.plot(scan_sadcp.isel(TransectName=ri).lon.values,scan_sadcp.isel(TransectName=ri).lat.values, lw = 4, color=plt.cm.tab10(rik/10), latlon=True)
                else:
                    m.plot(scan_sadcp[ri].lon.dropna('xy').values,scan_sadcp[ri].lat.dropna('xy').values, lw = 4, color=plt.cm.tab10(rik/10), latlon=True)
                
                if si == 0:
                    skip=5
                    z_ind = -40 # m depth
                    if 'TransectName' in scan_sadcp:
                        if rik==1: time1 = scan_sadcp.isel(TransectName=ri).time[0].values
                    else:
                        sadcp_plt = m.quiver(scan_sadcp[ri].lon.dropna('xy')[::skip].values, scan_sadcp[ri].lat.dropna('xy')[::skip].values, 
                                                  scan_sadcp[ri].u.sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                                  scan_sadcp[ri].v.sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                                  latlon=True,color=plt.cm.tab10(rik/10))
                        if rik==1: time1 = scan_sadcp[ri].time[0].values
                        
                        # make quiver key.
                        qk = plt.quiverkey(sadcp_plt, 0.9, 0.9, 0.1, '0.1 m/s', labelpos='W')
                rik = rik+1
            
            if 'TransectName' in scan_sadcp:
                start_end_date = pd.Series([time1,scan_sadcp.isel(TransectName=ri).time[-1].values])
            else:
                start_end_date = pd.Series([time1,scan_sadcp[ri].time[-1].values])
        #    start_end_date = pd.Series([scan_sadcp[0].time[0].values,scan_sadcp[-1].time[-1].values])

        m.drawmapscale(14.5, -25.5, 15, -25.5, 50,barstyle='fancy')
            
        #if si == 0:
        #    plt.colorbar(sst_plt)#, ticks=[-0.5, 0, 0.5])
        #else:
        #    plt.colorbar(ssh_plt)#, ticks=[-0.5, 0, 0.5])

        if si == 0:
            # add inset showing globe
            #add_globalmap_inset(m)
            # date range of data being plotted            
            if scan_sadcp is not None or ctd_ladcp is not None:
                plt.text(0,1,(str(start_end_date[0].year)+'-'+str(start_end_date[0].month)+'-'+str(start_end_date[0].day)+':'+str(start_end_date[1].month)+'-'+str(start_end_date[1].day)),
                              transform=m.ax.transAxes,horizontalalignment='left',verticalalignment='bottom')        

def quick_map(scan_sadcp,var1,var2,sst_map,lon_0,lat_0,bwidth, bheight,z_ind,ax_in,reverse_flag=None,sst_range=None,scale_avg=None,sla_flag=False):
    from mpl_toolkits.basemap import Basemap
    m = Basemap(width=bwidth,height=bheight,resolution='i',projection='aea',lon_0=lon_0,lat_0=lat_0)
    m.ax = ax_in
    if bwidth>5e5: 
        m.drawcoastlines(), m.fillcontinents(color='0.85')
    m.drawparallels(np.arange(-90.,91.,1.),labels=[True,False,False,False],dashes=[2,2]), m.drawmeridians(np.arange(-180.,181.,1.),labels=[False,False,False,True],dashes=[2,2])

    # === SST/SLA MAP ===
    lon2, lat2 = np.meshgrid(sst_map.lon.values,sst_map.lat.values)
    if sla_flag:
        lon2, lat2 = np.meshgrid(sst_map.lon.values,sst_map.lat.values)
        sst_plt = m.contour(lon2, lat2, sst_map.adt, levels = sst_range, cmap=plt.cm.PiYG,latlon=True)
    else:
        if sst_range is None: sst_range = np.arange(15.,17.51,0.25)       
        #m.contourf(lon2, lat2, sst_map.analysed_sst[0,:,:],40,cmap=plt.cm.coolwarm,latlon=True)
        #sst_plt = m.pcolormesh(lon2, lat2, sst_map.sst,vmin=sst_range[0],vmax=sst_range[1],cmap=plt.cm.coolwarm,latlon=True)
        sst_plt = m.contour(lon2, lat2, sst_map.sst,levels = sst_range, cmap=plt.cm.viridis,latlon=True)
    cbar = m.colorbar(sst_plt,location='right',pad="5%")
    plt.text(1,1,sst_map.time_coverage_start[:-8],transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        

    m.plot(scan_sadcp.lon.values,scan_sadcp.lat.values, lw = 2, latlon=True)
    if scale_avg is None:
        plt.text(1,1,(var1 + ' & ' + var2),transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='top')        
        skip=2
        if reverse_flag:
            sadcp_plt = m.quiver(scan_sadcp.lon.dropna('xy')[::skip].values, scan_sadcp.lat.dropna('xy')[::skip].values, 
                                      -1*scan_sadcp[var1].sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                      -1*scan_sadcp[var2].sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                      latlon=True)
        else:
            sadcp_plt = m.quiver(scan_sadcp.lon.dropna('xy')[::skip].values, scan_sadcp.lat.dropna('xy')[::skip].values, 
                                      scan_sadcp[var1].sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                      scan_sadcp[var2].sel(z=z_ind,method='nearest').dropna('xy')[::skip].values,
                                      latlon=True)
    else:
        plt.text(1,1,('Scale average: '+str(scale_avg.ScaleAvRange[0])+'-'+str(scale_avg.ScaleAvRange[1])+' km'),transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='top')
        if bwidth>5e5: 
            m.scatter(scan_sadcp.lon.values,scan_sadcp.lat.values,c=scale_avg.scale_avg.values, s=scale_avg.scale_avg.values*1e7, alpha=0.5, cmap='autumn', latlon=True)
        else:
            m.scatter(scan_sadcp.lon.values,scan_sadcp.lat.values,c=scale_avg.scale_avg.values, s=scale_avg.scale_avg.values*1e7, alpha=0.5, cmap='viridis', latlon=True)

    start_end_date = pd.Series([scan_sadcp.time[0].values,scan_sadcp.time[-1].values])
    plt.text(0,0,(str(start_end_date[0].year)+'-'+str(start_end_date[0].month)+'-'+str(start_end_date[0].day)+':'+str(start_end_date[1].month)+'-'+str(start_end_date[1].day)),
                  transform=m.ax.transAxes,horizontalalignment='left',verticalalignment='bottom')        

    #m.drawmapscale(14.5, -25.5, 15, -25.5, 50,barstyle='fancy')
    return m


def make_movie_ship_tracks(topo,sadcp,ladcp_data):
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from oceans.datasets import etopo_subset
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    from matplotlib import animation, rc
    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(15,8))

    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,10,-27.5,
    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,13,-26,

    # setup map
    m = Basemap(width=450000,height=300000,resolution='i',projection='aea',lon_0=lon_0,lat_0=lat_0)
    m.ax = ax
    m.drawcoastlines(), m.fillcontinents(color='0.85')
    m.drawparallels(np.arange(-90.,91.,1.),labels=[True,False,False,False],dashes=[2,2]), m.drawmeridians(np.arange(-180.,181.,1.),labels=[False,False,False,True],dashes=[2,2])

    # add contour lines of bathymetry
    lon2, lat2 = np.meshgrid(topo.lon.values,topo.lat.values)
    m.contourf(lon2, lat2,topo.Band1,40,cmap=plt.cm.Blues_r,latlon=True)

    quad1b = m.plot(sadcp.lon.values[0], sadcp.lat.values[0],'ok',latlon=True,alpha=0.5)
    quad1c = m.plot(sadcp.lon.values[0], sadcp.lat.values[0],'ok',latlon=True,alpha=0.5)
    #timelabel = m.text(0.1,1, "",horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    timelabel = plt.text(1,1,"",transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        

    # initialization function: plot the background of each frame
    # plots the parts of the image that don’t change between frames, and adds all of the parts that will change to the axes of this image
    def init():    
        #quad1.set_array([])
        #quad1b.set_array([])
        #quad1c.set_array([])
        return quad1b,quad1c, 

    # animation function.  This is called sequentially
    # The function will automatically get passed one argument: the frame number currently being generated. 
    # Additional arguments can be specified using the fargs argument in FuncAnimation.

    dt_advance = 10 # how many timesteps to advance each image
    t_start = 500
    t_end = sadcp.lon.shape[0]
    color_idx = np.linspace(0, 1, int(np.ceil((sadcp.lon.shape[0]-t_start)/dt_advance)))
    def animate(t):
        marker_style = dict(color=plt.cm.plasma(color_idx[int(t/dt_advance)]), linestyle='none', marker='.',
                            markersize=12, markerfacecolor=plt.cm.plasma(color_idx[int(t/dt_advance)]), alpha=0.5)
        t_plot = t_start + t
        timelabel.set_text('t = %s' %sadcp.time[t_plot].values)
        quad1c[0] = m.plot(sadcp.lon.values[t_plot-dt_advance:t_plot], sadcp.lat.values[t_plot-dt_advance:t_plot],'-',color='0.65',lw=2,latlon=True,alpha=0.5)
        # plot ladcp/ctd stations
        if t == 0:
            ind_pts_to_plot = ladcp_data.time <= sadcp.time[t_plot]
        else:
            ind_pts_to_plot = np.logical_and(ladcp_data.time <= sadcp.time[t_plot],ladcp_data.time > sadcp.time[t_plot-dt_advance]).values

        quad1b[0] = m.plot(ladcp_data.lon.values[ind_pts_to_plot], ladcp_data.lat.values[ind_pts_to_plot],latlon=True,**marker_style)

        return quad1b,quad1c,timelabel,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    # fargs are the other inputs
    anim1 = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=np.arange(0,t_end-t_start,dt_advance), interval=150) # blank with blit; bigger interval slows it down
    
    return(anim1)

def map_limits(m):
    from mpl_toolkits.basemap import Basemap

    llcrnrlon = min(m.boundarylons)
    urcrnrlon = max(m.boundarylons)
    llcrnrlat = min(m.boundarylats)
    urcrnrlat = max(m.boundarylats)
    return llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat

#def make_map(llcrnrlon=10, urcrnrlon=16, llcrnrlat=22, urcrnrlat=28,
#             projection='merc', resolution='i', figsize=(6, 6), inset=True):
#    m = Basemap(llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
#                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
#                projection=projection, resolution=resolution)
#def make_map(ax, lat_1=-20.,lat_2=-32,lon_0=13,lat_0=-25,
#             projection='merc', resolution='i', figsize=(6, 6), inset=True):
#    m = Basemap(llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
#                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
#                projection=projection, resolution=resolution)
#    m = Basemap(width=650000,height=550000,resolution='i',projection='aea',lat_1=lat_1,lat_2=lat_2,lon_0=lon_0,lat_0=lat_0)
    #fig, ax = plt.subplots(figsize=figsize)
    #m.drawstates()
#    m.drawcoastlines()
#    m.fillcontinents(color='0.85')
#    meridians = np.arange(llcrnrlon, urcrnrlon + 2, 2)
#    parallels = np.arange(llcrnrlat, urcrnrlat + 1, 1)
#    m.drawparallels(parallels, linewidth=0, labels=[1, 0, 0, 0])
#    m.drawmeridians(meridians, linewidth=0, labels=[0, 0, 0, 1])
#    m.drawparallels(np.arange(-90.,91.,1.),labels=[True,True,False,False],dashes=[2,2])
#    m.drawmeridians(np.arange(-180.,181.,1.),labels=[False,False,False,True],dashes=[2,2])
#    m.ax = ax

def add_globalmap_inset(m):
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Polygon

    axin = inset_axes(m.ax, width="30%", height="30%", loc=2)
    # Global inset map.
    inmap = Basemap(lon_0=np.mean(m.boundarylons),
                    lat_0=np.mean(m.boundarylats),
                    projection='ortho', ax=axin, anchor='NE')
    inmap.drawcountries(color='white')
    inmap.fillcontinents(color='gray')
    bx, by = inmap(m.boundarylons, m.boundarylats)
    xy = list(zip(bx, by))
    mapboundary = Polygon(xy, edgecolor='k', linewidth=1, fill=False)
    inmap.ax.add_patch(mapboundary)
    return m
            
def OLD_plot_profile_view(scan_sadcp, ctd_ladcp, x_lim=[0,180],M2_flag=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    import pandas as pd

    # ================        
    # Plot u,v,CT,RHO sections.
    # if no CTD, plot M2, Rib
    # ================        

    U_range = np.array((-0.4,0.4))
    T_range = np.array((15,18)) #sst_range
    Rho_range = np.array((1025.6,1026.6)) #((ctd_data.RHO.min(),ctd_data.RHO.max()))
    sigma_range = np.array((25.6,26.)) #Rho_range-1000
    N2_range = np.array((-5,-3)) #np.array((0.0001,0.1)) #Rho_range-1000
    M2_range = np.array((-7,-6)) #np.array((0.0001,0.1)) #Rho_range-1000
    Ri_range = np.array((0,1)) #Rho_range-1000
    #x_lim = [0,180]
    sigma_levels = np.arange(sigma_range[0]+.2,sigma_range[1]+.2,0.2)

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    from matplotlib.colors import BoundaryNorm
    cmap = plt.cm.RdBu_r
    levels = np.arange(-0.5,0.5,0.001)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    shading_type = 'flat'

    ncols, nrows = 8, 2
    if M2_flag is not None: ncols, nrows = 8, 2
       
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey=True, figsize=(20,10))
    for ri in range(len(scan_sadcp)):

        # now add SADCP
        ci_next = np.array((0,1)) + 2*ri
        rik = 0
        for ci in ci_next:
            if ci == ci_next[0] and M2_flag is None: var_name, v_range, cmap_in = 'u' , U_range, plt.cm.RdBu_r#, 'flat'
            if ci == ci_next[1] and M2_flag is None: var_name, v_range, cmap_in = 'v' , U_range, plt.cm.RdBu_r#, 'flat'
            if ci == ci_next[0] and M2_flag is not None: 
                var_name, v_range, cmap_in = 'db_dx_log10', M2_range, plt.cm.Blues#, 'flat'
                scan_sadcp[ri][var_name]  = np.log10(scan_sadcp[ri]['db_dx'])
            if ci == ci_next[1] and M2_flag is not None: var_name, v_range, cmap_in = 'Rib', Ri_range, plt.cm.Blues_r #, 'gouraud'
                
            conmap = scan_sadcp[ri][var_name].reset_index('xy').plot.contourf(x='x_km',y='z',ax=ax[rik,ci],vmin=v_range[0],vmax=v_range[1],
                                                            cmap = cmap_in, 
                                                            cbar_kwargs={'ticks': np.arange(v_range[0],v_range[1]+1,1), 
                                                                        'orientation':"horizontal",'pad': -0.2,
                                                                        'label':''})
            #conmap = ax[rik,ci].contourf(scan_sadcp[ri].distance/1000,scan_sadcp[ri].z,var2plot,
            #                              vmin=v_range[0],vmax=v_range[1],cmap = cmap_in,shading=shading_type)

            if ri > 0: # remove colorbar
                conmap.colorbar.remove()            

            if ci == ci_next[0] and ctd_ladcp is None: 
                ax[rik,ci].set_title('u$_{SADCP}$ [ms$^{-1}$]')
            if ci ==ci_next[1] and ctd_ladcp is None: 
                ax[rik,ci].set_title('v$_{SADCP}$ [ms$^{-1}$]')
            if ci == ci_next[0] and ctd_ladcp is not None: 
                ax[rik,ci].set_title('M$^2_{SADCP}$ [s$^{-1}$]')
            if ci ==ci_next[1] and ctd_ladcp is not None: 
                ax[rik,ci].set_title('Ri$^B_{SADCP}$ [-]')
                
            if ci == ci_next[0] and rik ==0:
                ax[rik,ci].set_ylabel('Depth [m]')
            else:
                ax[rik,ci].set_ylabel('')
                ax[rik,ci].tick_params(labelleft=False) 
            if M2_flag is not None:
                ax[rik,ci].set_xlabel('Distance [km]')
            else:
                ax[rik,ci].set_xlabel('')
                ax[rik,ci].tick_params(labelbottom=False) 
                
            if ci == ci_next[0]:
                dates = pd.Series(scan_sadcp[ri].time.values)
                ax[rik,ci].text(0,0,(str(dates[0].month)+'-'+str(dates[0].day)+' '+str(dates[0].hour)+':'+str(dates[0].minute)),
                               transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='bottom',fontsize=10)        
            ax[rik,ci].set_xlim(x_lim)
            if scan_sadcp[ri].lat.values[0] > scan_sadcp[ri].lat.values[-1]:
                ax[rik,ci].invert_xaxis()
                        
            dx,dy = 0,0 #np.mean(np.diff(scan_sadcp[ri].distance)), np.mean(np.diff(scan_sadcp[ri].z.values))
            # contour needs matrices
            X_grid, Y_grid = np.meshgrid(scan_sadcp[ri].distance +dx/2.,scan_sadcp[ri].z +dy/2.)
            ax[rik,ci].contour(X_grid/1000, Y_grid, scan_sadcp[ri].sigma_0.T, levels = sigma_levels,colors='0.25',linewidths=0.5)

        # now add Scanfish
        if M2_flag is None:
            rik = 1
            for ci in ci_next:
                if ci == ci_next[0]: 
                    var_name2 = 'CT' 
                    c_range = T_range
                    ax[rik,ci].set_title('T$_C$ [$^\circ$]')
                    cmap_in = plt.cm.RdBu_r
                else: 
                    var_name2 = 'sigma_0'
                    c_range = sigma_range #
                    ax[rik,ci].set_title(r'$\rho_{\theta}$ [$kg m^{-3}$]')
                    cmap_in = plt.cm.viridis

                conmap = ax[rik,ci].pcolormesh(scan_sadcp[ri].distance/1000,scan_sadcp[ri].z,scan_sadcp[ri][var_name2].T,
                                           vmin=c_range[0],vmax=c_range[1],
                                           cmap = cmap_in)

                dx,dy = np.mean(np.diff(scan_sadcp[ri].distance)), np.mean(np.diff(scan_sadcp[ri].z))
                # contour needs matrices
                X_grid, Y_grid = np.meshgrid(scan_sadcp[ri].distance +dx/2.,scan_sadcp[ri].z +dy/2.)
                ax[rik,ci].contour(X_grid/1000, Y_grid,scan_sadcp[ri].sigma_0.T, levels = sigma_levels,colors='0.25')

                ax[rik,ci].tick_params(labelbottom=False) 
                if ci > ci_next[0]:
                    ax[rik,ci].set_ylabel('')
                    ax[rik,ci].tick_params(labelleft=False) 
                #if rik == 1:
                #    ax[rik,ci].set_xlabel('Distance [km]')
                ax[rik,ci].set_xlim(x_lim)
                if scan_sadcp[ri].lat.values[0] > scan_sadcp[ri].lat.values[-1]:
                    ax[rik,ci].invert_xaxis()

        # LADCP 
        ci_next = np.array((4,5)) + 2*ri
        rik = 0
        for ci in ci_next:
            if ci == ci_next[0] and M2_flag is None: var_name, v_range, cmap_in = 'u' , U_range, plt.cm.RdBu_r#, 'flat'
            if ci == ci_next[1] and M2_flag is None: var_name, v_range, cmap_in = 'v' , U_range, plt.cm.RdBu_r#, 'flat'
            if ci == ci_next[0] and M2_flag is not None: 
                var_name, v_range, cmap_in = 'db_dz_log10', N2_range, plt.cm.Blues, 
                ctd_ladcp[ri][var_name] = np.log10(ctd_ladcp[ri]['db_dz'])
            if ci == ci_next[1] and M2_flag is not None: var_name, v_range, cmap_in = 'Rig', Ri_range, plt.cm.Blues_r #, 'gouraud'
                
            conmap = ctd_ladcp[ri][var_name].reset_index('xy').plot.contourf(x='x_km',y='z',ax=ax[rik,ci],vmin=v_range[0],vmax=v_range[1], 
                                                           cmap = cmap_in, 
                                                           cbar_kwargs={'ticks': np.arange(v_range[0],v_range[1]+1,1), 
                                                                        'orientation':"horizontal",'pad': -0.2,
                                                                        'label':''})
           #conmap = ax[rik,ci].pcolormesh(ctd_ladcp[ri].distance/1000,ctd_ladcp[ri].z,var2plot,
            #                              vmin=v_range[0],vmax=v_range[1],cmap = cmap_in,shading=shading_type)
            conmap.colorbar.set_label('') # remove
            if ri > 0: # remove colorbar
                conmap.colorbar.remove()

            if ci == ci_next[0] and ctd_ladcp is None: 
                ax[rik,ci].set_title('u$_{LADCP}$ [ms$^{-1}$]')
            if ci ==ci_next[1] and ctd_ladcp is None: 
                ax[rik,ci].set_title('v$_{LADCP}$ [ms$^{-1}$]')
            if ci == ci_next[0] and ctd_ladcp is not None: 
                ax[rik,ci].set_title('N$^2_{LADCP}$ [s$^{-1}$]')
            if ci ==ci_next[1] and ctd_ladcp is not None: 
                ax[rik,ci].set_title('Ri$^G_{LADCP}$ [-]')
            #if ci == ci_next[0]:
            #    ax[rik,ci].set_ylabel('Depth [m]')
            if M2_flag is not None:
                ax[rik,ci].set_xlabel('Distance [km]')
            else:
                ax[rik,ci].set_xlabel('')
                ax[rik,ci].tick_params(labelbottom=False) 
            ax[rik,ci].set_ylabel('')
            ax[rik,ci].tick_params(labelleft=False) 
            ax[rik,ci].set_ylim([-150, 0.0])
            ax[rik,ci].set_xlim(x_lim)
            if ci == 0 and rik == 0:
                #plt.colorbar(conmap)
                axins = inset_axes(ax[rik,ci],
                       width="3%",  # width = 10% of parent_bbox width
                       height="50%",  # height : 50%
                       loc=3,
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[ri,ci].transAxes,
                       borderpad=0,
                       )

                # Controlling the placement of the inset axes is basically same as that
                # of the legend.  you may want to play with the borderpad value and
                # the bbox_to_anchor coordinate.
                plt.colorbar(conmap, cax=axins)#, ticks=[-0.5, 0, 0.5])
            if ci == ci_next[0]:
                dates = pd.Series(ctd_ladcp[ri].time.values)
                ax[rik,ci].text(0,0,(str(dates[0].month)+'-'+str(dates[0].day)+' '+str(dates[0].hour)+':'+str(dates[0].minute)),
                               transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='bottom',fontsize=10)      
            else:
                ax[rik,ci].text(0,1,' S',transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=10)      
                ax[rik,ci].text(1,1,' N',transform=ax[rik,ci].transAxes,horizontalalignment='right',verticalalignment='top',fontsize=10)      

            if ctd_ladcp[ri].lat.values[0] > ctd_ladcp[ri].lat.values[-1]:
                ax[rik,ci].invert_xaxis()
            # contours are *point* based plots, so convert our bound into point centers
            dx,dy = np.mean(np.diff(ctd_ladcp[ri].distance)), np.mean(np.diff(ctd_ladcp[ri].z))
            cf = ax[rik,ci].contour((ctd_ladcp[ri].distance + dx/2. )/1000,
                              ctd_ladcp[ri].z + dy/2., 
                              ctd_ladcp[ri].sigma_0, levels = sigma_levels,colors='0.25',linewidths=0.5)

        # now add CTD Temperature, Density
        if M2_flag is None:
            rik = 1
            for ci in ci_next:
                if ci == ci_next[0]: 
                    var_name2 = 'CT' 
                    c_range = T_range
                    ax[rik,ci].set_title('T$_C$ [$^\circ$]')
                    cmap_in = plt.cm.RdBu_r
                else: 
                    var_name2 = 'sigma_0'
                    c_range=sigma_range
                    ax[rik,ci].set_title(r'$\rho_{\theta}$ [$kg m^{-3}$]')
                    cmap_in = plt.cm.viridis

                conmap = ax[rik,ci].pcolormesh(ctd_ladcp[ri].distance/1000,ctd_ladcp[ri].z,ctd_ladcp[ri][var_name2],
                                              vmin=c_range[0],vmax=c_range[1],cmap = cmap_in,shading=shading_type)
                ax[rik,ci].set_xlabel('')
                ax[rik,ci].tick_params(labelbottom=False) 
                ax[rik,ci].set_ylabel('')
                ax[rik,ci].tick_params(labelleft=False) 
                ax[rik,ci].set_xlim(x_lim)
                if ctd_ladcp[ri].lat.values[0] > ctd_ladcp[ri].lat.values[-1]:
                    ax[rik,ci].invert_xaxis()
                dx,dy = np.mean(np.diff(ctd_ladcp[ri].distance)), np.mean(np.diff(ctd_ladcp[ri].z))
                cf = ax[rik,ci].contour((ctd_ladcp[ri].distance + dx/2. )/1000,
                                  ctd_ladcp[ri].z + dy/2., 
                                  ctd_ladcp[ri].sigma_0, levels = sigma_levels,colors='0.25')

    #return ax
    
    
def OLD_plot_M2_UV_sections(sadcp=None,ind_sadcp_section=None,scanfish_gridded_section=None,scanfish_data_section=None,ladcp_data=None,
                            ind_LADCP_section=None,ctd_data=None,ind_CTD_section=None,topo=None, x_lim_Scan=[0,180],x_lim_CTD = [0,50], 
                            M2_range=None,sigma_range=None,N2_range=None):
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from oceans.datasets import etopo_subset
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    from matplotlib import animation, rc
    from IPython.display import HTML
    import pandas as pd

    #fig, ax = plt.subplots(ncols = 2, sharey=True, figsize=(15,8))
    sst_range = np.array((15,19))
    ssh_range = np.array((-0.1,0.1))
    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,10,-27.5
    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,13,-26

    # ================        
    # Plot u,v,CT,RHO sections.
    # ================        

    U_range = np.array((0,0.8))
    T_range = np.array((15,18)) #sst_range
    Rho_range = np.array((1025.6,1026.6)) #((ctd_data.RHO.min(),ctd_data.RHO.max()))
    if sigma_range is None: sigma_range = np.array((25.6,26.)) #Rho_range-1000
    sigma_levels = np.arange(sigma_range[0],sigma_range[1]+.2,0.1)

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    from matplotlib.colors import BoundaryNorm
    cmap = plt.cm.RdBu_r
    levels = np.arange(-0.5,0.5,0.001)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    shading_type = 'flat'

    nrows = 4

    fig, ax = plt.subplots(nrows = nrows, ncols = 2*len(ind_sadcp_section), sharey=True, figsize=(20,10))
    for ri in range(len(ind_sadcp_section)):

        if scanfish_gridded_section is not None:
            sigma_in = scanfish_gridded_section[ri].sigma_0
            distance_in = scanfish_gridded_section[ri].distance/1000
            depth_in = scanfish_gridded_section[ri].depth
            
            from calc_functions import calc_M2
            b,db_dx,db_dz,x4_dbdx,z4_dbdx,x4_dbdz,z4_dbdz =  calc_M2(sigma_in,distance_in,depth_in)
            
            # Horizontal Buoyancy Gradient of Scanfish
            #g = 9.81
            #rho_o = 1025-1000
            #b = -g*scanfish_gridded_section[ri].sigma_0.T/rho_o # not sure why .T is necessary, but keeps dims consistent
            #db_dx = abs(-b.diff(dim='x')/(scanfish_gridded_section[ri].distance.diff(dim='x')/1000))
            #db_dz = abs(-b.diff(dim='z')/scanfish_gridded_section[ri].depth.diff(dim='z'))
            #x4_dbdx = scanfish_gridded_section[ri].distance[:,0:-1] + scanfish_gridded_section[ri].distance.diff(dim='x')/2
            #z4_dbdx = scanfish_gridded_section[ri].depth[:,0:-1]
            #x4_dbdz = scanfish_gridded_section[ri].distance[0:-1,]
            #z4_dbdz = scanfish_gridded_section[ri].depth[0:-1,] + scanfish_gridded_section[ri].depth.diff(dim='z')/2
            if M2_range is None: M2_range = np.array((np.log10(db_dx.min()),np.log10(db_dx.max())))
            if N2_range is None: N2_range = np.array((np.log10(db_dz.min()),np.log10(db_dz.max())))

        # now add sadcp
        ci_next = np.array((0,1)) + 2*ri
        rik = 0
        ci = ri
        rik = 0
        velocity = (sadcp.u.isel(time=ind_sadcp_section[ri]).T**2 + sadcp.v.isel(time=ind_sadcp_section[ri]).T**2)**0.5
        distance_sadcp = np.cumsum(np.append(np.array(0),gsw.distance(sadcp.lon.isel(time=ind_sadcp_section[ri]), sadcp.lat.isel(time=ind_sadcp_section[ri]),p=0)))/1000
        conmap = ax[rik,ci].pcolormesh(distance_sadcp,-1*sadcp.depth.isel(time=ind_sadcp_section[ri]).mean(dim='time'),velocity,
                                      vmin=U_range[0],vmax=U_range[1],cmap = plt.cm.Reds,shading=shading_type)
        ax[rik,ci].set_title('uv$_{sadcp}$ [ms$^{-1}$]')
        #if rik == 1:
            #ax[rik,ci].set_xlabel('Distance [km]')
        if ci == 0: #ci_next[0]:
            dates = pd.Series(sadcp.time.isel(time=ind_sadcp_section[ri]).values)
            ax[rik,ci].text(0,0,(str(dates[0].month)+'-'+str(dates[0].day)+' '+str(dates[0].hour)+':'+str(dates[0].minute)),
                           transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='bottom',fontsize=10)        
        ax[rik,ci].set_xlim(x_lim_Scan)
        if sadcp.lat.isel(time=ind_sadcp_section[ri]).values[0] > sadcp.lat.isel(time=ind_sadcp_section[ri]).values[-1]:
            ax[rik,ci].invert_xaxis()
        ax[ri,ci].set_ylim([-100, 0.0])
        ax[rik,ci].text(0,1,' S',transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=10)      
        ax[rik,ci].text(1,1,' N',transform=ax[rik,ci].transAxes,horizontalalignment='right',verticalalignment='top',fontsize=10)      
        
        if scanfish_gridded_section is not None:
            dx,dy = np.mean(np.diff(scanfish_data_section[ri].distance.values)), np.mean(np.diff(scanfish_data_section[ri].depth.values))
            vals_2_plot = np.arange(0,len(scanfish_data_section[ri].distance.values),100)
            ax[rik,ci].tricontour(scanfish_data_section[ri].distance.values[vals_2_plot]/1000 +dx/2.,scanfish_data_section[ri].depth.values[vals_2_plot] +dy/2.,
                                           scanfish_data_section[ri].sigma_0.values[vals_2_plot],levels = sigma_levels,colors='0.25')
            ax[rik,ci].set_xticks([])

            # now add Scanfish
            rik = 1
            #c_range = M2_range 
            cmap_in = plt.cm.viridis
            #conmap = ax[rik,ci].pcolormesh(x4_dbdx.T/1000, z4_dbdx.T, db_dx,
            #                              vmin=c_range[0],vmax=c_range[1],cmap = plt.cm.RdBu_r,shading=shading_type)
            c_range = sigma_range 
            conmap = ax[rik,ci].pcolormesh(scanfish_gridded_section[ri].distance, scanfish_gridded_section[ri].depth, scanfish_gridded_section[ri].sigma_0,
                                          vmin=c_range[0],vmax=c_range[1],cmap = cmap_in,shading=shading_type)
            #ax[rik,ci].set_title('M$^2_{Scanfish}$ [s$^{-1}$]')
            ax[rik,ci].tricontour(scanfish_data_section[ri].distance.values[vals_2_plot]/1000 +dx/2.,scanfish_data_section[ri].depth.values[vals_2_plot] +dy/2.,
                                           scanfish_data_section[ri].sigma_0.values[vals_2_plot],levels = sigma_levels,colors='0.25')
            var_name2 = 'sigma_0'
            ax[rik,ci].set_title(r'$\rho_{\theta}$ [$kg m^{-3}$]')
            distance_scanfish = np.cumsum(np.trunc(np.append(np.array(0),gsw.distance(scanfish_data_section[ri].lon.values, scanfish_data_section[ri].lat.values,p=0))))/1000
            conmap = ax[rik,ci].scatter(scanfish_data_section[ri].distance/1000,scanfish_data_section[ri].depth.values,7,c=scanfish_data_section[ri][var_name2].values,
                                   vmin=c_range[0],vmax=c_range[1],cmap = cmap_in)#,alpha = 0.5)
            #ax[rik,ci].set_xlabel('Distance [km]')
            ax[rik,ci].set_xlim(x_lim_Scan)
            if scanfish_data_section[ri].lat.values[0] > scanfish_data_section[ri].lat.values[-1]:
                ax[rik,ci].invert_xaxis()
            ax[rik,ci].set_xticks([])

            #rik = 2
            ##c_range = M2_range 
            #cmap_in = plt.cm.viridis
            ##conmap = ax[rik,ci].pcolormesh(x4_dbdx.T/1000, z4_dbdx.T, db_dx,
            ##                              vmin=c_range[0],vmax=c_range[1],cmap = plt.cm.RdBu_r,shading=shading_type)
            #c_range = sigma_range 
            #conmap = ax[rik,ci].pcolormesh(scanfish_gridded_section[ri].distance/1000, scanfish_gridded_section[ri].depth, scanfish_gridded_section[ri].sigma_0,
            #                              vmin=c_range[0],vmax=c_range[1],cmap = cmap_in,shading=shading_type)
            ##ax[rik,ci].set_title('M$^2_{Scanfish}$ [s$^{-1}$]')
            #ax[rik,ci].tricontour(scanfish_data_section[ri].distance.values[vals_2_plot]/1000 +dx/2.,scanfish_data_section[ri].depth.values[vals_2_plot] +dy/2.,
            #                               scanfish_data_section[ri].sigma_0.values[vals_2_plot],levels = sigma_levels,colors='0.25')
            #var_name2 = 'sigma_0'
            #ax[rik,ci].set_title(r'$\rho_{\theta}$ [$kg m^{-3}$]')
            #ax[rik,ci].set_xlabel('Distance [km]')
            #ax[rik,ci].set_xlim(x_lim_Scan)
            #if scanfish_data_section[ri].lat.values[0] > scanfish_data_section[ri].lat.values[-1]:
            #    ax[rik,ci].invert_xaxis()
            #    print('Flipping rho x-axis')

            rik = 2
            cmap_in = plt.cm.Blues
            c_range = M2_range 
            conmap = ax[rik,ci].pcolormesh(x4_dbdx, z4_dbdx, np.log10(db_dx), vmin=M2_range[0],vmax=M2_range[1],
                                          cmap = cmap_in,shading=shading_type)
            ax[rik,ci].set_title('M$^2_{Scanfish}$ [s$^{-1}$]')
            ax[rik,ci].tricontour(scanfish_data_section[ri].distance.values[vals_2_plot]/1000 +dx/2.,scanfish_data_section[ri].depth.values[vals_2_plot] +dy/2.,
                                           scanfish_data_section[ri].sigma_0.values[vals_2_plot],levels = sigma_levels,colors='0.25')

            #ax[rik,ci].set_xlabel('Distance [km]')
            ax[rik,ci].set_xlim(x_lim_Scan)
            if scanfish_data_section[ri].lat.values[0] > scanfish_data_section[ri].lat.values[-1]:
                ax[rik,ci].invert_xaxis()
            ax[rik,ci].set_xticks([])

            rik = 3
            cmap_in = plt.cm.Blues
            c_range = N2_range
            conmap = ax[rik,ci].pcolormesh(x4_dbdz, z4_dbdz, np.log10(db_dz), vmin=N2_range[0],vmax=N2_range[1],
                                          cmap = cmap_in,shading=shading_type)
            ax[rik,ci].set_title('N$^2_{Scanfish}$ [s$^{-1}$]')
            ax[rik,ci].tricontour(scanfish_data_section[ri].distance.values[vals_2_plot]/1000 +dx/2.,scanfish_data_section[ri].depth.values[vals_2_plot] +dy/2.,
                                           scanfish_data_section[ri].sigma_0.values[vals_2_plot],levels = sigma_levels,colors='0.25')

            ax[rik,ci].set_xlabel('Distance [km]')
            ax[rik,ci].set_xlim(x_lim_Scan)
            if scanfish_data_section[ri].lat.values[0] > scanfish_data_section[ri].lat.values[-1]:
                ax[rik,ci].invert_xaxis()
    
    for ri,ci in zip(range(len(ind_sadcp_section)),np.arange(2,4,1)):
        # Horizontal Buoyancy Gradient of CTD
        distance_ctd = np.cumsum(np.append(np.array(0),gsw.distance(ctd_data.lon.isel(xy=ind_CTD_section[ri]), ctd_data.lat.isel(xy=ind_CTD_section[ri]),p=0)))/1000
        distance_ctd = distance_ctd*np.ones((ctd_data.CT.isel(xy=ind_CTD_section[ri]).shape))
        z_ctd = np.transpose(np.tile(ctd_data.z,(ctd_data.CT.isel(xy=ind_CTD_section[ri]).shape[1],1))) #  vector ctd_data.z*np.ones((ctd_data.CT.isel(xy=ind_CTD_section[ri]).T.shape))
        sigma_in = ctd_data.sigma_0.isel(xy=ind_CTD_section[ri])
        distance_in = distance_ctd
        depth_in = z_ctd
        
        from calc_functions import calc_M2
        b,db_dx,db_dz,x4_dbdx,z4_dbdx,x4_dbdz,z4_dbdz =  calc_M2(sigma_in,distance_in,depth_in)

        #g = 9.81
        #rho_o = 1025-1000
        #b = -g*ctd_data.sigma_0.isel(xy=ind_CTD_section[ri]).T/rho_o # not sure why .T is necessary, but keeps dims consistent
        #db_dx = abs(-b.diff(dim='xy').T/ np.diff(distance_ctd))
        #db_dz = abs(-b.diff(dim='z').T/ np.diff(z_ctd,axis=0))
        #x4_dbdx = distance_ctd[:,0:-1] + np.diff(distance_ctd)/2 
        #z4_dbdx = z_ctd[:,0:-1] #ctd_data.z.isel(xy=ind_CTD_section[ri])[:,0:-1]
        #x4_dbdz = distance_ctd[0:-1,:] 
        #z4_dbdz = z_ctd[0:-1,:] + np.diff(z_ctd,axis=0)/2
        #if M2_range is None: M2_range = np.array((db_dx.min(),db_dx.max()))
        #if N2_range is None: N2_range = np.array((db_dz.min(),db_dz.max()))

        rik = 0
        # uv
        velocity = (ladcp_data.u.isel(xy=ind_LADCP_section[ri])**2 + ladcp_data.v.isel(xy=ind_LADCP_section[ri])**2)**0.5
        distance_ladcp = np.cumsum(np.append(np.array(0),gsw.distance(ladcp_data.lon.isel(xy=ind_LADCP_section[ri]), ladcp_data.lat.isel(xy=ind_LADCP_section[ri]),p=0)))/1000
        conmap = ax[rik,ci].pcolormesh(distance_ladcp,ladcp_data.z,velocity,
                                      vmin=U_range[0],vmax=U_range[1],cmap = plt.cm.Reds,shading=shading_type)
        ax[rik,ci].set_title('uv$_{ladcp}$ [ms$^{-1}$]')
        if ci == 0: #ci_next[0]:
            dates = pd.Series(ladcp_data.time.isel(xy=ind_LADCP_section[ri]).values)
            ax[rik,ci].text(0,0,(str(dates[0].month)+'-'+str(dates[0].day)+' '+str(dates[0].hour)+':'+str(dates[0].minute)),
                           transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='bottom',fontsize=10)      
        else:
            ax[rik,ci].text(0,1,' S',transform=ax[rik,ci].transAxes,horizontalalignment='left',verticalalignment='top',fontsize=10)      
            ax[rik,ci].text(1,1,' N',transform=ax[rik,ci].transAxes,horizontalalignment='right',verticalalignment='top',fontsize=10)      
        ax[rik,ci].set_xlim(x_lim_CTD)
        if ladcp_data.lat.isel(xy=ind_LADCP_section[ri]).values[0] > ladcp_data.lat.isel(xy=ind_LADCP_section[ri]).values[-1]:
            ax[rik,ci].invert_xaxis()
        ax[ri,ci].set_ylim([-100, 0.0])
        # contours are *point* based plots, so convert our bound into point centers
        dx,dy = np.mean(np.diff(distance_ctd)), np.mean(np.diff(ctd_data.z))
        cf = ax[rik,ci].contour(distance_ctd[0,:] + dx/2. ,
                          ctd_data.z + dy/2., 
                          ctd_data.sigma_0.isel(xy=ind_CTD_section[ri]), levels = sigma_levels,colors='0.25')
        ax[rik,ci].set_xticks([])
        
        rik = 1
        # now add Density
        var_name2 = 'sigma_0'
        c_range=sigma_range
        ax[rik,ci].set_title(r'$\rho_{\theta}$ [$kg m^{-3}$]')
        cmap_in = plt.cm.viridis
        conmap = ax[rik,ci].pcolormesh(distance_ctd,ctd_data.z,ctd_data[var_name2].isel(xy=ind_CTD_section[ri]),
                                      vmin=c_range[0],vmax=c_range[1],cmap = cmap_in,shading=shading_type)
        ax[rik,ci].set_xlim(x_lim_CTD)
        if ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[0] > ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[-1]:
            ax[rik,ci].invert_xaxis()
        # contours are *point* based plots, so convert our bound into point centers
        dx,dy = np.mean(np.diff(distance_ctd)), np.mean(np.diff(ctd_data.z))
        cf = ax[rik,ci].contour(distance_ctd[0,:] + dx/2. ,
                          ctd_data.z + dy/2., 
                          ctd_data.sigma_0.isel(xy=ind_CTD_section[ri]), levels = sigma_levels,colors='0.25')
        ax[rik,ci].set_xticks([])

        rik = 2
        cmap_in = plt.cm.Blues
        c_range = M2_range 
        conmap = ax[rik,ci].pcolormesh(x4_dbdx, z4_dbdx, np.log10(db_dx), vmin=M2_range[0],vmax=M2_range[1],
                                      cmap = cmap_in,shading=shading_type)
        ax[rik,ci].set_title('M$^2_{CTD}$ [s$^{-1}$]')
        ax[rik,ci].set_xlim(x_lim_CTD)
        if ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[0] > ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[-1]:
            ax[rik,ci].invert_xaxis()
        # contours are *point* based plots, so convert our bound into point centers
        dx,dy = np.mean(np.diff(distance_ctd)), np.mean(np.diff(ctd_data.z))
        cf = ax[rik,ci].contour(distance_ctd[0,:] + dx/2. ,
                          ctd_data.z + dy/2., 
                          ctd_data.sigma_0.isel(xy=ind_CTD_section[ri]), levels = sigma_levels,colors='0.25')
        ax[rik,ci].set_xticks([])

        rik = 3
        cmap_in = plt.cm.Blues
        c_range = M2_range 
        conmap = ax[rik,ci].pcolormesh(x4_dbdz, z4_dbdz, np.log10(db_dz), vmin=M2_range[0],vmax=M2_range[1],
                                      cmap = cmap_in,shading=shading_type)
        ax[rik,ci].set_title('N$^2_{CTD}$ [s$^{-1}$]')
        ax[rik,ci].set_xlim(x_lim_CTD)
        if ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[0] > ctd_data.lat.isel(xy=ind_CTD_section[ri]).values[-1]:
            ax[rik,ci].invert_xaxis()
        # contours are *point* based plots, so convert our bound into point centers
        dx,dy = np.mean(np.diff(distance_ctd)), np.mean(np.diff(ctd_data.z))
        ax[rik,ci].set_xlabel('Distance [km]')
        cf = ax[rik,ci].contour(distance_ctd[0,:] + dx/2. ,
                          ctd_data.z + dy/2., 
                          ctd_data.sigma_0.isel(xy=ind_CTD_section[ri]), levels = sigma_levels,colors='0.25')
    #return ax
    
def OLD_plot_map_view(sadcp, ctd_data, glider_track, ladcp_data, scanfish_data,scan_sadcp, ctd_ladcp,
                              topo=None,sst_map=None,sst_map1=None,ssh_name='sst',x_lim=[0,180]):
    
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cbook
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from oceans.datasets import etopo_subset
    from oceans.sw_extras import gamma_GP_from_SP_pt
    from matplotlib.patches import Polygon
    import gsw
    from matplotlib import animation, rc
    from IPython.display import HTML
    import pandas as pd

    # Stations map.

    fig, ax = plt.subplots(ncols = 2, sharey=True, figsize=(15,8))
    sst_range = np.array((15,19))
    ssh_range = np.array((-0.1,0.1))
    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,10,-27.5,
    lat_1, lat_2, lon_0, lat_0 =-25.,-27.5,13,-26,

    for si in np.array((0,1)): 
        # setup map
        m = Basemap(width=450000,height=300000,resolution='i',projection='aea',lon_0=lon_0,lat_0=lat_0)
        m.ax = ax[si]
        m.drawcoastlines(), m.fillcontinents(color='0.85')
        m.drawparallels(np.arange(-90.,91.,1.),labels=[True,False,False,False],dashes=[2,2]), m.drawmeridians(np.arange(-180.,181.,1.),labels=[False,False,False,True],dashes=[2,2])

        # add contour lines of bathymetry
        lon2, lat2 = np.meshgrid(topo.lon.values,topo.lat.values)
        #m.contourf(lon2, lat2,topo.Band1,40,cmap=plt.cm.Blues_r,latlon=True)
        m.contour(lon2, lat2,topo.Band1,20,colors='0.85',latlon=True)

        if si == 0:
            # === SST MAP ===
            lon2, lat2 = np.meshgrid(sst_map.lon.values,sst_map.lat.values)
            #m.contourf(lon2, lat2, sst_map.analysed_sst[0,:,:],40,cmap=plt.cm.coolwarm,latlon=True)
            sst_plt = m.pcolormesh(lon2, lat2, sst_map.sst,vmin=sst_range[0],vmax=sst_range[1],cmap=plt.cm.coolwarm,latlon=True)
            plt.text(1,1,sst_map.time_coverage_start[:-8],transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        
        else:
            # === SST/SSH MAP ===
            lon2, lat2 = np.meshgrid(sst_map1.lon.values,sst_map1.lat.values)
            #m.contourf(lon2, lat2, sst_map1.analysed_sst[0,:,:],40,cmap=plt.cm.coolwarm,latlon=True)
            ssh_plt = m.pcolormesh(lon2, lat2, sst_map1[ssh_name],vmin=ssh_range[0],vmax=ssh_range[1],cmap=plt.cm.coolwarm,latlon=True)
            plt.text(1,1,sst_map1.time_coverage_start[:-8],transform=m.ax.transAxes,horizontalalignment='right',verticalalignment='bottom')        
            # === SSH MAP ===
            #lon2, lat2 = np.meshgrid(ssh_map.lon.values,ssh_map.lat.values)
            #ssh_plt = m.contour(lon2, lat2, ssh_map.sla[0,:,:],vmin=ssh_range[0],vmax=ssh_range[1],colors='0.5',latlon=True) # 
            #plt.text(0,1,ssh_map.time_coverage_start[:-4],transform=m.ax.transAxes,horizontalalignment='left',verticalalignment='bottom')  
            if hasattr(sst_map1, 'ugos'):
                # add Geostrophic current vectors
                gos_plt = m.quiver(lon2, lat2,sst_map1.ugos,sst_map1.vgos)#,scale=700)
                # make quiver key.
                qk = plt.quiverkey(gos_plt, 0.1, 0.1, 1, '1 m/s', labelpos='W')


        # plot ship track from SADCP data
        m.plot(sadcp.lon.values, sadcp.lat.values,'-k', latlon=True)

        # plot ladcp/ctd stations
        m.plot(ctd_data.lon.values, ctd_data.lat.values, 'ko', latlon=True)
        #m.plot(ladcp_data.lon.values, ladcp_data.lat.values, 'b.', latlon=True)

        # plot glider track
        m.plot(glider_track[0,:].values, glider_track[1,:].values, color='0.75',lw=2, latlon=True)

        # plot sections that are used below
        for ri in range(len(ctd_ladcp)):
            m.plot(ctd_ladcp[ri].lon.values,ctd_ladcp[ri].lat.values, '.', color= 'chartreuse', latlon=True)

        for ri in range(len(scan_sadcp)):
            m.plot(scan_sadcp[ri].lon.values,scan_sadcp[ri].lat.values, lw = 4, color= 'SlateGrey', latlon=True)

        m.drawmapscale(14.5, -25.5, 15, -25.5, 50,barstyle='fancy')
            
        #if si == 0:
        #    plt.colorbar(sst_plt, cax=axins)#, ticks=[-0.5, 0, 0.5])
        #else:
        #    plt.colorbar(ssh_plt, cax=axins)#, ticks=[-0.5, 0, 0.5])

        if si == 0:
            # add inset showing globe
            add_globalmap_inset(m)
            
def OLDfftwaveplt_KE(scan_sadcp_in, depth_range_in=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import Denmark_Strait.src.spectra_and_wavelet_functions as sw
    import scipy as sp
    import scipy.stats as ss
    import pycwt as wavelet
    
    # based on plotting in SubEx_Paper4_Analysis.ipynb
    
    # Loop through each depth and get spectra of each; then average together
      
    # for plotting lines of uniform slope
    slope_factor = np.array((.000000005,.0000000005,.0000000000005)) # to fit on plot

    # Second setup values for Welch windowing
    # N sample series, divided into P segments of D samples with a shift S between adjacent segments
    probability = 0.95
    #D = 25

    fig, axs = plt.subplots(nrows=3,ncols=len(depth_range_in), sharex=True, figsize=(20,12)) #
    
    for di in (range(len(depth_range_in))):
        color_idx = np.linspace(0, 1, len(scan_sadcp_in)+1)
        for ci, ti in zip(color_idx, range(len(scan_sadcp_in))): # loop through transects
            # get range of depths to be used
            if isinstance(depth_range_in[di],slice): 
                depth_range = scan_sadcp_in[ti].z.sel(z=depth_range_in[di]).values # depth_range_in must be negative 
            else: # use full depth range 
                depth_range = scan_sadcp_in[ti].z.values

            N = scan_sadcp_in[ti].xy.shape[0]
            D = round(N/3)
            S = D*0.5 # 50% shift
            # Number of estimations
            P = int((N - D) / S) + 1
            # degrees of freedom
            dof = 2*P

            #color_idx = np.linspace(0, 1, scan_sadcp_in[ti].z.shape[0])
            for zi in depth_range: # loop through depths

                dx = scan_sadcp_in[ti].x_m.diff('xy').mean().values

                wavenumKE, psdKE = sp.signal.welch(scan_sadcp_in[ti].ke.sel(z=zi).dropna('xy'), 
                                               fs=1/dx, window="hanning", nperseg=D, noverlap=S, detrend="linear")
                # remove frequency = 0; otherwise get warning every loop
                #wavenumKE, psdKE = wavenumKE[1:], psdKE[1:]

                #axs[0].loglog(1/(wavenumKE),psdKE/1, alpha=0.25,color=plt.cm.coolwarm(ci),lw=1)

                # variance preserving
                #axs[1].semilogx(1/(wavenumKE), psdKE * wavenumKE, color=plt.cm.coolwarm(ci), alpha=0.25,lw=1)

                # global wavelet spectrum
                a1,a2 = (10**3,10**4)   
                dat_in = scan_sadcp_in[ti].ke.sel(z=zi).dropna('xy').reset_index('xy').swap_dims({'xy': 'x_m'})
                if dat_in.size > 0:
                    waveKE = sw.run_wavelet(dat_in,(a1,a2),time_name='x_m',period_name='wavelength')

                # The global wavelet spectra 
                #waveKE.glbl_signif.plot(x='period',ax=axs[2], linestyle='-', linewidth=1., alpha=0.25,color=plt.cm.coolwarm(ci)) 
                waveKE['glbl_power_var'] = waveKE.std2.values * waveKE.glbl_power
                #waveKE.glbl_power_var.plot(x='wavelength',ax=axs[2], linestyle='-', linewidth=1., alpha=0.25,color=plt.cm.coolwarm(ci)) 
                axs[2,di].set_xscale('log')
                axs[2,di].set_yscale('log')
                #axs[2].invert_xaxis()

                if zi==depth_range[0]:
                    axs[0,di].set_title('a) KE - PSD')
                    axs[1,di].set_title('b) KE - Variance-preserving plot')
                    axs[2,di].set_title('c) KE -  GWS')

                if zi==depth_range[0]:
                    psdKE_transect = psdKE
                    waveKE_transect = waveKE['glbl_power_var'].values

                if psdKE.size > 0:
                    psdKE_transect = np.vstack((psdKE_transect, psdKE)) 
                    wavenumKE_transect = wavenumKE
                    #print(waveKE['glbl_power_var'].shape,waveKE_transect.shape)
                    waveKE_transect = np.vstack((waveKE_transect, waveKE['glbl_power_var'].values))
                    waveKE_wavelength = waveKE.wavelength.values
                    waveKE_last = waveKE
                if zi==depth_range[0]:
                    dat_last = dat_in

            num_spectra_transect = psdKE_transect.shape[0]
            # get mean spectra
            psdKE_transect = psdKE_transect.mean(axis = 0)
            dof_transect = dof+num_spectra_transect
            #psdKE_lower_transect,psdKE_upper_transect = sw.get_fft_conf_interval(psdKE_transect,probability,dof_transect)
            # get psd value for different locations of confidence limit bar
            conf_x = 1/np.array((3*10**4,10**4,5*10**3,10**3,6*10**2))
            conf_y1, conf1 = np.zeros(conf_x.shape), np.zeros((2,conf_x.shape[0]))
            for tti in range(conf_x.shape[0]):
                conf_y1[tti] = psdKE_transect[np.abs(wavenumKE_transect-conf_x[tti]).argmin()]
                conf1[:,tti] = conf_y1[tti] * dof_transect / ss.chi2.ppf([1-probability, probability], dof_transect)

            axs[0,di].loglog(1/(wavenumKE_transect),psdKE_transect, alpha=0.85,color=plt.cm.viridis(ci),lw=2,label=('Transect ',str(ti+1)))
            axs[0,di].plot([1/conf_x, 1/conf_x], conf1, color=plt.cm.viridis(ci), lw=4.5,alpha=0.5)
            axs[0,di].plot(1/conf_x, conf_y1, color=plt.cm.viridis(ci), linestyle='none', lw=4.5, 
                    marker='_', ms=8, mew=2,alpha=0.5)
            sw.plot_loglog_slope(axs[0,di],np.array((-5/3,-2,-3)),1/(wavenumKE_transect),factor=slope_factor)
            print()
            #axs[0].plot(1/(wavenumKE_transect), psdKE_lower_transect/1, '--',color=plt.cm.viridis(ci),lw=2,alpha=0.5)
            #axs[0].plot(1/(wavenumKE_transect), psdKE_upper_transect/1, '--',color=plt.cm.viridis(ci),lw=2,alpha=0.5)
            # variance preserving
            axs[1,di].semilogx(1/(wavenumKE_transect), psdKE_transect * wavenumKE_transect, color=plt.cm.viridis(ci), alpha=0.85,lw=2)
            axs[1,di].plot([1/conf_x, 1/conf_x], conf1*conf_x, color=plt.cm.viridis(ci), lw=4.5,alpha=0.5)
            axs[1,di].plot(1/conf_x, conf_y1*conf_x, color=plt.cm.viridis(ci), linestyle='none', lw=4.5, 
                    marker='_', ms=8, mew=2,alpha=0.5)
            #axs[1].semilogx(1/(wavenumKE_transect), psdKE_lower_transect * wavenumKE_transect,'--', color=plt.cm.coolwarm(ci),lw=2,alpha=0.5)
            #axs[1].semilogx(1/(wavenumKE_transect), psdKE_upper_transect * wavenumKE_transect,'--', color=plt.cm.coolwarm(ci),lw=2,alpha=0.5)

            waveKE_transect = waveKE_transect.mean(axis = 0)
            axs[2,di].loglog(waveKE_wavelength,waveKE_transect, alpha=0.85,color=plt.cm.viridis(ci),lw=2)
            # Calculates the global wavelet spectrum and determines its significance level.
            std = dat_last.std()                      # Standard deviation
            std2 = std ** 2                      # Variance
            dx = np.diff(dat_last.x_m).mean() #dat_last.time_secs.diff('time_secs').mean('time_secs').values*10**(-9)/(60*60*24) # using time_secs which is an integer
            N = dat_last.shape[0]                          # Number of measurements
            mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6
            slevel = 0.95                        # Significance level
            alpha, _, _ = wavelet.ar1(dat_last.values)
            # !!! IS THIS CORRECT
            dof_transect = dof_transect - waveKE_last.scales #
            waveKE_glbl_signif, tmp = wavelet.significance(std2.values, dx, waveKE_last.scales, 1, alpha,
                                                    significance_level=slevel, dof=dof_transect,
                                                    wavelet=mother)
            waveKE_glbl_signif = N*dx*waveKE_glbl_signif/waveKE_last.scales # rpn !!! N*dt*   /scales to match glbl_power correction
            #axs[2].loglog(waveKE_wavelength,waveKE_glbl_signif, alpha=0.85,color=plt.cm.viridis(ci),lw=2,linestyle='--')


        axs[0,di].set_xlabel('Wavelength [m]')
        axs[1,di].set_xlabel('Wavelength [m]')
        axs[2,di].set_xlabel('Wavelength [m]')
        axs[0,di].set_ylabel('Power Spectral Density [m$^4$ s$^{-4}$]')
        axs[0,di].legend()
        axs[0,di].set_ylim([10**-2, 10**1])
        axs[0,di].set_xlim([10**2, 10**5])
        #axs[1].set_ylim([0, 10**-4])
        #axs[2].set_ylim([10**-5, 10**-1])
        axs[0,di].invert_xaxis()