"""
    Beta python scripts for paper
    "Reaction-drift-diffusion models from master equations: application to material defects"
    (c) TD Swinburne, CNRS / CINaM, 2021
    tomswinburne.github.io
"""
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import msmtools.analysis as mana

def ordev(M,vec=False,pi=None):
    """
        Wrapper to order and normalize scipy.linalg.eig results
        M : numpy matrix
            Square matrix to be diagonalized
    """
    if pi is None:
        _nu,_w,_v =  spla.eig(M,left=True)
        _nu = -_nu.real
        _w = _w[:,_nu.argsort()]
        _v = _v[:,_nu.argsort()]
        _nu = _nu[_nu.argsort()]
        _w = _w@np.diag(1.0/np.diag(_w.T@_v)) * _v[:,0].sum()
        _v /= _v[:,0].sum()

        if not vec:
            return _nu
        else:
            return _nu,_w,_v
    else:
        assert pi.size==M.shape[0],"Dimension Mismatch!"
        rPi = np.diag(np.sqrt(pi))
        riPi = np.diag(1.0/np.sqrt(pi))
        nu,tw = np.linalg.eigh(-riPi@M@rPi)
        tw = tw[:,nu.argsort()]
        nu = nu[nu.argsort()]
        if not vec:
            return nu
        w = riPi@tw
        v = rPi@tw
        return nu,w,v


def generate_random_blocks( N=10,
                            mean_bar = [0.1,1.0],
                            std_bar = 0.3, #
                            std_min = 0.0, #
                            gen = np.random.uniform, #
                            seed = 123, # s
                            connectivity=3, # average number of connections
                            sub_basin=False, # sub-super-basins in cell?
                            sub_basin_barrier = 4.0, # inter-sub-basin transitions
                            sub_basin_migration = 0.0, # self migration barrier
                            sub_basin_height = 0.0, # relative heights
                            sub_basin_size=2, # size of "high" basin
                            sub_basin_aniso=False # sub-basin-dependent migration
                          ):
    """
        Helper function to generate random periodic transition rate matricies
        N : int
            number of states
        mean_bar : list, size dim+1
            mean barrier for Q(0), Q(x), Q(y)...
        std_bar : float
            std dev for barrier energies
        std_min : float
            std dev for minima energies
        gen : numpy.random distribution
            barrier distribution law
        seed : int
            random number seed
        sub_basin : bool
            If true, split cell into two sub-super-basins.
            To produce multiple coarse-grained states
    """
    np.random.seed(seed)
    """
        Detailed balance:
        C@Pi = (C@Pi).T
        L@Pi = (R@Pi).T
    """
    dim = len(mean_bar)-1


    pi = np.exp(-gen(size=N)*std_min)

    if sub_basin:
        pi[-sub_basin_size:] *= np.exp(-sub_basin_height) # scale is small
    pi /= pi.max()
    Pi = np.diag(pi)
    iPi = np.diag(1.0/pi)


    # intercell
    CPi = gen(size=(N,N))*std_bar
    CPi -= CPi.mean()
    CPi = np.exp(-CPi-mean_bar[0]) + np.exp(-CPi-mean_bar[0]).T
    if sub_basin:
        CPi[-sub_basin_size:,:][:,:sub_basin_size] *= np.exp(-sub_basin_barrier)
        CPi[:sub_basin_size,:][:,-sub_basin_size:] *= np.exp(-sub_basin_barrier)
    C = CPi@iPi

    kt = C.sum(axis=0)

    L,R = [],[]

    for _dim,mu in enumerate(mean_bar[1:]):
        _L = gen(size=(N,N))*std_bar
        _L -= _L.mean()
        if connectivity<N:
            _L = np.triu(np.exp(-_L-mu),max(0,N-connectivity))
        else:
            _L = np.exp(-_L-mu)

        if sub_basin_aniso:
            if _dim==0:
                ll = _L[-sub_basin_size:,:][:,-sub_basin_size:].shape
                sub_basin_k = np.exp(-0.1*np.random.uniform(size=ll))
                sub_basin_k *= np.exp(-sub_basin_migration)
                _L[-sub_basin_size:,:][:,-sub_basin_size:] = sub_basin_k
            else:
                ll = _L[:sub_basin_size,:][:,:sub_basin_size].shape
                sub_basin_k = np.exp(-0.1*np.random.uniform(size=ll))
                sub_basin_k *= np.exp(-sub_basin_migration)
                _L[:sub_basin_size,:][:,:sub_basin_size] = sub_basin_k
        else:
            if sub_basin_migration:
                _L[-sub_basin_size:,:][:,-sub_basin_size:] = sub_basin_k

        L += [_L]
        R += [(_L@Pi).T @ iPi]
        kt += L[-1].sum(axis=0)+R[-1].sum(axis=0)
    C -= np.diag(kt)

    M = np.zeros(shape=(3*N,(2*(dim>1)+1)*N))
    M[N:2*N,:] += np.hstack((L[0],C,R[0]))
    if dim>1:
        M[:,N:2*N] += np.hstack((L[1],np.zeros(C.shape),R[1])).T
    M -= np.diag(np.diag(M))

    sel = M<1.0e-6
    M[M==0.0] = 1.0
    M = -np.log(M)
    M[sel] = np.nan
    return C,L,R,M,pi/pi.sum()

def bloch_mat(C,L,R,k):
    """
        Generate bloch matrices from paper
    """
    if len(k)!=len(L):
        print("ERROR")
    M = C.copy() * complex(1.0,0)
    for i,ki in enumerate(k):
        M += np.exp(+complex(0,1) * ki )*L[i]
        M += np.exp(-complex(0,1) * ki )*R[i]
    return M

def bloch_spectrum(C,L,R,Bx=101,By=None):
    """
        Find full bloch spectrum from result in paper
    """
    if By is None:
        By = Bx
    k_val_vec = []

    N = C.shape[0]

    if len(L)==1:
        k_a = np.linspace(-np.pi,np.pi,Bx)
    else:
        k_a = np.meshgrid(np.linspace(-np.pi,np.pi,Bx),np.linspace(-np.pi,np.pi,By))
        k_a = np.stack((k_a[0].flatten(),k_a[1].flatten())).T

    all_data = []
    for k in k_a:
        M = bloch_mat(C,L,R,k=k)
        nu,w,v = ordev(M,vec=True)
        all_data += [np.vstack((np.outer(k,np.ones(N)),nu.reshape((1,-1)),w,v))]#,w,v))]
    all_data = np.r_[all_data]
    return all_data


def plot_model(fig,ax,M,my_cmap = 'cividis',lim=None):
    """
        Plot 2D model, setting zeros to nans so they are not shown
    """
    sel = (~np.isnan(M))
    if lim is None:
        lim = [M[sel].min(),M[sel].max()]
    im = ax.matshow(M,cmap=plt.get_cmap(my_cmap),vmin=lim[0],vmax=lim[1])
    cbar = fig.colorbar(im)
    cbar.set_ticks([])
    cbar.set_label('Energy Barrier [Arb.]',fontsize=9)
    ax.set_axis_off()
    return lim

def plot_spectrum_3d(fig,ax,ev_data,my_cmap = 'cividis',banded=False):
    """
        3D scatter plot of band spectrum
    """
    ax.view_init(elev=30., azim=45)
    kx = ev_data[:,0,:].flatten()
    ky = ev_data[:,1,:].flatten()
    nu = ev_data[:,2,:].flatten()

    if not banded:
        color = nu/nu.max()
    else:
        band = np.zeros(ev_data[:,2,:].shape)
        for i in range(ev_data.shape[2]):
            band[:,i] = i
        color = band.flatten()/band.max()

    ax.scatter(kx,ky,nu,c=color,s=1,cmap=my_cmap)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_facecolor('white')
    ax.set_xlabel(r'$\bf{k}\cdot\hat{\bf{x}}$',labelpad=-10,fontsize=8)
    ax.set_ylabel(r'$\bf{k}\cdot\hat{\bf{y}}$',labelpad=-10,fontsize=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'Eigenvalue $\lambda({\bf{k}})$',labelpad=-14,fontsize=8,rotation=90)
    ax.set_zlim(zmin=0)



def plot_spectrum(fig,ax,
                  ev_data, # true k, ev data
                  approx, # approx_ev function
                  my_cmap = 'cividis',
                  plot_approx=0, # plot long wavelength eigenvalue approximation
                  plot_ksel=False, # plot "long" wavelengths
                  l_homog=2.0, # homogenization length
                  band_key=True, # Show band legend
                  show_band_gap=0): # Show band gap
    """
        Produce spectrum plots
        plot_approx=0
            plot long wavelength eigenvalue approximation
        plot_ksel=False
            plot "long" wavelengths
        l_homog=2.0
            homogenization length
        band_key=True
            Show band legend
        show_band_gap=0
            Show band gap
    """

    """
        Band gap strip
    """
    all_ev = np.sort(ev_data[:,2,:].flatten())
    x = np.linspace(0,all_ev.size,2)*1.0
    dx = x.max()*0.05
    x[0]-=dx
    x[1]+=dx

    ax.set_xlim(x.min(),x.max())
    if show_band_gap>0:
        sel = ev_data[:,2,0]>=0.
        if plot_ksel:
            sel = np.max(np.abs(ev_data[:,:2,show_band_gap-1]),axis=1)<np.pi/l_homog
        yb = np.ones(x.size) * ev_data[sel,2,show_band_gap-1].max().real

        if plot_ksel:
            sel = np.max(np.abs(ev_data[:,:2,show_band_gap]),axis=1)<np.pi/l_homog
        ya = np.ones(x.size) * ev_data[sel,2,show_band_gap].min().real

        ax.fill_between(x,yb,ya,facecolor='b',alpha=0.25,label='Gap $\Delta$')

    if plot_ksel:
        _sel_lab = r'$|\mathbf{k}\cdot\mathbf{1}|_{\infty}\geq\pi/L$'

    sel_lab=None


    """
        Band-colored eigenvalues
    """
    band_lab=None
    sel_ev = []
    for i,c in enumerate(np.linspace(0.,1.0,ev_data.shape[2])):
        ev = ev_data[:,2,i]
        if band_key:
            band_lab = r'$\lambda_{%d\mathbf{k}}$' % i
        ax.scatter(np.searchsorted(all_ev,ev).real,ev.real,color=plt.get_cmap(my_cmap)(c),s=8.0,label=band_lab)
        if plot_ksel:
            sel = np.abs(ev_data[:,:2,i]).min(axis=1)>np.pi/l_homog
            if i==ev_data.shape[2]-1:
                sel_lab=_sel_lab

            ax.scatter(np.searchsorted(all_ev,ev)[sel].real,ev[sel].real,color='0.5',label=sel_lab,s=8.0) # s=4.0,


    """
        Long wavelength approximation to eigenvalues
    """
    if plot_approx>0:
        for i,c in enumerate(np.linspace(0.,1.0,ev_data.shape[2])):
            a_ev = np.sort(approx(ev_data[:,:2,0].real,i=i)).real
            ll = None
            if i==0:
                ll = r'$\lambda^0_{m\mathbf{k}}$'
            ax.plot(np.searchsorted(all_ev,a_ev),a_ev,color='r',label=ll,lw=1.5) # ,s=4.0,marker='D'
            if plot_approx==1:
                break

    ax.legend(fontsize=8)
    ax.set_ylabel(r'Eigenvalue $\lambda({\bf{k}})$',fontsize=8)
    ax.set_yticks([])
    ax.set_xticks([])


def coarse_grain(C,L,R,ev_data,reduced_modes=1,return_Q=False,pi=None):
    """
        Perform coarse graining following paper
    """
    M = C.copy()
    for idim in range(len(L)):
        M += L[idim]+R[idim]
    nu_0,w,v = ordev(M,vec=True,pi=pi)

    def approx_ev(k,i=0):
        """
            Calculate approximate eigenvalues from results in paper
            k = vector of wavevectors
        """
        eva = np.zeros(k.shape[0],np.complex128)
        for idim in range(len(L)):
            eva += -w[:,i]@L[idim]@v[:,i] * np.exp(complex(0,+1)*k[:,idim])
            eva += -w[:,i]@R[idim]@v[:,i] * np.exp(complex(0,-1)*k[:,idim])
        eva += -w[:,i]@C@v[:,i]
        return eva.real

    N = M.shape[0]
    rN = reduced_modes
    W = w[:,:rN]
    V = v[:,:rN]
    rQ = V@np.diag(np.exp(-nu_0[:rN]))@W.T

    """
        find CP linear combination matrix from PCCA+ membership functions
    """
    CP = V.T@mana.pcca_memberships(rQ.T, reduced_modes)
    iCP = np.linalg.inv(CP)

    """
        Form reduced rate matricies
    """
    rC = CP@W.T@C@V@iCP
    #rC -= np.diag(np.diag(rC))
    rL = [CP@W.T@L[0]@V@iCP,CP@W.T@L[1]@V@iCP]
    rR = [CP@W.T@R[0]@V@iCP,CP@W.T@R[1]@V@iCP]
    rkt = (rC+rL[0]+rL[1]+rR[0]+rR[1]).sum(axis=0)
    rC -= np.diag(rkt)

    rB = np.zeros((rN,rN))
    rM = np.block([[rB,rR[1],rB],
                   [rL[0],rC,rR[0]],
                   [rB,rL[1],rB]])
    lrM = np.where(np.abs(rM)<=1e-13,np.nan,-np.log(rM))
    if return_Q:
      rC -= np.diag((rC+rL[0]+rL[1]+rR[0]+rR[1]).sum(axis=0))
      rQ = np.block([[rB,rR[1],rB],
                   [rL[0],rC,rR[0]],
                   [rB,rL[1],rB]])
      return lrM, approx_ev, rQ
    else:
      return lrM, approx_ev
