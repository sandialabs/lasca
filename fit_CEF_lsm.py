"""
res_dir: folder to put results. 
"""

import re
import os
import numpy as np
import math
from shutil import copyfile,rmtree
import copy
from sympy import symbols, solve, simplify, sympify, preorder_traversal, Float, log, diff, apart
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from sympy import latex

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
	else:
		print('---  There is this folder!  ---')

r = 8.617333262E-5; # Boltzman constant in eV/K.
R=8.314 # gas constant in J / mol·K
emax=2.99 # highest O stoichiometry used for fitting.
color_table=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']+['xkcd:salmon','xkcd:light green','xkcd:yellow'
    ,'xkcd:navy','xkcd:light pink','xkcd:gold','xkcd:neon pink','xkcd:baby blue','xkcd:forrest green'
    ,'xkcd:cobalt blue','xkcd:reddish pink','xkcd:hot purple','xkcd:ecru','xkcd:rust red'
    ,'xkcd:fern green','xkcd:rose red']

y0_values=[0,0.1,0.2,0.3,0.4,0.5]
explore_alpha = True # whether to use crossvalidation to identify optimal alpha
#y0_values=[0,0.2,0.5]

# read Gibbs energy expression from file
cef_results='CEF_LaMnO'
with open(cef_results,'r') as f:
	c=f.readlines()

for i in range(len(c)):
	if 'Nonconfigurational-Gibbs energy with T expansion:' in c[i]:
		str_G_ncf=c[i+1].strip('\n')
	if 'Total Gibbs energy with T expansion:' in c[i]:
		str_G=c[i+1].strip('\n')
	if 'Independent site fractions:' in c[i]:
		str_y=c[i+1].strip('\n')
	if 'Endmember parameters:' in c[i]:
		str_a=c[i+1].strip('\n')
	if 'Independent excess parameters:' in c[i]:
		str_L=c[i+1].strip('\n')
	if 'Total Gibbs energy without expansion:' in c[i]:
		str_G_unexpanded=c[i+1].strip('\n')
	if 'L_names:' in c[i]:
		str_L_names=c[i+1].strip('\n')
	if 'a_names:' in c[i]:
		str_a_names=c[i+1].strip('\n')

latex_G_unexpanded=latex(str_G_unexpanded)
L_names=eval(str_L_names)
a_names=eval(str_a_names)
param_names={**L_names, **a_names}

sy=str_y.strip('[').strip(']').split(', ')
y=[symbols('y'+str(i)) for i in range(len(sy))]
a=symbols(str_a.strip('[').strip(']').replace(',',' '))
L=symbols(str_L.strip('[').strip(']').replace(',',' '))
T=symbols('T')

dreplace={}
for i in range(len(sy)):
	dreplace[sy[i]]=str(y[i])

for key,value in dreplace.items():
	str_G_ncf=str_G_ncf.replace(key,value)
	str_G=str_G.replace(key,value)

G_ncf=sympify(str_G_ncf)
G=sympify(str_G)

# fix to reference perfect compound T-dependence
G=G.subs(symbols('a_0_1'),0).subs(symbols('a_0_2'),0)
G_ncf=G_ncf.subs(symbols('a_0_1'),0).subs(symbols('a_0_2'),0)

G_ncf_0K=G_ncf.subs(log(T),0).subs(T,0)
G_ncf_0K=simplify(G_ncf_0K.expand())

# read 0K DFT data
data=np.loadtxt('data')
data_0K=data[data[:,-2]==0]

# lasso feature selection
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes

L_features=[diff(G_ncf_0K,i) for i in L]
a_features=[diff(G_ncf_0K,i) for i in a]

X=[]
f=[]
for i in range(data_0K.shape[0]):
	y_values=data_0K[i,:][0:len(y)]
	d_y_subs={}
	for j in range(len(y)):
		d_y_subs[y[j]]=y_values[j]
	L_features_values=[j.subs(d_y_subs) for j in L_features]
	a_features_values=[j.subs(d_y_subs) for j in a_features]
	X.append(L_features_values+a_features_values)
	f.append(data_0K[i,:][-1])

# read O2 Gibbs energy
mu_ft=np.loadtxt('O2_janaf')
T_O2=mu_ft[:,0]
S_O2=mu_ft[:,2]
H_O2=mu_ft[:,4]
G_O2=H_O2*1000-T_O2*S_O2 # J/mol O2
G_O2=G_O2/(1.6e-19)/(6.02e23) # eV per O2
G_O2=G_O2-G_O2[0]
G_O=G_O2/2

from scipy import interpolate
G_O_ip=interpolate.interp1d(T_O2, G_O)

for n in range(len(y0_values)):
    if y0_values[n]!=0:
        Ga_op=G.subs({y[1]*log(y[1]):0,y[2]*log(y[2]):0}).subs({y[0]:y0_values[n],y[1]:0,y[2]:0})
    else:
        Ga_op=G.subs({y[0]*log(y[0]):0,y[1]*log(y[1]):0,y[2]*log(y[2]):0}).subs({y[0]:0,y[1]:0,y[2]:0})
    mu_O_op=-diff(Ga_op,y[-1])/3
    mu_O_op=simplify(mu_O_op.expand())
    L_features_op=[diff(mu_O_op,i) for i in L]
    a_features_op=[diff(mu_O_op,i) for i in a]
    d_zeros={}
    for i in L:
        d_zeros[i]=0
    for i in a:
        d_zeros[i]=0
    c_op=mu_O_op.subs(d_zeros) # terms without L and a
    import glob
    fexpt0=glob.glob('expt_*')
    fexpt=[]
    for j in range(len(fexpt0)):
        with open(fexpt0[j],'r') as files:
            if float(files.readlines()[0].strip('#'))==y0_values[n]:
                fexpt.append(fexpt0[j])
    expt=[np.loadtxt(i) for i in fexpt]
    for j in range(len(expt)):
        expt_current=expt[j][expt[j][:,1]<emax]
        #expt_current=expt_current[(expt_current[:,2]==873) | (expt_current[:,2]==1273)] # filter T
        for i in range(expt_current.shape[0]):
            d_subs={T:expt_current[i,2],y[-1]:(3-expt_current[i,1])/3}
            L_features_op_values=[k.subs(d_subs) for k in L_features_op]
            a_features_op_values=[k.subs(d_subs) for k in a_features_op]
            X.append(L_features_op_values+a_features_op_values)        
            mu=1/2*r*expt_current[i,2]*np.log(expt_current[i,0])+G_O_ip(expt_current[i,2])
            #mu=1/2*r*expt_current[i,2]*np.log(expt_current[i,0])
            f.append(mu-c_op.subs(d_subs)) 

X=np.array(X)
f=np.array(f)
scaler = StandardScaler(with_mean=False)
scaler.fit(X)
X_st=scaler.transform(X)

features = [str(i) for i in L]+[str(i) for i in a]

if explore_alpha:
    # plot alpha-importance diagram
    print("\nGenerating alpha-importance diagram:")
    clf = Lasso(alpha=1e-7,tol=1e-13,max_iter=int(1e7),fit_intercept=False)
    clf.fit(X_st, f)
    importance0 = np.abs(clf.coef_)

    log_alpha=[-11+0.2*i for i in range(56)]
    importances=[]
    for i in log_alpha:
        print(f"Processing alpha = {10**i}")
        clf = Lasso(alpha=10**(i),tol=1e-13,max_iter=int(1e7),fit_intercept=False)
        clf.fit(X_st, f)
        importance = np.abs(clf.coef_)
        importances.append(importance)

    importances=np.array(importances)
    log_alpha=np.array(log_alpha)
    log_alpha=log_alpha.reshape(-1,1)
    aim=np.hstack([log_alpha,importances])
    np.savetxt('aim',aim,fmt="%.15f  ",header='log_alpha param1 param2 ...')
    # aim=np.loadtxt('aim')
    log_alpha,importances=aim[:,0],aim[:,1:]

    plt.cla()
    ia,iL=0,0
    for i in range(importances[:,importance0 > 0].shape[1]):
        label='$\mathrm{'+param_names[np.array(features)[importance0 > 0][i]]+'}$'
        #label='$'+param_names[np.array(features)[importance0 > 0][i]]+'$'
        if 'L_' in label:
            linestyle="dashdot"
            iL=iL+1;icolor=iL
        else:
            linestyle="solid"
            ia=ia+1;icolor=ia
        plt.plot(log_alpha,importances[:,importance0 > 0][:,i],label=label,linestyle=linestyle,color=color_table[icolor])

    plt.xlabel("$\mathregular{logα}$",fontsize=15)
    plt.ylabel('Importance',fontsize=15)
    #plt.subplots_adjust(left=0.07, right=0.85, bottom=0.1, top=0.93)
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.1, top=0.93)
    #plt.legend(handlelength=2.5,bbox_to_anchor=(1.01, 1.05), loc="upper left",frameon=False,ncol=1,columnspacing=1,fontsize=7)
    plt.legend(handlelength=2.5,bbox_to_anchor=(0.29, 1.0), loc="upper left",frameon=False,ncol=3,columnspacing=1,fontsize=9)
    #plt.legend(loc="lower left", bbox_to_anchor=(0.0,1.02), ncols=3) 
    #plt.show()
    plt.ylim([0,6])
    plt.xlim([-7,0])
    plt.savefig('aim.png',dpi=300, bbox_inches='tight')

    #CV
    from sklearn.linear_model import LassoCV

    nfolds=[5,10,20,50,100]
    lasso=['' for i in range(len(nfolds))]
    for i in range(len(nfolds)):
        lasso[i] = LassoCV(cv=nfolds[i],eps=1e-8).fit(X_st, f)

    for i in range(len(nfolds)):
        xmin, xmax = 1e-7, 1
        #plt.plot(lasso.alphas_, lasso.mse_path_, linestyle=":")
        plt.plot(
            lasso[i].alphas_,
            lasso[i].mse_path_.mean(axis=-1),
            color=color_table[i],
            label="MSE, "+str(nfolds[i])+'-fold',
            linewidth=2,
        )
        plt.axvline(lasso[i].alpha_, linestyle="--", color=color_table[i], label=r"$\alpha_\mathrm{opt}$, "+str(nfolds[i])+'-fold')

    plt.xlim(xmin, xmax)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\alpha$",fontsize=15)
    plt.ylabel(r"Mean square error (eV$^2$)",fontsize=15)
    plt.subplots_adjust(left=0.09, right=0.81, bottom=0.1, top=0.93)
    plt.legend(frameon=False,bbox_to_anchor=(1.0, 0.98), loc="upper left",prop={'size': 10})
    #plt.show()
    plt.savefig('cv.png',dpi=300, bbox_inches='tight')

# do lasso at the chosen alpha 
clf = Lasso(alpha=1e-4,tol=1e-13,max_iter=int(1e7),fit_intercept=False)
clf.fit(X_st, f)
print(clf.coef_)
print(clf.intercept_)

importance0 = np.abs(clf.coef_)
nzp=[param_names[i] for i in np.array(features)[importance0 > 0]]
np.array(features)[importance0 == 0]


# get fitted formula
d_subs={}
for i in range(len(features)):
	d_subs[symbols(features[i])]=clf.coef_[i]/scaler.scale_[i]

d_subs_names={}
for i in range(len(features)):
	if importance0[i] > 0:
		d_subs_names[param_names[features[i]]]=clf.coef_[i]/scaler.scale_[i]


G_ncf_0K=G_ncf_0K.subs(d_subs)
G_ncf_0K=simplify(G_ncf_0K.expand())

G=G.subs(d_subs)
G=simplify(G.expand())

S=-diff(G,T)
H=G+T*S
H=simplify(H.expand())

# O chemical potential
mu_O=-diff(G,y[-1])/3
mu_O=simplify(mu_O.expand())

# compare DFT data
from fractions import Fraction
def fr(dec):
    return str(round(dec,4))

energies_fit=[]
energies_dft=data_0K[:-1,-1] # exclude full-vacancy
nat=[]
label=['' for i in energies_dft]
for i in range(data_0K.shape[0]-1):
    y_values=data_0K[i,:][0:len(y)]
    nat.append(5-y_values[1]-y_values[2]-3*y_values[3])
    d_y_subs={}
    for j in range(len(y)):
        d_y_subs[y[j]]=y_values[j]
    energies_fit.append(G_ncf_0K.subs(d_y_subs))
    #label[i]='(La'+fr(1-y_values[0]-y_values[1])+'Sr'+fr(y_values[0])+'Va'+fr(y_values[1])+')(Mn'+fr(1-y_values[2])+'Va'+fr(y_values[2])+')(O'+fr(1-y_values[3])+'Va'+fr(y_values[3])+')3'
    label[i]='(La'+fr(1-y_values[0]-y_values[1])+'Sr'+fr(y_values[0])+')(Mn'+fr(1-y_values[2])+')(O'+fr(1-y_values[3])+')3'
    label[i]=label[i].replace('1.0','').replace('Sr0.0','').replace('La0.0','').replace('Va0.0','').replace('Mn0.0','').replace('O0.0','').replace('()','(Va)')
    label[i]=label[i].replace('625','Sr0.0625')

energies_fit=np.array(energies_fit)
nat=np.array(nat)
#plt.scatter(energies_dft/nat,energies_fit/nat)
energies_dft1=energies_dft[energies_dft[:]<-14]
energies_dft2=np.hstack([energies_dft[abs(energies_dft[:]+14.77919746)<1e-5],energies_dft[energies_dft[:]>=-14]])
energies_fit1=energies_fit[energies_fit[:]<-14]
energies_fit2=np.hstack([energies_fit[abs(energies_fit[:]+14.7522349157274)<1e-5],energies_fit[energies_fit[:]>=-14]])

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
for i in range(len(energies_dft1)):
    ax.scatter(energies_dft1[i],energies_fit1[i],color=color_table[i],label=label[list(energies_fit).index(energies_fit1[i])])

x=np.linspace(-15,-14)
ax.plot(x,x,'k--')
plt.ylim([-15,-14])
plt.xlim([-15,-14])
plt.xlabel("Energy, DFT (eV/formula)",fontsize=15)
plt.ylabel('Energy, fitted (eV/formula)',fontsize=15)
plt.legend(loc="lower right",prop={'size': 7},frameon=False)
plt.text(0.1,0.9,'(a)',transform=ax.transAxes)

ax = fig.add_subplot(1, 2, 2)
for i in range(len(energies_dft2)):
    ax.scatter(energies_dft2[i],energies_fit2[i],color=color_table[i],label=label[list(energies_fit).index(energies_fit2[i])])

x=np.linspace(-15,7)
ax.plot(x,x,'k--')
plt.ylim([-15,7])
plt.xlim([-15,7])
plt.xlabel("Energy, DFT (eV/formula)",fontsize=15)
plt.ylabel('Energy, fitted (eV/formula)',fontsize=15)
plt.legend(loc="lower right",prop={'size': 7},frameon=False)
plt.text(0.1,0.9,'(b)',transform=ax.transAxes)

#plt.yscale('log')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13, top=0.93, wspace=0.3, hspace=0.1)
#plt.legend(frameon=False,prop={'size': 11},loc=(1,0.1))
#plt.legend(bbox_to_anchor=(1.01, 0.9), loc="upper left",frameon=False,ncol=2,columnspacing=1)
#plt.show()
plt.savefig('dft.png',dpi=300, bbox_inches='tight')


# calculate thermodynamic properties
def thermocalc(temperature,n):
    print('temperature: '+str(temperature)+' K, n='+str(n))
    def g(x):
        T=temperature
        #"""lsm
        y0,y1,y2,y3=x
        ntot=5-y1-y2-3*y3
        g=eval(str(G))/ntot
        return g
    def muo(x):
        T=temperature
        y0,y1,y2,y3=x
        mu=eval(str(mu_O))
        return mu
    nl=len(y)
    def ef(n,d,x0=[1e-2 for i in range(nl)]):
        #bnds = tuple([(1e-15, 1-1e-15) for i in range(nl)])
        #bnds = tuple([(1e-15, 1-1e-15),(1e-15, 1e-2),(1e-15, 1e-2),(1e-15, 1-1e-15)])
        #cons=({'type': 'eq', 'fun': lambda x:  x[0]/(1-x[1])-n },
        #    {'type': 'eq', 'fun': lambda x:  x[1]-x[2] },
        #    {'type': 'eq', 'fun': lambda x:  3*(1-x[3])-3+d })
        #res=minimize(g, x0=x0, bounds=bnds, method='SLSQP',constraints=cons,tol=1e-15)
        resx=[n,0,0,d/3]
        gg=g(resx)
        mu=muo(resx)
        pO2=(mu-G_O_ip(temperature))/(1/2*r*temperature) # ln(p/p0)
        #pO2=mu/(1/2*r*temperature) # ln(p/p0)
        pO2=pO2/(np.log(10))
        return resx,float(gg),float(mu),float(pO2)
    dmax=0.35
    #dmax=0.1 
    dmin=0
    dstep=0.01
    #dstep=0.05
    #d=[dmin+1e-3]+[dmin+i*dstep for i in range(1,int((dmax-dmin)/dstep)+1)]
    d=[dmin+1e-5]+[dmin+i*dstep for i in range(1,int((dmax-dmin)/dstep)+1)]+[i*1e-3 for i in range(1,10)]
    d.sort()
    dd=[0 for i in d]
    x=[[] for i in d]
    gg=[0 for i in d]
    mu=[0 for i in d]
    pO2=[0 for i in d]
    for i in range(len(d)):
        if i==0:
            x[i],gg[i],mu[i],pO2[i]=ef(n,d[i],x0=[1e-2 for i in range(nl)])
        else:
            x[i],gg[i],mu[i],pO2[i]=ef(n,d[i],x0=np.array(x[i-1])+1e-4)
            print(x[i]);print(mu[i]);print(pO2[i])
        dd[i]=3-(3-d[i])/(1-x[i][2]) # consider cation vacancy when calculating delta
    return dd,x,gg,mu,pO2


import math
from monty.serialization import loadfn,dumpfn


y0_values=[0,0.1,0.2,0.3,0.4,0.5]
temperatures=[873+100*i for i in range(10)]

res_dir='results_wexpt/'
#res_dir='results_025/'
#res_dir='results_dft/'
mkdir(res_dir)
results=[[] for m in range(len(y0_values))]
for m in range(len(y0_values)):
#for m in [1,5]:
    if not os.path.exists(res_dir+'results_'+str(y0_values[m])+'.json'):
        results[m]=[thermocalc(i,n=y0_values[m]+1e-5) for i in temperatures]
        #results[m]=[thermocalc(i,n=y0_values[m]) for i in temperatures]
        d=[i[0] for i in results[m]]
        x=[i[1] for i in results[m]]
        gg=[i[2] for i in results[m]]
        mu=[i[3] for i in results[m]]
        pO2=[i[4] for i in results[m]]
        res={'d':d,'x':x,'gg':gg,'mu':mu,'pO2':pO2}
        dumpfn(res,res_dir+'results_'+str(y0_values[m])+'.json')

# plot x(O)-p(O2)
# res_dir=['results_wexpt/','results_025/','results_dft/']
res_dir=['results_wexpt/']
#res_dir=['results_dft/']
la=[' K, calc., full data',' K, calc., reduced data',' K, calc., DFT only']
linestyles=['solid','dotted',"dashed"]

plt.cla()
#fig = plt.figure(figsize=(2*5,math.ceil(len(y0_values)/2)*5))
fig = plt.figure(figsize=(15,15))
handles, labels = [], []
for m in range(len(y0_values)):
#for m in [1,5]:
    #ax = fig.add_subplot(2, math.ceil(len(y0_values)/2), m+1)
    ax = fig.add_subplot(math.ceil(len(y0_values)/2),2, m+1)
    for k in range(len(res_dir)):
        res = loadfn(res_dir[k]+'results_'+str(y0_values[m])+'.json')
        d=res['d'];x=res['x'];gg=res['gg'];mu=res['mu'];pO2=res['pO2']
        for i in range(len(temperatures)):
            ntot=[5-x[i][j][1]-x[i][j][2]-3*x[i][j][3] for j in range(len(x[i]))]
            x_O=[(3-3*x[i][j][3])/ntot[j] for j in range(len(x[i]))]
            ax.plot(pO2[i],x_O,label=str(temperatures[i])+la[k],color=color_table[i],linestyle=linestyles[k])
    import glob
    fexpt0=glob.glob('expt_*')
    names0=[i.split('_')[1].capitalize() for i in fexpt0]
    names0=[''.join(i for i in j if not i.isdigit()) for j in names0]
    names = []
    null=[names.append(i) for i in names0 if not i in names]
    fexpt=[]
    for j in range(len(fexpt0)):
        with open(fexpt0[j],'r') as files:
            if float(files.readlines()[0].strip('#'))==y0_values[m]:
                fexpt.append(fexpt0[j])
    expt=[np.loadtxt(i) for i in fexpt]
    shape=['o','s','v','^','+','*']
    for i in range(len(temperatures)):
        for j in range(len(expt)):
            try:
                expt_current=expt[j][expt[j][:,2]==temperatures[i]]
                expt_current=expt_current[expt_current[:,1]<emax]
                if len(expt_current)!=0:
                    name=''.join(i for i in fexpt[j] if not i.isdigit()).replace('_','., ').replace('expt., ','').capitalize()
                    label=str(temperatures[i])+' K, expt., '+name
                    plt.scatter(np.log10(expt_current[:,0]),expt_current[:,1]/(expt_current[:,1]+2),marker=shape[names.index(name)],label=label,color=color_table[i])
            except:
                pass
    plt.text(0.83,0.1,'b='+str(y0_values[m]),transform=ax.transAxes)
    plt.ylim([0.57,0.6])
    plt.xlim([-40,5])
    if m/2>=2:
        plt.xlabel("$\mathregular{logP_{O_2}}$ (bar)",fontsize=15)
    if m%2==0:
        plt.ylabel('Mole fraction O',fontsize=15)
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

handles_labels = dict(zip(labels, handles))
#fig.legend(handles_labels.values(), handles_labels.keys(), loc="upper right",bbox_to_anchor=(0.97, 0.91),prop={'size': 11},frameon=False)
fig.legend(handles_labels.values(), handles_labels.keys(), loc="upper right",bbox_to_anchor=(0.97, 0.91),frameon=False)
plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9, wspace=0.2, hspace=0.1)
#plt.show()
plt.savefig('main.png',dpi=300, bbox_inches='tight')

# calculate reduction enthalpy and entropy
ho=[[[] for m in range(len(y0_values))] for k in range(len(res_dir))] #reduction enthalpy 
so=[[[] for m in range(len(y0_values))] for k in range(len(res_dir))] #eduction entropy
for m in range(len(y0_values)):
    for k in range(len(res_dir)):
        res = loadfn(res_dir[k]+'results_'+str(y0_values[m])+'.json')
        d=res['d'];x=res['x'];gg=res['gg'];mu=res['mu'];pO2=res['pO2']
        for j in range(0,len(d[0])):
            lnp=[0.5*pO2[i][j]*np.log(10) for i in range(len(temperatures))]
            hoy,soy = np.polyfit(1/np.array(temperatures),lnp, 1)
            hoy=-hoy*R/1000 # kJ/mol
            soy=soy*R # J/K/mol
            ho[k][m].append(hoy)
            so[k][m].append(soy)

# plot reduction enthalpy and entropy for b=0.4
la=['Calc., full data','Calc., reduced data','Calc., DFT only']
#linestyles=['solid','dotted',"dashed"]
fig = plt.figure(figsize=(2*5,4))
ax = fig.add_subplot(1, 2, 1)
for k in range(len(res_dir)):
    #ax.scatter(d[-2],ho[k][-2],label=la[k])
    ax.plot(d[-2],ho[k][-2],label=la[k],color=color_table[k])

ho2,ho3,ho4=np.loadtxt('ho_Calc_Rupp'),np.loadtxt('ho_Expt_Tagawa'),np.loadtxt('ho_Expt_Takacs')
ax.scatter(ho2[:,0],ho2[:,1],label='Calc., Bork',color=color_table[3])
ax.scatter(ho3[:,0],ho3[:,1],label='Expt., Tagawa',color=color_table[4])
ax.scatter(ho4[:,0],ho4[:,1],label='Expt., Takacs',color=color_table[5])
plt.xlabel("$\mathregular{δ}$",fontsize=15)
plt.ylabel("$\mathregular{Δh_O}$ (kJ/mol)",fontsize=15)
plt.xlim([0,0.2])

ax = fig.add_subplot(1, 2, 2)
for k in range(len(res_dir)):
    #ax.scatter(d[-2],so[k][-2],label=la[k])
    ax.plot(d[-2],so[k][-2],label=la[k],color=color_table[k])

so2,so3,so4=np.loadtxt('so_Calc_Rupp'),np.loadtxt('so_Expt_Tagawa'),np.loadtxt('so_Expt_Takacs')
ax.scatter(so2[:,0],so2[:,1],label='Calc., Bork',color=color_table[3])
ax.scatter(so3[:,0],so3[:,1],label='Expt., Tagawa',color=color_table[4])
ax.scatter(so4[:,0],so4[:,1],label='Expt., Takacs',color=color_table[5])
plt.xlabel("$\mathregular{δ}$",fontsize=15)
plt.ylabel("$\mathregular{Δs_O}$ (J/mol/K)",fontsize=15)
plt.xlim([0,0.2])

plt.legend(loc="upper right",bbox_to_anchor=(0.97, 0.95),frameon=False)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13, top=0.93, wspace=0.3, hspace=0.1)
#plt.show()
plt.savefig('hs.png',dpi=300, bbox_inches='tight')
