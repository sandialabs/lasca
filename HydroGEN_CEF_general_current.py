import re
import os
import numpy as np
import math
from shutil import copyfile,rmtree
import copy
from sympy import symbols, solve, simplify, preorder_traversal, Float, log, diff, apart
import itertools

"""
Motivation of changing from the previous version:
Printing sentences to correlate L and site fraction terms.
Allow separating A to G_DFT(0)+A' in the endmember Gibbs energy G(T)=A+dG(T).
"""

"""
TO-DOs:
Remove terms like R*T*log(4): solved
g_ratio needs to be rounded before judging if it is a float
dG_soln: when d is not used, it becomes zero
"""

"""
The sentences where user input may be needed are followed by comments containing the word "customizable":
lattice
G_neutral_values: if not specified, use zeros (recommended).
simp
T_terms: T terms used for expanding parameters, e.g., [1,T]; [1,T,T*log(T)]; [1,T,T*log(T),T**2]; ...
d_y_subs: site fraction variables substitution, e.g., {y[i][j]:symbols('d')} means that in the ith defect 
sublattice the jth species site fraction is substituted by symbols('d').
"""

# Constants
r = 8.617333262E-5; # Boltzman constant in eV/K.
# Symbols
R,T=symbols('R T')


def generate_cef_module(lattice,G_neutral_values=[],T_terms=[1,T,T*log(T)],ir=(-1,-1),d_y_subs="{}",use_neutral=True):

	"""
	Examples:
	#lattice='(Fe +2)1(Al +3,Fe +2,Fe +3)2(O -2,Va 0)4' # FAO.
	#lattice='(Ba +2,Sr +2)1(Fe +3,Fe +4)1(O -2,Va 0)3' # BSFO. 
	#lattice='(Sr +2,Ce +2)1(Mn +3,Mn +4)1(O -2,Va 0)3' # SCM.
	#lattice='(La +3,Sr +2)1(Mn +3,Mn +4)1(O -2,Va 0)3' # LSM.
	#lattice='(La,Sr,Va)1(Mn,Va)1(O,Va)3' # LSM without specifying charges. 
	#d_y_subs="{y[0][1]:symbols('d'),y[0][2]:symbols('x')}" # FAO. 
	#d_y_subs="{y[0][-1]:1.0-symbols('x'),y[1][-1]:1.0-2.0*symbols('d')}" # BSFO. 
	"""

	# Function of parsing the lattice string to the list of species, charges and stoichiometries. 
	def parse_lattice(lattice):
		c=re.split('[()]',lattice.strip('\n'))
		for j in range(c.count('')):
			c.remove('')
		for j in range(len(c)//2):
			c[2*j]=c[2*j].split(',')
			c[2*j+1]=float(c[2*j+1])
			for k in range(len(c[2*j])):
				c[2*j][k]=c[2*j][k].split(' ')
				if len(c[2*j][k])==1:
					c[2*j][k].append(0)
				c[2*j][k][1]=float(c[2*j][k][1])
		return c

	c=parse_lattice(lattice) # the list of species, charges and stoichiometries of the lattice string.
	s=[c[2*j+1] for j in range(len(c)//2)] # stoichiometries
	print(str(s))
	q=[[k[1] for k in c[2*j]] for j in range(len(c)//2)] # charges

	y0=[[0 for k in range(1,len(c[2*j]))] for j in range(len(c)//2) if len(c[2*j])>1] # initialize the site fraction list (for the site with #species>1); the site fraction of the first species is excluded due to dependency.
	defect_sites=[] # sites that have #species>1
	perfect_sites=[] # sites that have #species=1
	for j in range(len(c)//2):
		if len(c[2*j])>1:
			defect_sites.append(j)
		else:
			perfect_sites.append(j)

	s_defect=[s[i] for i in range(len(s)) if i in defect_sites]

	"""
	Function of generating all the possible 0-1 endmembers recursively. 
	n: the first n sites in a lattice string to be occupied.
	y0: the site fraction list 
	"""
	def rend(n,y0):
		yy=[]
		if n==1:
			yy.append([[0 for i in y0[0]]])
			for i in range(len(y0[0])):
				ee=[0 for j in y0[0]]
				ee[i]=1
				yy.append([ee])
		else:
			s=rend(n-1,y0)
			e=[]
			e.append([0 for i in y0[n-1]])
			for i in range(len(y0[n-1])):
				ee=[0 for j in y0[n-1]]
				ee[i]=1
				e.append(ee)
			for i in range(len(e)):
				for j in range(len(s)):
					ss=copy.deepcopy(s[j])
					ss.append(e[i])
					yy.append(ss)
		return yy

	# List of the site fraction list of all the neutral endmembers.
	ys=[]
	for i in range(len(y0)):
		y2=[y0[j] for j in range(len(y0)) if j!=i]
		yy=rend(len(y2),y2)
		# construct A matrix
		si=s[defect_sites[i]]
		qi=np.array(q[defect_sites[i]])
		A=np.vstack((si*(qi-qi[0])[1:],np.identity(len(qi)-1),np.ones(len(qi)-1)))
		# construct b vector
		b=np.zeros(len(qi)+1)
		b[-1]=1
		sp=[s[j] for j in range(len(s)) if j in perfect_sites]
		qp=[q[j][0] for j in range(len(q)) if j in perfect_sites]
		sp=np.array(sp);qp=np.array(qp)
		sd_others=[s[j] for j in range(len(s)) if j in defect_sites and j!=defect_sites[i]]
		qd_others=[q[j] for j in range(len(q)) if j in defect_sites and j!=defect_sites[i]]
		iequations=itertools.combinations([j for j in range(1,len(qi)+1)], len(qi)-2) # select square matrix from A 
		for j in range(len(yy)):
			b[0]=-np.dot(sp,qp)-si*qi[0]
			for k in range(len(yy[j])):
				bb=-sd_others[k]*(np.dot(np.array(yy[j][k]),np.array(qd_others[k][1:]))+(1-sum(yy[j][k]))*qd_others[k][0])
				b[0]=b[0]+bb
			for k in itertools.combinations([m for m in range(1,len(qi)+1)], len(qi)-2):
				kk=[0]+list(k)
				A_select=A[kk,:]
				b_select=np.array([b[m] for m in range(len(b)) if m in kk])
				yi=''
				try:
					yi = list(np.linalg.solve(A_select, b_select))
				except:
					print('No "exact" solution for this combination of conditions!')
				if yi!='':
					yyj=copy.deepcopy(yy[j])
					yyj.insert(i, yi)
					ys.append(yyj)


	# Function of converting the structured site fraction list to an unstructrued list.
	def flatten(l):
		return [item for sublist in l for item in sublist]

	ys=[i for i in ys if min(flatten(i))>=0 and max(flatten(i))<=1] # exclude unphysical solutions of neutral endmembers.

	"""
	Function of measuring the distance between two endmembers.
	y1,y2: site fraction lists of the endmembers.
	s: stoichiometries.
	norm: order of the norm used in defining the distance.
	"""
	def distance_y(y1,y2,s,norm=2):
		y1=[s[i]*np.array(y1[i]) for i in range(len(y1))]
		y2=[s[i]*np.array(y2[i]) for i in range(len(y2))]
		y1=np.array(flatten(y1))
		y2=np.array(flatten(y2))
		dy12=y1-y2
		d=sum(abs(dy12)**norm)**(1./norm)
		return d

	"""
	Remove duplicated neutral endmembers.
	"""
	ys_duplicated=[]
	for i in range(len(ys)):
		for j in range(i+1,len(ys)):
			if distance_y(ys[i],ys[j],s_defect)<1e-23:
				ys_duplicated.append(j)

	ys=[ys[i] for i in range(len(ys)) if not i in ys_duplicated]

	# remove "-" before 0 
	for i in range(len(ys)):
		for j in range(len(ys[i])):
			for k in range(len(ys[i][j])):
				ys[i][j][k]=abs(ys[i][j][k])

	# ys with the site fraction of the main species in each sublattice
	ys_full=[[[1.0-sum(i)]+i for i in j] for j in ys]

	"""
	Function of mapping the site fraction list to chemical formula.
	c: the list of species, charges and stoichiometries of the lattice string.
	y: the site fraction list
	original: if True, the sites with the same species are not merged.
	"""
	def y2f(c,y,original=False):	
		comp={}
		ele=[]
		for i in range(len(c)//2):
			if len(c[2*i])==1:
				site_fractions=[1]
			else:
				site_fractions=[1-sum(y[defect_sites.index(i)])]+y[defect_sites.index(i)]
			for j in range(len(c[2*i])):
				if not c[2*i][j][0] in comp.keys():
					comp[c[2*i][j][0]]=site_fractions[j]*c[2*i+1]
					ele.append(c[2*i][j][0])
				else:
					comp[c[2*i][j][0]]+=site_fractions[j]*c[2*i+1]
		comp.pop('Va', None)
		ele.remove('Va')
		f=''
		for i in range(len(ele)):
			if comp[ele[i]]>0:
				if comp[ele[i]]-1==0:
					f=f+ele[i]
				else:
					f=f+ele[i]+('%f' % round(comp[ele[i]],3)).rstrip('0').rstrip('.')
		if original:
			f=''
			for i in range(len(c)//2):
				if len(c[2*i])==1:
					site_fractions=[1]
				else:
					site_fractions=[1-sum(y[defect_sites.index(i)])]+y[defect_sites.index(i)]
				f=f+'('
				for j in range(len(c[2*i])):
					if site_fractions[j]>0:
						if site_fractions[j]-1==0:
							f=f+c[2*i][j][0]+('%f' % c[2*i][j][1]).rstrip('0').rstrip('.')
						else:
							f=f+c[2*i][j][0]+('%f' % c[2*i][j][1]).rstrip('0').rstrip('.')+'_'+('%f' % round(site_fractions[j],3)).rstrip('0').rstrip('.')
				f=f+')'					
		f=re.sub(r"([a-zA-Z])0", r"\1",f)
		return f

	"""
	The formula list of neutral endmembers.
	"""
	neutral_endmembers=[]
	for i in range(len(ys)):
		neutral_endmembers.append(y2f(c,ys[i],original=True))

	print('All the neutral endmembers:')
	for i in neutral_endmembers:
		print(i)

	"""
	The symbol list of Gibbs energies of neutral endmembers.
	"""
	G_neutral=[]
	for i in range(len(neutral_endmembers)):
		G_neutral.append(symbols('G_'+neutral_endmembers[i]))

	# These are 0K values, but obviously they should and can be extended to finite-T.
	if len(G_neutral_values)==0:
		G_neutral_values=np.zeros(len(G_neutral)) # It is recommended to use the default zeros for G_neutral_values.

	# Introduce symbols for site fractions.
	y_raw=[[symbols('y'+str(j)+'_('+re.sub(r"([a-zA-Z])0", r"\1", c[2*j][k][0]+('%f' % c[2*j][k][1]).rstrip('0').rstrip('.'))+')') for k in range(len(c[2*j]))] for j in range(len(c)//2) if len(c[2*j])>1] # no constraint applied
	y=[[symbols('y'+str(j)+'_('+re.sub(r"([a-zA-Z])0", r"\1", c[2*j][k][0]+('%f' % c[2*j][k][1]).rstrip('0').rstrip('.'))+')') for k in range(1,len(c[2*j]))] for j in range(len(c)//2) if len(c[2*j])>1]
	# Independent y
	y_idp=flatten(y)
	for i in range(len(y)):
		y_main=1.0-sum(y[i])
		y[i]=[y_main]+y[i]

	"""
	Gibbs energy of each endmember.
	"""
	y_all=rend(len(y0),y0)
	# y_all with the site fraction of the main species in each sublattice
	y_all_full=[[[1.0-sum(i)]+i for i in j] for j in y_all]
	G_all=[symbols('G_'+y2f(c,i,original=True)) for i in y_all]
	if use_neutral:
		print('Expressing Gibbs energies of all the endmembers using those of the neutral ones:')
		for i in range(len(y_all)):
			dist_y=[distance_y(y_all[i],ys[j],s,norm=1) for j in range(len(ys))]
			iys_min=[j for j in range(len(ys)) if dist_y[j]==min(dist_y)][0]
			diff_y=[[y_all_full[i][j][k]-ys_full[iys_min][j][k] for k in range(len(y_all_full[i][j]))] for j in range(len(y_all_full[i]))]
			G_all[i]=G_neutral[iys_min]
			entropy_correction=0.0
			for j in range(len(ys_full[iys_min])):
				for k in range(len(ys_full[iys_min][j])):
					if (ys_full[iys_min][j][k]>0 and ys_full[iys_min][j][k]<1):
						entropy_correction=entropy_correction+s[j]*symbols('R')*symbols('T')*ys_full[iys_min][j][k]*np.log(ys_full[iys_min][j][k])
			G_all[i]=G_all[i]-entropy_correction
			for j in range(len(diff_y)):
				for k in range(len(diff_y[j])):
					G_all[i]=G_all[i]+diff_y[j][k]*s[j]*symbols('G_'+c[2*defect_sites[j]][k][0])
			print('G_'+y2f(c,y_all[i],original=True)+' = '+str(G_all[i]))

	"""
	The endmember part of Gibbs energy of the system.
	"""
	G_end=0.0
	for i in range(len(y_all_full)):
		coeff=1.0
		for j in range(len(y_all_full[i])): 
			coeff=coeff*np.dot(np.array(y_all_full[i][j]),np.array(y[j]))
		G_end=G_end+coeff*G_all[i]

	"""
	Entropy
	"""
	S=0.0
	for i in range(len(y)):
		for j in range(len(y[i])):
			S=S-s[i]*symbols('R')*y[i][j]*log(y[i][j])

	# Excess Gibbs energy
	G_ex=0.0
	L_terms=[]
	iL=1
	L_names=[]
	for i in range(len(y)):
		y2=[y0[j] for j in range(len(y0)) if j!=i]
		yy=rend(len(y2),y2);print(yy)
		for m in range(len(yy)):
			yy[m]=[[1-sum(mm)]+mm for mm in yy[m]]
		y_no_mixing=[y[j] for j in range(len(y)) if j!=i]
		y_no_mixing_raw=[y_raw[j] for j in range(len(y)) if j!=i]
		for j in range(len(y[i])):
			for k in range(j+1,len(y[i])):
				for m in range(len(yy)):				
					product_y=1
					no_mixing_species=[]
					for mm in range(len(yy[m])):
						product_y=product_y*np.dot(np.array(yy[m][mm]),np.array(y_no_mixing[mm]))
						yd=np.dot(np.array(yy[m][mm]),np.array(y_no_mixing_raw[mm]))
						no_mixing_species.append(str(yd).split('_')[-1].strip('(').strip(')'))
					G_ex=G_ex+product_y*y[i][j]*y[i][k]*(symbols('L'+str(iL))+(y[i][j]-y[i][k])*symbols('L'+str(iL+1)))
					L_terms.append({'original':product_y*y[i][j]*y[i][k]})
					L_terms.append({'original':product_y*y[i][j]*y[i][k]*(y[i][j]-y[i][k])})
					iL=iL+2
					#no_mixing_species=[ for m in y_no_mixing_raw[m]]
					no_mixing_species.insert(i, str(y_raw[i][j]).split('_')[-1].strip('(').strip(')')+','+str(y_raw[i][k]).split('_')[-1].strip('(').strip(')'))
					L_names.append('L_{'+':'.join(no_mixing_species)+';0}')
					L_names.append('L_{'+':'.join(no_mixing_species)+';1}')

	#print(L_names)
	# Customizable function for simplifying an expression.
	def simp(G,expand=True): 
		# use charge neutrality to remove one assigned site fraction.
		total_charges=0.0
		y_with_perfect=[[1] for i in s]
		for i in range(len(defect_sites)):
			y_with_perfect[defect_sites[i]]=y[i]
		for i in range(len(y_with_perfect)):
			total_charges=total_charges+np.dot(np.array(y_with_perfect[i]),np.array(q[i]))*s[i]
		# assign the removed site fraction, (lattice, species)
		i_removed_by_neutrality=ir # In terms of all the defect sites, (-1,-1) means the last site fraction in the last defect site. Customizable
		yrbn=solve(total_charges,y[i_removed_by_neutrality[0]][i_removed_by_neutrality[1]])
		if len(yrbn)>0:
			y_removed_by_neutrality=yrbn[0]
			G=G.subs(y[i_removed_by_neutrality[0]][i_removed_by_neutrality[1]], y_removed_by_neutrality)
			yidp=[i for i in y_idp if not i!=y[i_removed_by_neutrality[0]][i_removed_by_neutrality[1]]]
		yidp=y_idp
		# customize variables of site fraction
		dict_y_subs=eval(d_y_subs)
		G_new=G.subs(dict_y_subs)
		print('Customize variables of site fraction:')
		for i in dict_y_subs.keys():
			print(str(i)+' = '+str(dict_y_subs[i]))
		G_new=G_new.subs(symbols('G_Va'),0)
		if expand:
			G_new=simplify(G_new.expand())
		# rounding
		ndigit=15 # Float accuracy. Customizable
		G_rounded=G_new
		for a in preorder_traversal(G_new):
			if isinstance(a, Float):
				G_rounded = G_rounded.subs(a, round(a, ndigit))
		return G_rounded, dict_y_subs, yidp

	"""
	Remove redundant L
	"""
	G_ex=simp(G_ex,expand=False)[0] # Not expand the excess part.
	g=[]
	for i in range(1,iL):
		dict_L_subs={}
		for j in range(1,iL):
			dict_L_subs[symbols('L'+str(j))]=0
		dict_L_subs[symbols('L'+str(i))]=1	
		g.append(G_ex.subs(dict_L_subs))
		L_terms[i-1]['simplified']=g[-1]

	dict_L_remove={}
	for i in range(len(g)):
		for j in range(i+1,len(g)):
			g_ratio=g[j]/g[i]
			if isinstance(g_ratio, Float):
				dict_L_remove[symbols('L'+str(j+1))]=0

	G_ex=G_ex.subs(dict_L_remove) # Throw the redundant L-terms directly. 

	L_kept=[symbols('L'+str(i)) for i in range(1,iL) if not symbols('L'+str(i)) in dict_L_remove.keys()]  
	dict_L_renumber={} 
	for i in range(len(L_kept)):
		dict_L_renumber[L_kept[i]]=symbols('L'+str(i+1))

	G_ex=G_ex.subs(dict_L_renumber) # Renumber the remained L parameters.

	dict_L_names={}
	for i in range(1,iL):
		if symbols('L'+str(i)) in L_kept:
			print('L'+str(i)+' -> '+str(dict_L_renumber[symbols('L'+str(i))])+'	'+str(L_terms[i-1]['original'])+'	'+str(L_terms[i-1]['simplified']))
			dict_L_names[str(dict_L_renumber[symbols('L'+str(i))])]=L_names[i-1]
			print(dict_L_names)
		else:
			print('L'+str(i)+' -> '+'Redundant'+'	'+str(L_terms[i-1]['original'])+'	'+str(L_terms[i-1]['simplified']))

	# Complete Solution Model 
	G_sol=G_end+G_ex-symbols('T')*S;G_sol_unexpanded=G_sol
	G_sol,dict_y_subs,y_idp=simp(G_sol) # simplifying the total Gibbs energy

	# Make expansion of G -> G(T) 1 = B*T 2 = B*T + C*T*ln(T) 3 = B*T + C*T*ln(T) + T^2/2
	# T_terms=[1,T,T*log(T)] # Customizable: [1,T]; [1,T,T*log(T)]; [1,T,T*log(T),T**2]; ...
	if use_neutral:
		aG_end=[[symbols('a_'+str(i)+'_'+str(j)) for j in range(len(T_terms))] for i in range(len(G_neutral))] # T coefficients of the endmember part of Gibbs energy.
		a_names=[['a_{'+str(G_neutral[i]).strip('G_').strip('(').strip(')').replace(')(',':')+';'+str(j)+'}' for j in range(len(T_terms))] for i in range(len(G_neutral))]
	else:
		aG_end=[[symbols('a_'+str(i)+'_'+str(j)) for j in range(len(T_terms))] for i in range(len(G_all))]
		a_names=[['a_{'+str(G_all[i]).strip('G_').strip('(').strip(')').replace(')(',':')+';'+str(j)+'}' for j in range(len(T_terms))] for i in range(len(G_all))]
	print(a_names)
	dict_a_names=dict(zip([str(i) for sub in aG_end for i in sub], [i for sub in a_names for i in sub]))
	"""
	Substitute the endmember Gibbs energy by T series with coefficients.
	"""
	dict_end_subs={}
	if use_neutral:
		for i in range(len(G_neutral)):
			dict_end_subs[G_neutral[i]]=np.dot(np.array(aG_end[i]),np.array(T_terms))+G_neutral_values[i]
	else:
		for i in range(len(G_all)):
			dict_end_subs[G_all[i]]=np.dot(np.array(aG_end[i]),np.array(T_terms))	

	# Full finite-T model
	G_soln=G_sol.subs(dict_end_subs).subs(R,r)
	G_soln=simplify(G_soln.expand())

	# 0K model
	G_0K=G_soln.subs(log(symbols('T')),0).subs(T,0)
	G_0K=simplify(G_0K.expand())

	# Endmembers only model
	G_end_only=G_end-T*S
	G_end_only=simp(G_end_only)[0]
	G_end_only=G_end_only.subs(dict_end_subs).subs(R,r)
	G_end_only=simplify(G_end_only.expand())

	G_end_only_0K=G_end_only.subs(log(symbols('T')),0).subs(T,0)
	G_end_only_0K=simplify(G_end_only_0K.expand())

	# Entropy
	S=S.subs(dict_end_subs).subs(R,r)
	S=simplify(S.expand())

	# Gibbs energy without configurational entropy
	G_nonconfig=G_soln+T*S
	G_nonconfig=simplify(G_nonconfig.expand())

	"""
	Make a file to save the results.
	"""
	short_name=''.join([c[2*j][0][0] for j in range(len(c)//2)]) # informal
	with open('CEF_'+short_name,'w') as f:
		f.write('Sublattice model:'+'\n')
		f.write(lattice+'\n')
		f.write('\n')
		f.write('All the neutral endmembers:'+'\n')
		for i in neutral_endmembers:
			f.write(i+'\n')
		f.write('\n')
		f.write('Expressing Gibbs energies of all the endmembers using those of the neutral ones:'+'\n')
		for i in range(len(y_all)):
			f.write('G_'+y2f(c,y_all[i],original=True)+' = '+str(G_all[i])+'\n')
		f.write('\n')
		f.write('G_end:'+'\n')
		f.write(str(G_end)+'\n')
		f.write('\n')
		f.write('S:'+'\n')
		f.write(str(S)+'\n')
		f.write('\n')
		f.write('G_ex:'+'\n')
		f.write(str(G_ex)+'\n')
		f.write('\n')
		f.write('Customize variables of site fraction:'+'\n')
		for i in dict_y_subs.keys():
			f.write(str(i)+' = '+str(dict_y_subs[i])+'\n')
		f.write('\n')
		f.write('All the excess parameters with redundancy removed:'+'\n')
		for i in range(1,iL):
			if symbols('L'+str(i)) in L_kept:
				f.write('L'+str(i)+' -> '+str(dict_L_renumber[symbols('L'+str(i))])+'	'+str(L_terms[i-1]['original'])+'	'+str(L_terms[i-1]['simplified'])+'\n')
			else:
				f.write('L'+str(i)+' -> '+'Redundant'+'	'+str(L_terms[i-1]['original'])+'	'+str(L_terms[i-1]['simplified'])+'\n')
		f.write('\n')
		f.write('Total Gibbs energy without expansion:'+'\n')
		#f.write(str(G_sol)+'\n')
		f.write(str(G_sol_unexpanded)+'\n')
		f.write('\n')
		f.write('T terms for expansion:'+'\n')
		f.write(str(T_terms)+'\n')
		f.write('\n')
		f.write('Substitute the endmember Gibbs energy by T series with coefficients:'+'\n')
		for i in dict_end_subs.keys():
			f.write(str(i)+' = '+str(dict_end_subs[i])+'\n')
		f.write('\n')
		f.write('Total Gibbs energy with T expansion:'+'\n')
		f.write(str(G_soln)+'\n')
		f.write('\n')
		f.write('Total Gibbs energy with T expansion at 0 K:'+'\n')
		f.write(str(G_0K)+'\n')
		f.write('\n')
		f.write('Endmember-only Gibbs energy with T expansion:'+'\n')
		f.write(str(G_end_only)+'\n')
		f.write('\n')
		f.write('Endmember-only Gibbs energy with T expansion at 0 K:'+'\n')
		f.write(str(G_end_only_0K)+'\n')
		f.write('\n')
		f.write('Nonconfigurational-Gibbs energy with T expansion:'+'\n')
		f.write(str(G_nonconfig)+'\n')
		f.write('\n')
		f.write('Independent site fractions:'+'\n')
		f.write(str(y_idp)+'\n')
		f.write('\n')
		f.write('Endmember parameters:'+'\n')
		f.write(str(flatten(aG_end))+'\n')
		f.write('\n')
		f.write('Independent excess parameters:'+'\n')
		f.write(str(list(dict_L_renumber.values()))+'\n')
		f.write('\n')
		f.write('a_names:'+'\n')
		f.write(str(dict_L_names)+'\n')
		f.write('\n')
		f.write('L_names:'+'\n')
		f.write(str(dict_a_names)+'\n')