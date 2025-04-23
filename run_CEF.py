from HydroGEN_CEF_general_current import *

#lattice='(A +2)1(Mn +3,Mn +4)1(O -2,Va 0)3'
#lattice='(La +3,Sr +2,Va 0)1(Mn +2,Mn +3,Mn +4,Va 0)1(O -2,Va 0)3'
#lattice='(La 0,Sr 0,Va 0)1(Mn 0,Va 0)1(O 0,Va 0)3'
#lattice='(La +3,Sr +3,Va +3)1(Mn +3,Va +3)1(O -2,Va -2)3'
#lattice='(Fe +3,Ni +3,Va +3)2(Ni +2,Fe +2,Va +2)1(O -2,Va -2)4'
#lattice='(Ba 0)4(Ce 0,Mn 0,Ba 0)1(Mn 0)3(O 0,Va 0)6(O 0,Va 0)6'
lattice='(La,Sr,Va)1(Mn,Va)1(O,Va)3'
#d_y_subs="{y[0][1]:symbols('x'),y[0][2]:1e-9,y[1][1]:1e-9,y[2][1]:1/3*symbols('d')}"
#d_y_subs="{y[0][1]:symbols('x'),y[0][2]:0,y[1][1]:0,y[2][1]:1/3*symbols('d'),log(y[0][2]):0,log(y[1][1]):0}" #lsm
#d_y_subs="{y[0][1]:0.5*y[1][1],y[0][2]:0,y[1][2]:0,y[2][1]:1/4*symbols('d'),log(y[0][2]):0,log(y[1][2]):0}" #nfo

# For fitting Gibbs energy without configurational entropy
generate_cef_module(lattice,G_neutral_values=[],T_terms=[1,T,T*log(T)],ir=(-1,-1),use_neutral=False)


