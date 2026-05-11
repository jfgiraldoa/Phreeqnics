"""
Script Name: Phreeqc + Fenics Reactive Transport. Built on top of Phreeqpy scripts
Author: Juan Giraldo
Email: juan.giraldo@csiro.au
Created: 2024-04-30
Version: Version 1.0
"""

def addInputParameters(params):
# ----------------------------------------------------------------------#
  #Method
# ----------------------------------------------------------------------#
  params["DG"]    = False
  params["ASFEM"] = False
# ----------------------------------------------------------------------#
  #Element space
# ----------------------------------------------------------------------#
  params["pdegree"]   = 1 #Polynomial Degree
  params["TestType"]  = "CG" #Test type function 
  params["TrialType"] = "CG" #Trial type function
  params["Ptrial"]    = params["pdegree"] # trial polynomial degree
  params["Ptest"]     = params["pdegree"]  # test polynomial degree
  params['Degree']    = 8  #Approximation polynomial
  params['boundconst'] = False
  params['boundarydomain'] = ''

 # ----------------------------------------------------------------------#
   #DG method
 # ----------------------------------------------------------------------# 
  params['epsilon'] = -1 # SIP: epsilon=1 or NIP: epsilon=-1 (Symmetric or Non Symmetric formulation)
  params['eta_adv'] = 1 # advection penalization (1 upwinding 0 CF)
  params['superpen'] = 0  #Super penalization parameter 
  params["SolverDG"] = 'DirectDG'  
  params["SolverASFEM"] = 'DirectCGDG'
  # ----------------------------------------------------------------------#
    #Refinement
  # ----------------------------------------------------------------------# 
  
  params["MAX_REF"] = 1;
  params['REF_TYPE'] = 1   
  params['tolref'] = 0.35   # refinement until an error iqual to 1+tolref times the cut off error. (tolref=0 default)
  params['REFINE_RATIO'] = 0.3   # Percentage (as fraction between 0 and 1) of elements to refine
  params['critref'] = 'mix'
  params['meshdomain'] = 'Unitdomain'

# ----------------------------------------------------------------------#
  #Directories
# ----------------------------------------------------------------------#    
  params['phreeqcpath']   = "/usr/local/lib/libiphreeqc.dylib" #Phreeqc dir
  params['DBpath']        = '../database/phreeqc.dat' #Phreeqc database
  #params['DBpath']        = '../database/PHREEQC_novel.dat' #Phreeqc database

  params['folderresults'] = '../results/'  
# ----------------------------------------------------------------------#
  #Miscelaneous
  params["show_plot_end"] = False # plot show at the end
  params["show_plot_timestep"] = False # Plot show every time step
  params["show_plot_iteration"] = False
  params["show_plot_snapshots"] = False
  params['save_pvd_iteration'] = False
  params['save_pvd_timestep_2d'] = False
  params['save_pvd_markers_2d'] = False
  
  params['Solution_modify'] = True
  params['accumulate_solution'] = False
  params['accumulate_solution_adapta'] = False
  params["reaction"] = True #True Reactive flow, false only diffusion-advection
  params['shows_plot'] = True #pop up plot while running
  params['plotboundaries'] = False

  params['ident'] = '0' 
  params['view_png'] = 0 
  params['view_pvd'] = 0 

  return params