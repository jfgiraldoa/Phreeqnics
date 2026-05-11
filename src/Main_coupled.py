"""
Script Name: Phreeqc + Fenics Reactive Transport 
Author: Juan Giraldo
Email: juan.giraldo@csiro.au
Created: 2024-04-30
Version: Version 1.0
"""
from Coupled_model_dar  import CoupledModelDAR
from Phreeqc_reaction import ScriptGenerator

from Postprocess import Postprocess

def main(problem,p,dim,processes):
    p['dim'] = dim
    nshifts=problem.Temporal(p)['time_steps']; 
    generator = ScriptGenerator(problem,p)
    initial_conditions = generator.generate_script()
    model = CoupledModelDAR(nshifts, initial_conditions,processes,problem,p)
    results_dar,results_prop,accum,accumadapta,dofstotal = model.run_coupled() 
    postprocess = Postprocess(p,problem)         
    if p['dim']==1: 
        sol = model.outflow1D(results_dar,nshifts)
        if p["show_plot_end"]: postprocess.plot1D(sol,nshifts)
    elif p['dim']==2: 
        sol = postprocess.outflow2D(results_dar,nshifts)
        if p["show_plot_end"]: postprocess.plot2D(results_dar,nshifts)
    output = []        
    if problem.ExactSoltution():  
            output = postprocess.L2normcompute(results_dar)
    else:   
            output = sol     
    if p["show_plot_timestep"]:        
        print('Statistics')
        print('==========')
        print('number of nodes:    ', p['mesh'].num_vertices())
        print('number of shifts:   ', nshifts)
        print('number of processes:', processes)
        print()       
    return output,results_prop,accum,accumadapta,dofstotal
