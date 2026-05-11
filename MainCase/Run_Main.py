"""
Script Name: Phreeqc + Fenics Reactive Transport V1 
Author: Juan Giraldo
Email: jfgiraldoa@gmail.com
"""
import sys
import os

sys.path.append(os.path.abspath('../src'))

from AddInputparameters import addInputParameters
from Main_coupled import main
from Postprocess import Postprocess
from Exchange_outputs import save_exchange_outputs_h5, save_exchange_outputs_h5_adaptive
from Cases import Test_Cases


def configure_problem():
    p = addInputParameters({})
    p["case"] = "Engeesgard" #"Pe100", "Pe1000", "Pe10000", "Engeesgard"
    p["show_plot_end"] = True 
    if p["case"] in ["Pe100", "Pe1000", "Pe10000"]: 
        p['accumulate_solution_adapta'] = True

    p['dim'] = 1
    if p["case"] == 'Case_2D':
        p['dim'] = 2
        p['save_pvd_timestep_2d'] = True
        p['save_pvd_markers_2d'] = True
        p['boundarydomain'] = 'CalciteDolomite2D-adaptive'
    p['ASFEM'] = False if p["case"] == 'Engeesgard' else True   

    problem = Test_Cases(p)

    p["MAX_REF"] = problem.max_ref
    p["Nx_save"] = problem.nx
    p['snap_time_list'] = problem.snaplist
    p['h_transport_min_factor'] = problem.h_transport_min_factor
    p['chem_refine_tol'] = problem.chem_refine_tol
    p['ASFEM'] = False if problem.case == 'Engeesgard' else True   

    return p, problem

def main_runner():
    p, problem = configure_problem()

    print(
        f"Péclet = {problem.Pe:.3g}, "
        f"Artificial diffusion (BE) = {problem.D_BE:.3g}, "
        f"Péclet_eff = {problem.Pe_eff:.3g}, "
        f"Courant = {problem.Cou:.3g}"
    )

    data, results_prop, accum, accumadapta, dofstotal = main(problem, p, dim=p['dim'], processes=1)

    problem.save_outputs(p, data, accum, accumadapta, dofstotal)
    #print('DOFs totales:', dofstotal)
    print('Finish')

if __name__ == '__main__':
    main_runner()
