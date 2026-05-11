"""
Script Name: Phreeqc + Fenics Reactive Transport 
Author: Juan Giraldo
Email: jfgiraldo@gmail.com
Created: 2024-04-30
Version: Version 1.0
Dependencies: Fenics version 2019 + phreeqcpy
"""

import timeit
import matplotlib.pyplot as plt
import numpy as np
import fenics
import plotly.graph_objects as go
from dolfin import *
import os
import itertools
import plotly.offline as pyo
from matplotlib.cm import get_cmap
import math
from dolfin import plot as dfplot
import h5py
from matplotlib.ticker import ScalarFormatter  
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class Postprocess(object):    
    def __init__(self, params,problem):
      self.p = params   
      self.problem = problem
      self.diff = str(next(iter(self.problem.diffusionCoeff().values())).values()[0])
         
    def postprocessplots(self,sol,plotval,i,pvd_files): 
        self.plot_outflow(plotval, i+1)
        if self.p['dim']==2: 
            if self.p['view_pvd']:
             self.save_to_paraview(sol,plotval, i+1, pvd_files)
            if self.p['view_png']:
             self.plot2D(sol,i+1)
 
    def measure_time(self,func, *args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        return result, timeit.default_timer() - start
    
    def plotspecies(self,outflow, specie_names):
        colors = self.problem.Colordic()
        args = []
        num_data_points = len(next(iter(outflow.values())))
        x_values = np.linspace(0, self.p['time_steps'], num_data_points)
        for name in specie_names:
            args.extend([x_values, outflow[name], colors[name]])
        plt.plot(*args)
        plt.legend(specie_names, loc=(0.8, 0.5))
        plt.ylabel('MILLIMOLES PER KILOGRAM WATER')
        plt.xlabel('time-steps')
        plt.savefig("ex11.png")
        if self.p['shows_plot']: plt.show()
        
    def L2normcompute(self,results_dar):    
        Exa = self.problem.ExactSoltution()
        namesplot, keys = self.problem.Components_plot()
        errors = {ion: [] for ion in keys}
        components = results_dar.split() 
        for i, key in enumerate(keys):  
           error = errornorm(Exa[key], components[i], 'L2')
           errors[key].append(error)
        return errors   
    
    def initialize_pvd_files(self):
        namesplot, keys = self.problem.Components_plot()
        class_name = self.problem.__class__.__name__
        diff = self.diff
        folder_path = os.path.join(self.p['folderresults']+class_name+'/', f'Diff{diff}')
        properties_plot = [key for key, value in self.problem.properties_key().items() if value]

        pvd_files = {}
        for key in keys+properties_plot:
            component_folder = os.path.join(folder_path, key)
            os.makedirs(component_folder, exist_ok=True)
            pvd_path = os.path.join(component_folder, f'{key}.pvd')
            pvd_files[key] = File(pvd_path)
        return pvd_files
    
    def save_to_paraview(self, un, params, time_step,pvd_files):
        components = un.split()
        keys = self.problem.Components_plot()[1]
        properties_plot = [key for key, value in self.problem.properties_key().items() if value]

        for i, key in enumerate(keys):
            components[i].rename(key,key)
            pvd_file = pvd_files[key]
            pvd_file << (components[i], float(time_step))
        
        if params != []:
            for key in properties_plot:
                params[key].rename(key,key)
                pvd_file = pvd_files[key]
                pvd_file << (params[key], float(time_step))    
                        
    def outflow2D(self,un, time_step):
        namesplot, keys = self.problem.Components_plot()
        components = un.split() 
        components_dict = dict(zip(keys, components))         
        return components_dict
    
    def plot2D(self,un, time_step):  
        pada = 0.04
        namesplot, keys = self.problem.Components_plot()
        components = un.split() 
        num_components = len(components) 
        cols = 4
        rows = (num_components + cols - 1) // cols 
        fig = plt.figure(figsize=(15, 5 * rows)) 
        color_scales = self.problem.color_scale()
        for i, key in enumerate(keys):  # Loop through keys instead of components directly
            ax = plt.subplot(rows, cols, i + 1)
            if color_scales['color-mod']:
                vmin, vmax = color_scales.get(key, color_scales['default'])
                norm = Normalize(vmin=vmin, vmax=vmax)
                mappable = ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array([])
                pl = plot(components[i], cmap='jet', vmin=vmin, vmax=vmax)  # Apply fixed vmin and vmax
                cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=pada, format=ScalarFormatter(useMathText=True))
            else: 
                pl = plot(components[i], cmap='jet')
                cbar = plt.colorbar(pl,ax=ax, fraction=0.046, pad=pada)
            plt.title(f'{key} ts: {time_step}')
 
        diff = self.diff

        class_name = self.problem.__class__.__name__

        folder_path = os.path.join(self.p['folderresults']+class_name+'/', f'Diff{diff}'+'/Concentretation_png')
        os.makedirs(folder_path, exist_ok=True)
        plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)  #
    
        file_name = f'solution2D-ts{time_step}.pdf'
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path, format='pdf', dpi=300)    
        if self.p['shows_plot']: plt.show()

        return components

    def plot2Dspecies(self, un, time_step):  
        pada = 0.04
        namesplot, keys = self.problem.Components_plot()
        components = un.split() 
        num_components = len(components)
        
        cols = 4  
        rows = (num_components + cols - 1) // cols 
        
        fig = plt.figure(figsize=(15, 5 * rows))
        color_scales = self.problem.color_scale()
    
        for i, key in enumerate(keys):
            if i >= num_components:
                break  
    
            ax = plt.subplot(rows, cols, i + 1)
            
            pl = plot(components[i], cmap='jet')
            cbar = plt.colorbar(pl, ax=ax, fraction=0.046, pad=pada)
    
            plt.title(f'{key} ts: {time_step}')
     
        diff = self.diff
        class_name = self.problem.__class__.__name__
    
        folder_path = os.path.join(self.p['folderresults'] + class_name, f'Diff{diff}'+'/', 'Concentration_png')
        os.makedirs(folder_path, exist_ok=True)
        
        plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
        
        file_name = f'solution2D-ts{time_step}.pdf'
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path, format='pdf', dpi=300)
        
        if self.p.get('shows_plot', False): 
            plt.show()
    
        return components

    def plot_outflow(self, plotval, time_step):
        class_name = self.problem.__class__.__name__
        folder_path = os.path.join(self.p['folderresults'], class_name)
        os.makedirs(folder_path, exist_ok=True)
        if plotval: 
            if self.p['dim'] == 2:
                num_plots = len(plotval)
                cols = 3  
                rows = (num_plots + cols - 1) // cols  
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        
                for i, (key, data) in enumerate(plotval.items()):
                    plt.subplot(rows, cols, i + 1)
                    if hasattr(data, 'ufl_domain'):
                        c = plot(data) 
                        plt.colorbar(c)
                        plt.title(f'{key.capitalize()} at Time Step {time_step}')
                    else:
                        print(f"Cannot plot {key}. Object lacks 'ufl_domain'. Type of object: {type(data)}")
                        plt.axis('off') 
        
                plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
                plt.subplots_adjust(hspace=0.5)
        
            elif self.p['dim'] == 1:
                num_plots = len(plotval)
                fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots)) 
        
                if num_plots == 1:
                    axes = [axes]  
        
                for ax, (key, values) in zip(axes, plotval.items()):
                    
                    x_values = np.linspace(self.p['Linfx'], self.p['Lsupx'], num=len(values))
                    ax.plot(x_values, values, label=key)
                    ax.set_xlabel('Distance (m)')
                    ax.set_ylabel(key, color='b')
                    ax.tick_params(axis='y', colors='b')
                    ax.legend()
                    ax.set_title(f'{key.capitalize()} at Time Step {time_step}')
        
                plt.tight_layout()  
            
            diff = self.diff

            class_name = self.problem.__class__.__name__
    
            folder_path = os.path.join(self.p['folderresults']+class_name+'/', f'Diff{diff}'+'/Properties_png')
            os.makedirs(folder_path, exist_ok=True)
            plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)  #
      
            file_name = f'Properties-ts{time_step}.pdf'
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path, format='pdf', dpi=300)    
        if self.p['shows_plot']: plt.show()
   

    from matplotlib.ticker import ScalarFormatter

    def _nice_sci(ax):
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)     
        fmt.set_useOffset(False)     
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
       
    def plot_fields(self, time_step, p, v):
        import os
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        from dolfin import plot, project, FunctionSpace
    
        def _nice_sci(ax):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_useOffset(False)         
            ax.yaxis.set_major_formatter(fmt)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        title_p = "Pressure Head"
        title_v = "Velocity Field"
    
        plt.sca(axes[0])
        p_plot = plot(p, title=f"{title_p} - ts: {time_step} (Pa)")
        _nice_sci(axes[0])
        if self.p["dim"] == 2:
            plt.colorbar(p_plot)
    
        plt.sca(axes[1])
        if self.p["dim"] == 1:
            v_x = project(v[0], FunctionSpace(v.function_space().mesh(), "P", 1))
            v_plot = plot(v_x, title=f"{title_v} X-Component - ts: {time_step} (m/s)")
        else:
            v_plot = plot(v, title=f"{title_v} - ts: {time_step} (m/s)")
        _nice_sci(axes[1])
        if self.p["dim"] == 2:
            plt.colorbar(v_plot)
    
        diff = self.diff
        class_name = self.problem.__class__.__name__
        folder_path = os.path.join(
            self.p["folderresults"] + class_name + "/", f"Diff{diff}", "Fields-vel-press"
        )
        os.makedirs(folder_path, exist_ok=True)
    
        plt.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    
        file_name = f"Field-vel-press-ts{time_step}.pdf"
        plt.savefig(os.path.join(folder_path, file_name), format="pdf", dpi=300)
    
        if self.p["shows_plot"]:
            plt.show()
        plt.close(fig)  
           
        def plot_boundaries(self,mesh, boundaries, marker_to_name):
               plt.figure()
               plot(mesh, alpha=0.3, linewidth=0.5)  
               
               color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
               handles = {}
               
               for facet in facets(mesh):
                   marker = boundaries[facet]
                   if marker > 0:
                       x = [v.point().x() for v in vertices(facet)]
                       y = [v.point().y() for v in vertices(facet)]
                       color = color_list[(marker-1) % len(color_list)] 
                       if marker not in handles:
                           line, = plt.plot(x, y, color=color, linewidth=2, label=marker_to_name.get(marker, f"Marker {marker}"))
                           handles[marker] = line
                       else:
                           plt.plot(x, y, color=color, linewidth=2)
           
               plt.legend(handles.values(), [marker_to_name.get(m, f"Marker {m}") for m in handles.keys()])
               plt.title("Boundary markers with legend")
               plt.axis("equal")
               plt.show()   

    def plot_1d_solution_with_markers(self, solution, mesh, ts, show=False, refinement_level=None):

        diff = self.diff
        class_name = self.problem.__class__.__name__
    
        x_values = mesh.coordinates().flatten()
        sorted_indices = np.argsort(x_values)
        x_sorted = x_values[sorted_indices]
    
        species = list(self.problem.chem_species()) 
        colors  = ['red', 'orange', 'blue', 'green', 'black']
    
        solutions = solution.split()
    
        plt.figure()
        for sol, sp, color in zip(solutions, species, colors):
            y_sorted = np.array([sol(Point(x)) for x in x_sorted])
            plt.plot(
                x_sorted,
                y_sorted,
                marker='o',
                markersize=1.0,   
                linestyle='-',
                linewidth=0.5,   
                color=color,
                label=sp
            )
    
        plt.xlabel('Distance (m)')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    
        base_folder = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}'
        )
    
        if refinement_level is None:
            plt.title(f'Solutions at t = {ts}')
            folder_path = os.path.join(base_folder, 'SolutionSpecies')
            filename = os.path.join(folder_path, f'solutions_t{ts}.png')
        else:
            plt.title(f'Solutions at t = {ts}, ref = {refinement_level}')
            folder_path = os.path.join(base_folder, 'RefinementPerTimestep', f't{ts}')
            filename = os.path.join(folder_path, f'solutions_t{ts}_ref{refinement_level}.png')
    
        os.makedirs(folder_path, exist_ok=True)
    
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
        if show:
            plt.show()
        else:
            plt.close()
    
        return filename
        
    def plot_1d_errors_with_refinement(self, g_list, mesh, cellmark, ts, show=False, refinement_level=None):

        diff = self.diff
        class_name = self.problem.__class__.__name__
    
        x_vertices = mesh.coordinates().flatten()       # size: num_vertices
        x_cells = []                                    # size: num_cells
        marker_positions = []
    
        for cell in cells(mesh):
            xm = cell.midpoint().x()
            x_cells.append(xm)
            if cellmark[cell]:
                marker_positions.append(xm)
    
        x_cells = np.array(x_cells)
        marker_positions = np.array(marker_positions) if marker_positions else None
    
        n_vertices = len(x_vertices)
        n_cells = len(x_cells)
        n_err = len(g_list[0])  
    
        if n_err == n_vertices:
            x = x_vertices
        elif n_err == n_cells:
            x = x_cells
        else:
            raise ValueError(
                f"Error array length {n_err} does not match #vertices ({n_vertices}) "
                f"or #cells ({n_cells})."
            )
    
        max_err_raw = 0.0
        for e in g_list:
            arr = np.asarray(e)
            if arr.size > 0:
                max_err_raw = max(max_err_raw, np.max(np.abs(arr)))
    
        if max_err_raw == 0.0:
            max_err_raw = 1.0
    
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
    
        species = list(self.problem.chem_species())    
        colors  = ['red', 'orange', 'blue', 'green', 'black']
    
        plt.figure()
        all_errors = []
    
        for err_arr, sp, color in zip(g_list, species, colors):
            err_arr = np.asarray(err_arr)
            if len(err_arr) != n_err:
                raise ValueError(
                    "Inconsistent error array lengths: "
                    f"expected {n_err}, got {len(err_arr)} for species {sp}"
                )
    
            err_norm = err_arr / max_err_raw
            err_sorted = err_norm[sorted_indices]
            all_errors.append(err_sorted)
    
            plt.plot(
                x_sorted,
                err_sorted,
                marker='o',
                markersize=1.0,  
                linestyle='-',
                linewidth=0.5,   
                color=color,
                label=sp
            )
    
        if marker_positions is not None and len(marker_positions) > 0:
            max_err = max(np.max(e) for e in all_errors) if all_errors else 1.0
            marker_y = np.full_like(marker_positions, max_err * 1.05, dtype=float)
    
            plt.scatter(
                marker_positions,
                marker_y,
                marker='x',
                s=5,             
                label='Refinement marker'
            )
    
        plt.xlabel('Distance (m)')
        plt.ylabel('Normalized error')
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    
        base_folder = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}'
        )
    
        if refinement_level is None:
            plt.title(f'Error at t = {ts}')
            folder_path = os.path.join(base_folder, 'ErrorSpecies')
            filename = os.path.join(folder_path, f'errors_t{ts}.png')
        else:
            plt.title(f'Error at t = {ts}, ref = {refinement_level}')
            folder_path = os.path.join(base_folder, 'ErrorRefinementPerTimestep', f't{ts}')
            filename = os.path.join(folder_path, f'errors_t{ts}_ref{refinement_level}.png')
    
        os.makedirs(folder_path, exist_ok=True)
    
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
        if show:
            plt.show()
        else:
            plt.close()
    
        return filename
    
    def plot1D(self, outflow, time_step):

        namesplot, nkey = self.problem.Components_plot()
        specie_names, equi_solid, kin_solid = namesplot
        colors = self.problem.Colordic()
    
        num_data_points = len(next(iter(outflow.values())))
        x_values = np.linspace(self.p['Linfx'], self.p['Lsupx'], num_data_points)

        if equi_solid:
            _ = self.problem.get_scaling_factor(equi_solid[0]) 
        elif kin_solid:
            _ = self.problem.get_scaling_factor(kin_solid[0])
    
        solid_mode = getattr(self.problem, "_solid_plot_mode", "mol_m3_bulk")
    
        if solid_mode == "bulk_vol_frac":
            equi_label = "Equilibrium solids volume fraction (m³/m³ bulk)"
            kin_label  = "Kinetic solids volume fraction (m³/m³ bulk)"
            equi_default_lim = (0.0, 0.35)
            kin_default_lim  = (0.0, 0.35)
        elif solid_mode == "benchmark_mol_m3":
            equi_label = "Equilibrium solids (mol/m³) [benchmark scaling]"
            kin_label  = "Kinetic solids (mol/m³) [benchmark scaling]"
            equi_default_lim = self.problem.get_axis_limit('y_lim_solid_equilibrium')
            kin_default_lim  = self.problem.get_axis_limit('y_lim_solid_kinetic')
        else: 
            equi_label = "Equilibrium solids (mol/m³ bulk)"
            kin_label  = "Kinetic solids (mol/m³ bulk)"
            equi_default_lim = self.problem.get_axis_limit('y_lim_solid_equilibrium')
            kin_default_lim  = self.problem.get_axis_limit('y_lim_solid_kinetic')

        AQUEOUS_AS_MOLKGW = True 
    
        if AQUEOUS_AS_MOLKGW:
            aq_label = "Concentration species (mol/kgw)"
            aq_scale = 1.0
            aq_ylim_key = 'y_lim_species' 
        else:
            por = self.problem.Initial_condition(0)[1].get('porosity', 0.35)
            rho_w = 1000.0
            Sw = 1.0
            aq_scale = rho_w * por * Sw
            aq_label = "Concentration species (mol/m³ bulk)"
            aq_ylim_key = 'y_lim_species'
    

        fig, ax1 = plt.subplots(figsize=(10, 6))
        axes = [ax1]
    
        for sp, conc in outflow.items():
            if sp in specie_names:
                ax1.plot(
                    x_values,
                    np.asarray(conc) * aq_scale,
                    label=sp,
                    color=colors.get(sp, None)
                )
    
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel(aq_label, color='b')
        ax1.tick_params('y', colors='b')
    
        if self.problem.get_axis_limit('lim-mod'):
            ax1.set_ylim(*self.problem.get_axis_limit(aq_ylim_key))
            ax1.set_xlim(self.p['Linfx'], self.p['Lsupx'])
    
        ax2 = None
        if equi_solid:
            ax2 = ax1.twinx()
            for sp, conc in outflow.items():
                if sp in equi_solid:
                    ax2.plot(
                        x_values,
                        np.asarray(conc),  
                        label=sp,
                        marker='+',
                        linestyle='-',
                        color=colors.get(sp, None)
                    )
            ax2.set_ylabel(equi_label, color='r')
            ax2.tick_params('y', colors='r')
            ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    
            if self.problem.get_axis_limit('lim-mod'):
                if solid_mode == "bulk_vol_frac":
                    ax2.set_ylim(*equi_default_lim)
                else:
                    ax2.set_ylim(*self.problem.get_axis_limit('y_lim_solid_equilibrium'))
    
            axes.append(ax2)
    
        ax3 = None
        if kin_solid:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
    
            for sp, conc in outflow.items():
                if sp in kin_solid:
                    ax3.plot(
                        x_values,
                        np.asarray(conc),
                        label=sp,
                        marker='+',
                        linestyle='-',
                        color=colors.get(sp, None)
                    )
            ax3.set_ylabel(kin_label, color='m')
            ax3.tick_params('y', colors='m')
    
            if self.problem.get_axis_limit('lim-mod'):
                if solid_mode == "bulk_vol_frac":
                    ax3.set_ylim(*kin_default_lim)
                else:
                    ax3.set_ylim(*self.problem.get_axis_limit('y_lim_solid_kinetic'))
    
            axes.append(ax3)
    
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
    
        ax1.legend(handles, labels, loc='center right')
        plt.title(f"Solution {time_step}  |  solids_mode={solid_mode}")
    
        diff = self.diff
        class_name = self.problem.__class__.__name__
        folder_path = os.path.join(self.p['folderresults'] + class_name + '/', f'Diff{diff}', 'Concentration-png')
        os.makedirs(folder_path, exist_ok=True)
    
        fig.tight_layout()
        file_name = f'solution1D-ts{time_step}.pdf'
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path, format='pdf', dpi=300)
    
        if self.p.get('shows_plot', False):
            plt.show()
        plt.close(fig)            


    def save_gplot_to_vtk(self, mesh, g_list, timestep, refinement, name="g_plot_overall"):
        class_name = self.problem.__class__.__name__
        diff = self.diff
    
        folder_path = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}',
            f'timestep_{timestep}',
            'refinement'
        )
        os.makedirs(folder_path, exist_ok=True)
    
        vtkfile_path = os.path.join(
            folder_path,
            f'{name}_refinement_{refinement}.pvd'
        )
    
        V0 = FunctionSpace(mesh, "DG", 0)
        g_fun = Function(V0)
        g_fun.rename(name, name)
    
        values = g_fun.vector().get_local()
    
        for cell in cells(mesh):
            idx = cell.index()
            cell_vals = []
    
            for g in g_list:
                try:
                    cell_vals.append(float(g[cell]))
                except Exception:
                    try:
                        cell_vals.append(float(g[idx]))
                    except Exception:
                        cell_vals.append(0.0)
    
            values[idx] = max(cell_vals) if cell_vals else 0.0
    
        g_fun.vector().set_local(values)
        g_fun.vector().apply("insert")
    
        vtkfile = File(vtkfile_path)
        vtkfile << g_fun
    
        print(
            f"Saved {name} for timestep {timestep} and refinement {refinement} "
            f"to {vtkfile_path}"
        )

    def save_markers_to_vtk(self, mesh, cellmark, timestep, refinement):
        class_name = self.problem.__class__.__name__
        diff = self.diff
    
        folder_path = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}',
            f'timestep_{timestep}',
            'refinement'
        )
        os.makedirs(folder_path, exist_ok=True)
    
        vtkfile_path = os.path.join(
            folder_path,
            f'cell_marking_refinement_{refinement}.pvd'
        )
    
        V0 = FunctionSpace(mesh, "DG", 0)
        marker_fun = Function(V0)
        marker_fun.rename("cell_marking", "cell_marking")
    
        values = marker_fun.vector().get_local()
    
        for cell in cells(mesh):
            idx = cell.index()
            try:
                values[idx] = float(cellmark[cell])
            except Exception:
                try:
                    values[idx] = float(cellmark[idx])
                except Exception:
                    try:
                        values[idx] = 1.0 if bool(cellmark[cell]) else 0.0
                    except Exception:
                        values[idx] = 0.0
    
        marker_fun.vector().set_local(values)
        marker_fun.vector().apply("insert")
    
        vtkfile = File(vtkfile_path)
        vtkfile << marker_fun
    
        print(
            f"Saved cell marking for timestep {timestep} and refinement {refinement} "
            f"to {vtkfile_path}"
        )


    def save_error_to_vtk(self, mesh, error_values, timestep, refinement, name="error"):
        class_name = self.problem.__class__.__name__
        diff = self.diff
    
        folder_path = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}',
            f'timestep_{timestep}',
            'refinement'
        )
        os.makedirs(folder_path, exist_ok=True)
    
        vtkfile_path = os.path.join(
            folder_path,
            f'{name}_refinement_{refinement}.pvd'
        )
    
        V0 = FunctionSpace(mesh, "DG", 0)
        err_fun = Function(V0)
        err_fun.rename(name, name)
    
        values = err_fun.vector().get_local()
    
        for cell in cells(mesh):
            idx = cell.index()
            try:
                values[idx] = float(error_values[cell])
            except Exception:
                try:
                    values[idx] = float(error_values[idx])
                except Exception:
                    values[idx] = 0.0
    
        err_fun.vector().set_local(values)
        err_fun.vector().apply("insert")
    
        vtkfile = File(vtkfile_path)
        vtkfile << err_fun
    
        print(
            f"Saved {name} for timestep {timestep} and refinement {refinement} "
            f"to {vtkfile_path}"
        )

    def save_profiles_h5(self, filename, data, x=None, t=None, attrs=None):
        import numpy as np
        import h5py
    
        def check_dict(d):
            keys = list(d.keys())
            n = len(d[keys[0]])
            for k in keys:
                a = np.asarray(d[k])
                if a.ndim != 1 or len(a) != n:
                    raise ValueError(f"Dataset {k} has shape {a.shape}, expected (n,) with n={n}")
            return keys, n
    
        with h5py.File(filename, "w") as h5:
            if attrs:
                for k, v in attrs.items():
                    h5.attrs[k] = v
    
            if x is not None and not isinstance(x, list):
                h5.create_dataset("x", data=np.asarray(x), compression="gzip", shuffle=True)
    
            if t is not None:
                h5.create_dataset("t", data=np.asarray(t), compression="gzip", shuffle=True)
    
            grp = h5.create_group("fields")
    
            if isinstance(data, dict):
                check_dict(data)
                for k, arr in data.items():
                    grp.create_dataset(k, data=np.asarray(arr), compression="gzip", shuffle=True)
    
            elif isinstance(data, list):
                for i, d in enumerate(data):
                    check_dict(d)
                    subgrp = grp.create_group(f"step_{i:04d}")
                    for k, arr in d.items():
                        subgrp.create_dataset(k, data=np.asarray(arr), compression="gzip", shuffle=True)
    
                    if isinstance(x, list):
                        subgrp.create_dataset("x", data=np.asarray(x[i]), compression="gzip", shuffle=True)
    
            else:
                raise TypeError("data must be a dict or a list of dicts")

    def save_overall_error_to_vtk(self, mesh, E_list, timestep, refinement, name="error_overall"):
        class_name = self.problem.__class__.__name__
        diff = self.diff
    
        folder_path = os.path.join(
            self.p['folderresults'] + class_name + '/',
            f'Diff{diff}',
            f'timestep_{timestep}',
            'refinement'
        )
        os.makedirs(folder_path, exist_ok=True)
    
        vtkfile_path = os.path.join(
            folder_path,
            f'{name}_refinement_{refinement}.pvd'
        )
    
        V0 = FunctionSpace(mesh, "DG", 0)
        err_fun = Function(V0)
        err_fun.rename(name, name)
    
        values = err_fun.vector().get_local()
    
        for cell in cells(mesh):
            idx = cell.index()
            cell_vals = []
    
            for E in E_list:
                try:
                    cell_vals.append(float(E[cell]))
                except Exception:
                    try:
                        cell_vals.append(float(E[idx]))
                    except Exception:
                        cell_vals.append(0.0)
    
            values[idx] = max(cell_vals) if cell_vals else 0.0
    
        err_fun.vector().set_local(values)
        err_fun.vector().apply("insert")
    
        vtkfile = File(vtkfile_path)
        vtkfile << err_fun
    
        print(
            f"Saved overall error for timestep {timestep} and refinement {refinement} "
            f"to {vtkfile_path}"
        )

    def plot_vertex_groups(self,mesh, iniv, uDval, labels):

            coords = mesh.coordinates()
            fig=plt.figure(figsize=(8, 6))
            dfplot(mesh, linewidth=0.5)
        
            color_list = ['bo', 'go', 'mo', 'co', 'yo', 'po']
            
            if isinstance(iniv[0], int):
                iniv = [iniv]  # Make it a list of one group
                if labels is None:
                    labels = ['Domain']
            else:
                if labels is None:
                    labels = [f'Domain {i+1}' for i in range(len(iniv))]
        
            for i in uDval:
                x, y = coords[i]
                plt.plot(x, y, 'ro', markersize=1)
        
            for idx, group in enumerate(iniv):
                color = color_list[idx % len(color_list)]
                for i in group:
                    x, y = coords[i]
                    plt.plot(x, y, color, markersize=1)
        
            all_indices = set(range(len(coords)))
            grouped_indices = set(np.concatenate(iniv))
            boundary_indices = set(uDval)
            other_indices = all_indices - grouped_indices - boundary_indices
        
            for i in other_indices:
                x, y = coords[i]
                plt.plot(x, y, 'ko', markersize=1)
        
            plt.axis('equal')
            plt.title("Vertex Domain and Boundary")
        
            handles = []
            from matplotlib.lines import Line2D
            handles.append(Line2D([0], [0], marker='o', color='w', label='Boundary (uDval)',
                                   markerfacecolor='r', markersize=5))
            for idx, label in enumerate(labels):
                handles.append(Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color_list[idx % len(color_list)][0], markersize=5))
            handles.append(Line2D([0], [0], marker='o', color='w', label='Other Vertices',
                                   markerfacecolor='k', markersize=3))
        
            plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        
            class_name = self.problem.__class__.__name__
            folder_path = os.path.join(self.p['folderresults'], class_name, 'Domain')
            os.makedirs(folder_path, exist_ok=True)
        
            file_name = f'vertex_groups-ts0.pdf'
            file_path = os.path.join(folder_path, file_name)
        
            fig.tight_layout()
            plt.savefig(file_path, format='pdf', dpi=300)
        
            if self.p.get('shows_plot', True):
                plt.show()
            else:
                plt.close()
                