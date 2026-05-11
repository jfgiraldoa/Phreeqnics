"""
Script Name: Phreeqc + Fenics Reactive Transport 
Author: Juan Giraldo
Email: jfgiraldo@gmail.com
Created: 2024-04-30
Version: Version 1.0
Dependencies: Fenics version 2019 + phreeqcpy
"""

from DA_transport import DiffusionAdvectionModel
from Phreeqc_reaction import ReactionModel
from fenics import *
from dolfin import plot
from meshcreator import MeshGenerator,BoundaryCreator
import time
import matplotlib.pyplot as plt
from Postprocess import Postprocess
import numpy as np
import h5py
from dolfin import SpatialCoordinate, project, FunctionSpace
import os
import re

class ResultsWriter:
    def __init__(self, filename, mesh, V_for_pressure, rho=1000.0, g=9.81, vertical_axis="x"):
        """
        vertical_axis: for 1D meshes, 'x' is the only coordinate.
                       for 2D, use 'y' if y is elevation; for 3D, use 'z'.
        """
        self.f = h5py.File(filename, "w")
        self.rho = float(rho)
        self.g = float(g)

        dof_coords = V_for_pressure.tabulate_dof_coordinates()
        dim = mesh.geometry().dim()
        dof_coords = dof_coords.reshape((-1, dim))

        if dim == 1:
            z = dof_coords[:, 0]
        elif dim == 2:
            z = dof_coords[:, 1] if vertical_axis == "y" else dof_coords[:, 0]
        else:
            z = dof_coords[:, 2] if vertical_axis == "z" else dof_coords[:, 1]

        self.f.create_dataset("dof_coords", data=dof_coords)
        self.f.create_dataset("elevation_z_at_dofs", data=z)
        self.z = z

    def write_timestep(self, ts, calcite, porosity, pressure_fun):
        tkey = f"t_{ts:.6e}".replace("+", "").replace(".", "_")
        grp = self.f.create_group(tkey)

        grp.create_dataset("Calcite", data=np.asarray(calcite))
        grp.create_dataset("porosity", data=np.asarray(porosity))

        p_arr = pressure_fun.vector().get_local()
        grp.create_dataset("pressure", data=p_arr)

        head_arr = p_arr / (self.rho * self.g) + self.z
        grp.create_dataset("head", data=head_arr)

    def close(self):
        self.f.close()


class CoupledPostprocessController:
    def __init__(self, owner):
        self.owner = owner
        self.p = owner.p
        self.problem = owner.problem
        self.post = owner.postprocess
        self.diff = getattr(owner, 'diff', self.p.get('diff', None))

    def init_timestep_pvd_files(self):
        if self.p.get('dim', 1) != 2:
            return []
        if not self.p.get('save_pvd_timestep_2d', False):
            return []
        return self.post.initialize_pvd_files()

    def init_iteration_pvd_file(self):
        if self.p.get('dim', 1) != 2:
            return None
        if not self.p.get('save_pvd_iteration', False):
            return None
        outdir = self.p.get("output_dir", ".")
        return File(f"{outdir}/solution_iterations_2d.pvd")

    def init_marker_pvd_file(self):
        if self.p.get('dim', 1) != 2:
            return None
        if not self.p.get('save_pvd_markers_2d', False):
            return None
        outdir = self.p.get("output_dir", ".")
        return File(f"{outdir}/refinement_markers_2d.pvd")

    def plot_iteration_mesh_png(self, mesh):
        if self.p.get('view_png', False):
            plot(mesh)


    def save_solution_to_postprocess_timestep_folder(self, uk, ts, iterR, species_keys):
        import os
    
        solutions = uk.split(deepcopy=True)
    
        class_name = self.problem.__class__.__name__
    
        base_results_folder = os.path.join(
            self.p['folderresults'],
            class_name
        )
        os.makedirs(base_results_folder, exist_ok=True)
    
        diff = getattr(self.owner, 'diff', None)
        if diff is None:
            diff = self.p.get('diff', None)
        if diff is None:
            diff = getattr(self.problem, 'diff', None)
        if diff is None:
            raise ValueError("Could not determine diff from owner, p, or problem")
    
        diff_folder_name = f"Diff{diff}"
        diff_folder = os.path.join(base_results_folder, diff_folder_name)
        os.makedirs(diff_folder, exist_ok=True)
    
        if isinstance(ts, float):
            timestep_str = f"{ts:.6f}".rstrip('0').rstrip('.')
        else:
            timestep_str = str(ts)
        timestep_str = timestep_str.replace('.', 'p')
    
        timestep_folder = os.path.join(
            diff_folder,
            f'timestep_{timestep_str}'
        )
        os.makedirs(timestep_folder, exist_ok=True)
        
        for i, sol_i in enumerate(solutions):
            species_name = species_keys[i]
    
            species_folder = os.path.join(timestep_folder, species_name)
            os.makedirs(species_folder, exist_ok=True)
    
            sol_i.rename(species_name, species_name)
    
            vtkfile_path = os.path.join(
                species_folder,
                f'solution_refinement_{iterR}.pvd'
            )
    
            vtkfile = File(vtkfile_path)
            vtkfile << (sol_i, float(ts))
    
            print(
                f"Saved {species_name} solution for timestep {ts}, "
                f"refinement {iterR} to {vtkfile_path}"
            )
            
    def final_refinement_step(self, uk, mesh, ts, g_list, oldmesh, cellmark, max_ref):
        if not self.p.get("show_plot_iteration", False):
            return
        if self.p.get('dim', 1) != 1:
            return

        Postprocess(self.p, self.problem).plot_1d_solution_with_markers(
            uk, mesh, ts, show=True
        )

        if max_ref > 1:
            Postprocess(self.p, self.problem).plot_1d_errors_with_refinement(
                g_list, oldmesh, cellmark, ts
            )

    def iteration_snapshot(self, uk, mesh, ts, iterR, g_list, oldmesh, cellmark, max_ref):
        if not self.p.get("show_plot_snapshots", False):
            return
        if self.p.get('dim', 1) != 1:
            return

        Postprocess(self.p, self.problem).plot_1d_solution_with_markers(
            uk, mesh, ts, refinement_level=iterR
        )

        if max_ref > 1:
            Postprocess(self.p, self.problem).plot_1d_errors_with_refinement(
                g_list, oldmesh, cellmark, ts, refinement_level=iterR
            )

    def timestep_reactive(self, sol, plotval, ts, pvd_files, dar_model, outflow1d):
        
        self.post.postprocessplots(sol, plotval, ts, pvd_files)

        if not self.p.get('show_plot_timestep', False):
            return

        if self.problem.pressure()['Compute_velocity']:
            self.post.plot_fields(ts, dar_model.pressure, dar_model.velocity)

        if self.p.get('dim', 1) == 1:
            outdata = outflow1d(sol, ts)
            self.post.plot1D(outdata, ts)
        else:
            self.post.plot2Dspecies(sol, ts)

    def timestep_nonreactive(self, sol, ts):
        if not self.p.get('show_plot_timestep', False):
            return
        if self.p.get('dim', 1) == 2:
            self.post.plot2Dspecies(sol, ts)

    def end_step(self, sol, plotval, ts, pvd_files, dar_model):
        if not self.p.get('show_plot_end', False):
            return

        self.post.postprocessplots(sol, plotval, ts, pvd_files)

        if self.problem.pressure()['Compute_velocity']:
            self.post.plot_fields(ts, dar_model.pressure, dar_model.velocity)

    def write_2d_marker_pvd(self, mesh, cellmark, ts, iterR):
        if self.p.get('dim', 1) != 2:
            return
        if not self.p.get('save_pvd_markers_2d', False):
            return
        if mesh is None or cellmark is None:
            return
    
        Postprocess(self.p, self.problem).save_markers_to_vtk(
            mesh, cellmark, ts, iterR
        )

    def write_2d_error_pvd(self, mesh, error_values, ts, iterR, name="error"):
        if self.p.get('dim', 1) != 2:
            return
        if mesh is None or error_values is None:
            return
    
        Postprocess(self.p, self.problem).save_error_to_vtk(
            mesh, error_values, ts, iterR, name=name
        )
    
    def write_2d_gplot_pvd(self, mesh, g_list, ts, iterR):
        if self.p.get('dim', 1) != 2:
            return
        if not self.p.get('save_pvd_markers_2d', False):
            return
        if mesh is None or g_list is None:
            return
    
        Postprocess(self.p, self.problem).save_gplot_to_vtk(
            mesh, g_list, ts, iterR
        )
    
    def write_2d_overall_error_pvd(self, mesh, E_list, ts, iterR):
        if self.p.get('dim', 1) != 2:
            return
        if mesh is None or E_list is None:
            return
        if not self.p.get('save_pvd_markers_2d', False):
            return
    
        Postprocess(self.p, self.problem).save_overall_error_to_vtk(
            mesh, E_list, ts, iterR
        )
class CoupledModelDAR(object):

    def __init__(self, nshifts, initial_conditions, processes,problem,params):
        self.p = params
        self.problem = problem
        self.nshifts = nshifts
        self.TotalDOFS = [] if self.p['MAX_REF'] == 1 else {}
        self.snaplist = self.p.get('snap_time_list',[10]) 
        self.save_ts_list = sorted(set(self.p.get("save_ts_list",self.snaplist )))
        self.mesh = MeshGenerator(self.p).Create_mesh(problem)
        self.verticesind = DiffusionAdvectionModel(0,0,self.problem,self.p).vertex_ind# in 0: Boundary index, in 1: inner index
        self.nvertices  =  self.mesh.num_vertices()
        self.reaction_model = ReactionModel(self.nvertices,self.verticesind,initial_conditions, processes,problem,self.p)
        self.reaction_model.make_initial_state()
        self.dar_model = DiffusionAdvectionModel(self.reaction_model.init_conc,self.reaction_model.inflow_conc, self.problem,self.p)
        self.component_names = self.reaction_model.component_names
        self.results = {}
        self.results_time = {}
        self.species_keys = self.problem.chem_species()
        self.properties_key = self.problem.properties_key()
        self.reaction = any(lst for lst in problem.mechanism().values())
        self.postprocess = Postprocess(self.p,self.problem)
        self.nvar = problem.nx
        self.accumulated_list = [None] * (self.nvar if self.p.get('accumulate_solution') else 1)
        self.pp = CoupledPostprocessController(self)
        for name in self.component_names:
             self.results[name] = []
    
    def sol_accumulate(self, outval):
       positions = [-i for i in range(1, self.nvar + 1)]
       
       if self.p['accumulate_solution']:
           for i in range(self.nvar):
               if self.accumulated_list[i] is None:
                   self.accumulated_list[i] = {key: [] for key in outval.keys()}       
           for key, value in outval.items():
               for i, pos in enumerate(positions):
                   self.accumulated_list[i][key].append(value[pos])            
    
                
    def save_transport_timestep(self, uk, ts):
    
        if self.p['dim'] != 1:
            return
    
        entry = {}
    
        V0 = uk.sub(0).function_space().collapse()
        x = V0.tabulate_dof_coordinates().reshape((-1, self.p['dim']))[:, 0]
        order = np.argsort(x)
        entry["dist"] = x[order].tolist()
    
        for i, key in enumerate(self.species_keys):
            Vi = uk.sub(i).function_space().collapse()
            ui = Function(Vi)
            assign(ui, uk.sub(i))
            entry[key] = ui.vector().get_local()[order].tolist()
    
        self.accum_transport_timestep[ts] = entry
            

    def save_transport_snap(self, uk, ts, refinement_level, oldmesh, cell_mark, g_list=None):
    
        if self.p['dim'] != 1:
            return
    
        ts_targets = set(self.save_ts_list + [self.nshifts - 1])
        if ts not in ts_targets:
            return
    
        ts_key = str(ts)
    
        if ts_key not in self.accum_transport_snaps:
            self.accum_transport_snaps[ts_key] = {}
    
        entry = {}
    
        V0 = uk.sub(0).function_space().collapse()
        x_nodes = V0.tabulate_dof_coordinates().reshape((-1, self.p['dim']))[:, 0]
        order = np.argsort(x_nodes)
    
        entry["dist"] = x_nodes[order].tolist()
    
        for i, key in enumerate(self.species_keys):
            Vi = uk.sub(i).function_space().collapse()
            ui = Function(Vi)
            assign(ui, uk.sub(i))
            entry[key] = ui.vector().get_local()[order].tolist()

        x_cells = []
        marked_x = []
    
        for cell in cells(oldmesh):
            xm = cell.midpoint().x()
            x_cells.append(xm)
            if cell_mark[cell]:
                marked_x.append(xm)
    
        x_cells = np.array(x_cells, dtype=float)
        x_cells.sort()
    
        marked_x = np.array(marked_x, dtype=float)
        if marked_x.size > 0:
            marked_x.sort()
    
        entry["x_cells"] = x_cells.tolist()
        entry["marked_x"] = marked_x.tolist()
    

        if g_list is not None:
            g_saved = []
            for g in g_list:
                arr = np.asarray(g, dtype=float)
                g_saved.append(arr.tolist())
            entry["g_list"] = g_saved
    
        self.accum_transport_snaps[ts_key][refinement_level] = entry

    
    def outflow1D(self,usol, time_step): #TODO move outflow1D to Postprocess (avoiding circular call) 
        namesplot, nkey = self.problem.Components_plot()  
        W = DiffusionAdvectionModel.Functionspace(0, len(nkey), self.p)
        outflow = DiffusionAdvectionModel.functiontoarray(0, usol, nkey, W) 
        return outflow

    def run_coupled(self):
        n_esp = len(self.species_keys)
        nkey = self.problem.Components_plot()[1]
    
        self.accum_transport_timestep = {}
        self.accum_transport_snaps = {}
    
        state = self._initialize_coupled_state(n_esp, nkey)
    
        for ts in range(self.nshifts):
            plotval = self._prepare_material_properties()
    
            self._prepare_timestep_transport_mesh(ts)
    
            result = self._run_transport_with_refinement(
                ts=ts,
                un=state["un"],
            )
                    
            sol, un = self._update_chemistry_after_transport(
                ts=ts,
                n_esp=n_esp,
                nkey=nkey,
                mesh0=result["mesh0"],
                uk=result["uk"],
                plotval=plotval,
                pvd_files=state["pvd_files"],
            )
    

            state["sol"] = sol
            state["un"] = un
            state["plotval"] = plotval
            
        self._postprocess_end(
            sol=state["sol"],
            plotval=state["plotval"],
            ts=ts,
            pvd_files=state["pvd_files"],
        )
    
        transport_accum = 0
        if self.p['dim'] == 1 and self.p['ASFEM'] != 0 and self.p['accumulate_solution_adapta']:
            transport_accum = {
                "timestep": self.accum_transport_timestep,
                "snaps": self.accum_transport_snaps,
            }
    
        return state["sol"], state["plotval"], self.accumulated_list, transport_accum, self.TotalDOFS 

    
    
    def _initialize_coupled_state(self, n_esp, nkey):
        W = self.dar_model.Functionspace(len(nkey), self.p)
        un = self.dar_model.array2function(
            self.reaction_model.conc,
            self.reaction_model.inflow_conc
        )
        sol0 = self.dar_model.Fullarray2fun(
            self.reaction_model.conc,
            self.reaction_model.inflow_conc,
            W
        )
        
        pvd_files = self.pp.init_timestep_pvd_files()
        self._iter_pvd = self.pp.init_iteration_pvd_file()
        self._marker_pvd = self.pp.init_marker_pvd_file()
    
        V_p = FunctionSpace(self.p['mesh'], 'P', 1)
    
        return {
            "W": W,
            "un": un,
            "sol0": sol0,
            "pvd_files": pvd_files,
            "V_p": V_p,
            "sol": sol0,
            "plotval": [],
        }
    
    def _prepare_material_properties(self):
        perm0 = self.problem.Initial_condition(0)[1]['permeability']
        porodom = self.reaction_model.conc['porosity']
        poroini = self.reaction_model.inflow_conc['porosity']
    
        n = len(porodom)
        nin = len(poroini)
        perm = [[perm0] * n, [perm0] * nin]
        funperm = self.dar_model.singlefuntoplot(perm[0], perm[1])
    
        funporo = self.dar_model.singlefuntoplot(porodom, poroini)
    
        self.reaction_model.properties_udpate(perm)
        self.dar_model.funperm = funperm
        self.dar_model.funporo = funporo
    
        if any(value is True for value in self.properties_key.values()):
            plotval = self.dar_model.fun_properties(
                self.reaction_model.conc,
                self.reaction_model.inflow_conc,
                self.properties_key
            )
        else:
            plotval = []
    
        return plotval
    
    
    def _prepare_timestep_transport_mesh(self, ts):
        self._current_ts = ts
        self._mesh0 = self.p['mesh']
    
        print(f"TimeStep: {ts}")
    
        if self.p['ASFEM']:
            self.p['Nx'] = self.p['Nx_transp']
            self.p['Ny'] = self.p['Ny_transp']
            self.p['mesh'] = MeshGenerator(self.p).Create_mesh_new(self.problem)
        else:
            self.p['MAX_REF'] = 1
    
    
    def _run_transport_with_refinement(self, ts, un):
        iterR = 0
        refi = 1
        total_dofs = 0
        maxref = self.p['MAX_REF']
    
        uk = None
        refp = None
        oldmesh = None
        cellmark = None
        g_list = None
        refined_now = True
        final_ref_step = False
        E_list = None
        ts_dofs = []

    
        while (iterR < maxref) and (refi == 1):
            step = self._run_one_transport_iteration(ts, iterR, un)
            uk = step["uk"]
            refp = step["refp"]
            ndofs = step["num_dofs"]
            total_dofs += ndofs
            ts_dofs.append(ndofs)    
    
            if iterR == maxref - 1:
                refi = 0
                refined_now = False
                final_ref_step = True
            else:
                refine = self._handle_refinement_step(refp)
                oldmesh = refine["oldmesh"]
                cellmark = refine["cellmark"]
                g_list = refine["g_list"]
                refined_now = refine["refined_now"]
                E_list = refine["E_list"]
    
                if not refined_now:
                    refi = 0
    
                final_ref_step = (iterR == maxref - 1) or (not refined_now)
    
            if final_ref_step:
                self._handle_final_refinement_step(
                    uk=uk,
                    ts=ts,
                    g_list=g_list,
                    oldmesh=oldmesh,
                    cellmark=cellmark,
                )
    
            self._postprocess_iteration_outputs(
                uk=uk,
                ts=ts,
                iterR=iterR,
                oldmesh=oldmesh,
                cellmark=cellmark,
                g_list=g_list,
                E_list=E_list,
                species_keys=self.species_keys,
            )
            
            if self.p['view_pvd']:
                self.pp.save_solution_to_postprocess_timestep_folder(
                    uk,
                    ts=ts,
                    iterR=iterR,
                    species_keys=self.species_keys,
                )
    
            if refi == 1:
                iterR += 1
    
    
        if maxref == 1:
            while len(self.TotalDOFS) <= ts:
                self.TotalDOFS.append(None)
            self.TotalDOFS[ts] = ts_dofs[0] if ts_dofs else 0
        else:
            self.TotalDOFS[ts] = ts_dofs
        return {
            "uk": uk,
            "refp": refp,
            "iterR": iterR,
            "oldmesh": oldmesh,
            "cellmark": cellmark,
            "g_list": g_list,
            "E_list": E_list,
            "refined_now": refined_now,
            "total_dofs": total_dofs,
            "mesh0": self._mesh0,
        }
    
    
    def _run_one_transport_iteration(self, ts, iterR, un):
        start_time = time.time()
    
        uk, refp = self.dar_model.run_dar(un, ts)
        subspace_0 = uk.sub(0).function_space().mesh()
    
        self.pp.plot_iteration_mesh_png(subspace_0)
    
        num_dofs = subspace_0.num_vertices()
        dt = float(self.p['dt'])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        return {
            "uk": uk,
            "refp": refp,
            "num_dofs": num_dofs,
            "elapsed_time": elapsed_time,
        }
    

    def _handle_refinement_step(self, refp):
        oldmesh = None
        cellmark = None
        g_list = None
        E_list = None
        refined_now = True
    
        E_list = refp[0]
        g_list = refp[1]
        oldmesh = self.p['mesh']

        self.p['h_chem'] = self.problem.lsupx / self.p['Nx_chem']

        self.p['mesh'], cellmark = self.dar_model.refinement_multi_species_union_modified(
            self.p['mesh'], E_list, g_list, self.p, scale_theta=False
        )
        newmesh = self.p['mesh']
        refined_now = (newmesh.num_cells() > oldmesh.num_cells())

        return {
            "oldmesh": oldmesh,
            "cellmark": cellmark,
            "g_list": g_list,
            "E_list": E_list,
            "refined_now": refined_now,
        }
    
    def _handle_final_refinement_step(self, uk, ts, g_list, oldmesh, cellmark):
        if self.p['dim'] == 1 and self.p['ASFEM'] and self.p['accumulate_solution_adapta']:
            self.save_transport_timestep(uk, ts)
    
        self.pp.final_refinement_step(
            uk=uk,
            mesh=self.p['mesh'],
            ts=ts,
            g_list=g_list,
            oldmesh=oldmesh,
            cellmark=cellmark,
            max_ref=self.p["MAX_REF"],
        )

    
    def _postprocess_iteration_outputs(self, uk, ts, iterR, oldmesh, cellmark, g_list, E_list, species_keys):
        self._postprocess_iteration_accumulation(
            uk=uk,
            ts=ts,
            iterR=iterR,
            oldmesh=oldmesh,
            cellmark=cellmark,
            g_list=g_list,
        )
    
        self._postprocess_iteration_snapshots(
            uk=uk,
            ts=ts,
            iterR=iterR,
            oldmesh=oldmesh,
            cellmark=cellmark,
            g_list=g_list,
        )
    
        self._postprocess_iteration_2d_markers(
            ts=ts,
            iterR=iterR,
            oldmesh=oldmesh,
            cellmark=cellmark,
            E_list=E_list,
            g_list=g_list,

        )

    def _postprocess_iteration_accumulation(self, uk, ts, iterR, oldmesh, cellmark, g_list):
        if self.p['dim'] == 1 and self.p['ASFEM'] and self.p["MAX_REF"] > 1 and self.p['accumulate_solution_adapta']:
            self.save_transport_snap(
                uk,
                ts,
                iterR,
                oldmesh,
                cellmark,
                g_list if g_list is not None else None
            )
    
    
    def _postprocess_iteration_snapshots(self, uk, ts, iterR, oldmesh, cellmark, g_list):
        if ts not in self.snaplist:
            return
    
        self.pp.iteration_snapshot(
            uk=uk,
            mesh=self.p['mesh'],
            ts=ts,
            iterR=iterR,
            g_list=g_list,
            oldmesh=oldmesh,
            cellmark=cellmark,
            max_ref=self.p["MAX_REF"],
        )
    
    def _postprocess_iteration_2d_markers(self, ts, iterR, oldmesh, cellmark, E_list, g_list):
        self.pp.write_2d_marker_pvd(
            mesh=oldmesh,
            cellmark=cellmark,
            ts=ts,
            iterR=iterR,
        )
        self.pp.write_2d_gplot_pvd(
            mesh=oldmesh,
            g_list=g_list,
            ts=ts,
            iterR=iterR,
        )
        
    def _update_chemistry_after_transport(self, ts, n_esp, nkey, mesh0, uk, plotval, pvd_files):

        self.p['mesh'] = mesh0
        W = self.dar_model.Functionspace(n_esp, self.p)
        u = Function(W)

        if self.p['ASFEM'] == 0:
            W = self.dar_model.Functionspace(n_esp, self.p)
            u = Function(W)
            assign(u, uk)
        else:
            for i, key in enumerate(self.species_keys):
                Vh = W.sub(i).collapse()
                assign(u.sub(i), project(uk.sub(i), Vh))

        if self.reaction:
            W0 = self.dar_model.Functionspace(len(nkey), self.p)
            v_index = self.dar_model.GetVertexChem()

            species_vector_dict, boundary_dic = self.dar_model.function2array(u)

            if self.p['DG'] == 0 and self.p['ASFEM'] == 0:
                tol_zero = 1e-30
                for k in self.species_keys:
                    if k in species_vector_dict:
                        species_vector_dict[k] = [
                            0.0 if abs(v) < tol_zero else v
                            for v in species_vector_dict[k]
                        ]

            self.reaction_model.modify(species_vector_dict, ts, v_index)
            self.reaction_model.modify_inflow(boundary_dic, ts, v_index)

            filtered_dict = {
                k: self.reaction_model.conc[k]
                for k in self.species_keys
                if k in self.reaction_model.conc
            }

            if self.p['accumulate_solution']:
                self.sol_accumulate(self.reaction_model.conc)

            un = self.dar_model.array2function(filtered_dict, boundary_dic)
            sol = self.dar_model.Fullarray2fun(
                self.reaction_model.conc,
                self.reaction_model.inflow_conc,
                W0,
            )

            self._postprocess_timestep(
                sol=sol,
                plotval=plotval,
                ts=ts,
                pvd_files=pvd_files,
            )

        else:
            sol = u.copy()
            un = u.copy()
            self._postprocess_timestep_nonreactive(
                sol=sol,
                ts=ts,
            )
    
        return sol, un
    
    
    def _postprocess_timestep(self, sol, plotval, ts, pvd_files):
        self.pp.timestep_reactive(
            sol=sol,
            plotval=plotval,
            ts=ts,
            pvd_files=pvd_files,
            dar_model=self.dar_model,
            outflow1d=self.outflow1D,
        )
    
    
    def _postprocess_timestep_nonreactive(self, sol, ts):
        self.pp.timestep_nonreactive(sol, ts)
    
    
    def _postprocess_end(self, sol, plotval, ts, pvd_files):
        self.pp.end_step(
            sol=sol,
            plotval=plotval,
            ts=ts,
            pvd_files=pvd_files,
            dar_model=self.dar_model,
        )

        