"""
Script Name: Phreeqc + Fenics Reactive Transport. 
Author: Juan Giraldo
Email: juan.giraldo@csiro.au
Created: 2024-04-30
Version: Version 1.0
"""

import fenics
from dolfin import *
import numpy as np
from dolfin import solve
import copy
import os
import gmsh
import meshio
from meshcreator import BoundaryCreator
from Postprocess import Postprocess
import numbers
set_log_level(LogLevel.WARNING)   # or LogLevel.ERROR to be stricter


class DiffusionAdvectionModel(object): 

    def __init__(self, init_conc, inflow_conc,problem,params):
        self.p = params
        self.conc = init_conc
        self.inflow_conc = inflow_conc
        self.outflow = {}
        self.funperm = 0
        self.funporo = 0
        self.problem = problem
        self.species_keys = self.problem.chem_species()
        self.update_function_spaces_for_current_mesh()
        self.press = 0
        self.velocity = 0

    def update_function_spaces_for_current_mesh(self):
        n_esp = len(self.species_keys)
        self.W = self.Functionspace(n_esp, self.p)
        self.listver2dof = self.ver2dofmap(self.W, self.species_keys)
        self.dof_ind = self.GetDoF(self.W)
        self.vertex_ind = self.GetVertex(self.W, self.species_keys)      

    def build_chem_function_on_mesh(self, conc, inflow_conc, mesh, nkey):
        old_mesh = self.p['mesh']
        self.p['mesh'] = mesh
        W0 = self.Functionspace(len(nkey), self.p)
        fun = self.Fullarray2fun(conc, inflow_conc, W0)    
        self.p['mesh'] = old_mesh
        return fun, W0

    def interpolate_chemistry_to_mesh(self, conc, inflow_conc, mesh_new, nkey):
        """
        Given chemistry arrays (conc, inflow_conc) on some 'old' mesh,
        build a Function, project it to 'mesh_new', and extract new arrays
        (one value per vertex of mesh_new).
        """
        mesh_old = self.p['mesh']
        chem_fun_old, W_old = self.build_chem_function_on_mesh(conc, inflow_conc,mesh_old, nkey)
    
        old_mesh_backup = self.p['mesh']
        self.p['mesh'] = mesh_new
        W_new = self.Functionspace(len(nkey), self.p)
        chem_fun_new = Function(W_new)
    
        for i, sp in enumerate(nkey):
            V_new = W_new.sub(i).collapse()
            assign(chem_fun_new.sub(i), project(chem_fun_old.sub(i), V_new))

        species_vector_dict_new = self.functiontoarray(chem_fun_new, nkey, W_new)
    
        self.p['mesh'] = old_mesh_backup
    
        return species_vector_dict_new
    

    def fun_properties(self, innerval, boundaryval, properties):
        fun_con = {}
        outflow = {}
        properties_plot = [key for key, value in properties.items() if value]
        #concatenated_list = sum(self.problem.mechanism().values(), [])
        for val in properties_plot:
            #if val == 'SI_solid': val ='SI_'+concatenated_list[0]
            ival = innerval[val]
            bval = boundaryval[val]
            computed_value = self.singlefuntoplot(ival, bval)
            fun_con[val] = computed_value
            if self.p['dim'] == 1:
                W = FunctionSpace(self.p['mesh'], 'P', 1)   
                outflow.update(DiffusionAdvectionModel.functiontoarray(0, fun_con[val], [val], W))
            else:
                outflow = fun_con
        return outflow
    

    def get_boundary_dofs(self, W):
        boundary_markers = MeshFunction('size_t', self.p['mesh'], self.p['mesh'].topology().dim() - 1)
        boundary_markers.set_all(0)
        boundaries = CompiledSubDomain("on_boundary")
        boundaries.mark(boundary_markers, 1)
    
        boundary_dofs = DirichletBC(W, Constant(0), boundary_markers, 1).get_boundary_values().keys()
        return list(boundary_dofs)

    def Fullarray2fun(self, innervalues_ns, boundaryvalues_ns,W0):                
        nkey = self.problem.Components_plot()[1]
        innervalues = {}; boundaryvalues = {}        

        for tag, values_list in innervalues_ns.items():
            if tag in nkey:
                scaling_factor = self.problem.get_scaling_factor(tag)
                scaled_values = [value * scaling_factor for value in values_list]
                innervalues[tag] = scaled_values
        
        for tag, values_list in boundaryvalues_ns.items():
            if tag in nkey:
                scaling_factor = self.problem.get_scaling_factor(tag)
                scaled_values = [value * scaling_factor for value in values_list]
                boundaryvalues[tag] = scaled_values
   
        fun = self.array2function_gen(W0,innervalues,boundaryvalues,nkey)
        

        return fun
        
    def singlefuntoplot(self,inner_val,boundary_val):
        W = FunctionSpace(self.p['mesh'], 'P', 1)   
        nkey = ['whole'] 
        innervalues = {nkey[0]: inner_val}; boundaryvalues= {nkey[0]: boundary_val} 
        fun = self.array2function_gen(W, innervalues, boundaryvalues, nkey)
        return fun
    
    def plot_mesh_with_vertex_numbers(self,mesh):
        plt.figure()
        plot(mesh)
        plt.title("Mesh with Vertex Numbers")
    
        coordinates = mesh.coordinates()
        vertex_indices = range(mesh.num_vertices())
    
        for idx, coord in zip(vertex_indices, coordinates):
            plt.text(coord[0], coord[1], str(idx), fontsize=8, color='red')
    
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    def array2function_gen(self, W, innervalues, boundaryvalues, nkey):
        func = Function(W)
        interior_dofs, boundary_dofs, _, _ = self.GetDoF2(W)
    
        vec = func.vector().get_local()
        interior_values = vec[interior_dofs].copy()
        boundary_values = vec[boundary_dofs].copy()
    
        vertex_inner_index, vertex_boundary_index = self.GetVertex(W, nkey)
        v2d = self.ver2dofmap(W, nkey)
    
        interior_pos = {dof: i for i, dof in enumerate(interior_dofs)}
        boundary_pos = {dof: i for i, dof in enumerate(boundary_dofs)}
    
        for sp in nkey:

            if sp in innervalues:
                vals_in = innervalues[sp]
                for ind, vtx in enumerate(vertex_inner_index):
                    if ind >= len(vals_in):
                        break
                    dofmod = v2d[sp][vtx]
                    idx = interior_pos.get(dofmod, None)
                    if idx is not None:
                        interior_values[idx] = vals_in[ind]
            else:
                pass

            if boundaryvalues is not None and sp in boundaryvalues:
                vals_b = boundaryvalues[sp]
                for ind, vtx in enumerate(vertex_boundary_index):
                    if ind >= len(vals_b):
                        break
                    dofmod = v2d[sp][vtx]
                    idx = boundary_pos.get(dofmod, None)
                    if idx is not None:
                        boundary_values[idx] = vals_b[ind]
    
        combined_vector = np.zeros(func.vector().size())
        combined_vector[interior_dofs] = interior_values
        if boundaryvalues is not None:
            combined_vector[boundary_dofs] = boundary_values
    
        func.vector()[:] = combined_vector
        return func    

    def array2function(self, innervalues, boundaryvalues): 
        func = Function(self.W)
        interior_dofs,boundary_dofs,_,_ = self.dof_ind
        vec = func.vector().get_local()
        interior_values = vec[interior_dofs]
        boundary_values = vec[boundary_dofs]
        
        vertex_inner_index,vertex_boundary_index = self.vertex_ind
                
        for sp in self.species_keys:
            for ind,vtx in enumerate(vertex_inner_index):
                dofmod=self.listver2dof[sp][vtx]
                index_position = interior_dofs.index(dofmod)
                interior_values[index_position] = innervalues[sp][ind]
            for ind,vtx in enumerate(vertex_boundary_index):
                dofmod=self.listver2dof[sp][vtx]
                index_position = boundary_dofs.index(dofmod)
                boundary_values[index_position] = boundaryvalues[sp][ind]  
                
        combined_vector = np.zeros(func.vector().size())
        combined_vector[interior_dofs] = interior_values
        combined_vector[boundary_dofs] = boundary_values

        func.vector()[:] = combined_vector
        return func

    def function2array(self, func):
        vec = func.vector().get_local()
        interior_dofs, boundary_dofs, _, _ = self.dof_ind
    
        interior_set = set(interior_dofs)
        boundary_set = set(boundary_dofs)
    
        interior_val_by_dof = {dof: vec[dof] for dof in interior_dofs}
        boundary_val_by_dof = {dof: vec[dof] for dof in boundary_dofs}
    
        interior_dict_values = {}
        boundary_dict_values = {}
    
        for space_key, dofs in self.listver2dof.items():
            interior_dict_values[space_key] = [
                interior_val_by_dof[dof] for dof in dofs if dof in interior_set
            ]
            boundary_dict_values[space_key] = [
                boundary_val_by_dof[dof] for dof in dofs if dof in boundary_set
            ]
    
        return interior_dict_values, boundary_dict_values             

    def ver2dofmap(self, W, nkey):
          listdof = {}
          num_subspaces = W.num_sub_spaces()
          if num_subspaces == 0:
              spaces = [(W, nkey[0])] 
          else:
              spaces = [(W.sub(i), key) for i, key in enumerate(nkey) if i < num_subspaces]
          for space, key in spaces:
              dofmap = space.dofmap()  
              subspace_dofs = dofmap.dofs()
              dofs2vertex = vertex_to_dof_map(W)
              listdof[key] = [dofs2vertex[dof] for dof in subspace_dofs if dof < len(dofs2vertex)]  
          return listdof
      
    def assignfunction(self,inner_val,boundary_val,n_key,W):
        un = Function(W)   
        vertex_BC = self.vertex_ind[1]
        vertex_internal = self.vertex_ind[0]
      
        for spi, specie in enumerate(n_key):
            
            V = W.sub(spi).collapse()  
            u_sub = Function(V)
            ver2dof = vertex_to_dof_map(V)

            for i, idx in enumerate(vertex_internal):
                   dof_index = ver2dof[idx]  
                   u_sub.vector()[dof_index] = inner_val[specie][i]  #assuming the vertex index location in vertex_interal is in the same location than its respective value in inner_val. 
                   
            for i, idx in enumerate(vertex_BC):
                   dof_index = ver2dof[idx]  # Get DOF index for this vertex
                   u_sub.vector()[dof_index] = boundary_val[specie][i]  #assuming the vertex index location in vertex_interal is in the same location than its respective value in inner_val. 
        
            assigner = FunctionAssigner(W.sub(spi), V)
            assigner.assign(un.sub(spi), u_sub)       

        return un
    
    def solfun2(self, innervalues, boundaryvalues):   
        W = self.Functionspace(len(self.species_keys),p)
        un = self.assignfunction(innervalues,boundaryvalues,self.species_keys,W)
             
        return un
            
    def assignfunction2(self,inner_val,boundary_val,n_key,W):
        un = Function(W)   
        vertex_BC = self.verticesind[1]
        vertex_internal = self.verticesind[0]
        
        plt.figure()
        plt.figure(figsize=(10, 10))
        plot(p['mesh'], title="Finite Element Mesh")
        coordinates = p['mesh'].coordinates()
        plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red', s=50, zorder=10)  # Plot vertices
      
        for spi, specie in enumerate(n_key):
            
            V = W.sub(spi).collapse()  
            u_sub = Function(V)
            ver2dof = vertex_to_dof_map(V)

            for i, idx in enumerate(vertex_internal):
                   vertex_coordinates = p['mesh'].coordinates()[idx]

                   dof_index = ver2dof[idx]  
                   u_sub.vector()[dof_index] = inner_val[specie][i]  #assuming the vertex index location in vertex_interal is in the same location than its respective value in inner_val. 
            
                   plt.annotate(str(idx),
                                      vertex_coordinates,
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center')
                   plt.annotate(str(dof_index),
                                      vertex_coordinates,
                                      textcoords="offset points",
                                      xytext=(0, -20),
                                      ha='center',
                                      color='red')       
            
            for i, idx in enumerate(vertex_BC):
                   vertex_coordinates = p['mesh'].coordinates()[idx]

                   dof_index = ver2dof[idx]  # Get DOF index for this vertex
                   u_sub.vector()[dof_index] = boundary_val[specie][i]  #assuming the vertex index location in vertex_interal is in the same location than its respective value in inner_val. 
        
                   plt.annotate(str(idx),
                                      vertex_coordinates,
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center')
                   plt.annotate(str(dof_index),
                                      vertex_coordinates,
                                      textcoords="offset points",
                                      xytext=(0, -20),
                                      ha='center',
                                      color='green') 
        
        
            assigner = FunctionAssigner(W.sub(spi), V)
            assigner.assign(un.sub(spi), u_sub)       
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Mesh Vertices and DOFs')
        plt.legend(['Vertices'])
        plt.show()
        return un
    
    def functiontoarray(self,sol, n_key, W):
        u_vector = sol.vector().get_local() 
        u_concatenated = {}  
        if W.num_sub_spaces() == 0: 
            dofmap = W.dofmap().dofs() 
            vertex2dof = vertex_to_dof_map(W) 
            values_at_vertices = u_vector[dofmap][vertex2dof] 
            u_concatenated[n_key[0]] = values_at_vertices 
        else:
            for i, species in enumerate(n_key):
                subspace = W.sub(i)  
                dofmap = subspace.dofmap().dofs()  
                vertex2dof = vertex_to_dof_map(subspace.collapse())  
                subspace_values_vertex = u_vector[dofmap][vertex2dof]  
                u_concatenated[species] = subspace_values_vertex 
        return u_concatenated

    def Functionspace(self, n_key, p):  
        element_family = FiniteElement(p["TrialType"], p['mesh'].ufl_cell(), p["Ptrial"])     
        W = FunctionSpace(p['mesh'], MixedElement([element_family] * n_key) if n_key > 1 else element_family)
        return W
    
    def FunctionspaceDG(self, n_key, p):  
        element_family = FiniteElement('DG', p['mesh'].ufl_cell(), p["Ptrial"])     
        W = FunctionSpace(p['mesh'], MixedElement([element_family] * n_key) if n_key > 1 else element_family)
        return W
    
    def get_FunctionSpaces(self, n_key, p):
        V1 = FiniteElement('DG', p['mesh'].ufl_cell(), 1)
        W1 = FunctionSpace(p['mesh'], MixedElement([V1] * n_key) if n_key > 1 else V1)

        V2 = FiniteElement('CG', p['mesh'].ufl_cell(), 1)
        W2 = FunctionSpace(p['mesh'], MixedElement([V2] * n_key) if n_key > 1 else V2)

        return W1,W2

    def BoundaryConditions_species(self, W, p):
        uD, uN = self.problem.BoundaryValues_species(0, p)
        
        boundaries,_ = BoundaryCreator(p['mesh'],p).create_boundaries()
        bdid = {species: [int(tag) for tag in uD[species].keys()] for species in self.species_keys}
        DBC = []
    
        num_subspaces = W.num_sub_spaces()
        target_space = lambda i: W if num_subspaces == 0 else W.sub(i)
    
        for i, value in enumerate(self.species_keys):
            for index in bdid[value]:
                DBC.append(DirichletBC(target_space(i), uD[value][str(index)], boundaries, index))
    
        return DBC
     
    def BoundaryConditions_pressure(self, W, p):
        uD, uN = self.problem.BoundaryValues_pressure(0, p)
        boundaries,_ = BoundaryCreator(p).create_boundaries()
        bdid = [int(tag) for tag in uD.keys()]
        DBC = []
        num_subspaces = W
        target_space = lambda i: W  
        for index in bdid:
            DBC.append(DirichletBC(target_space(i), uD[str(index)], boundaries, index))
        return DBC
    
    
    def GetDoF(self, W):
        func = Function(W)
        boundaries,_ = BoundaryCreator(self.p['mesh'],self.p).create_boundaries()

        DBC = self.BoundaryConditions_species(W, self.p)
        boundary_dofs = set()
        for dbc in DBC:
            boundary_dofs.update(dbc.get_boundary_values().keys())
        all_dofs = set(range(func.vector().size()))
        interior_dofs = all_dofs - boundary_dofs 
        boundary_dict = {}
        interior_dict = {}
        if W.num_sub_spaces() == 0:
            key = next(iter(self.listver2dof))
            boundary_dict[key] = [dof for dof in self.listver2dof[key] if dof in boundary_dofs]
            interior_dict[key] = [dof for dof in self.listver2dof[key] if dof in interior_dofs]
        else:
            for space_key, dofs in self.listver2dof.items():
                boundary_dict[space_key] = [dof for dof in dofs if dof in boundary_dofs]
                interior_dict[space_key] = [dof for dof in dofs if dof in interior_dofs]
        return list(interior_dofs), list(boundary_dofs), interior_dict, boundary_dict

    def GetDoF2(self, W):
        func = Function(W)
        num_subspaces = W.num_sub_spaces()

        if num_subspaces == 0:  
            num_subspaces = 1 
            subspaces = [W]
        else:
            subspaces = [W.sub(i) for i in range(num_subspaces)]

        boundaries,_ = BoundaryCreator(self.p['mesh'],self.p).create_boundaries()
        uD, uN = self.problem.BoundaryValues_species(0, self.p)
        listmark = list(uD[next(iter(uD))].keys()) 
        markers = [int(s) for s in listmark]
        bc = []

        for subspace in subspaces:
            dim = subspace.ufl_element().value_size()
            zero_vector = Constant((0,)*dim) if dim > 1 else Constant(0)
            for marker in markers:
                bc.append(DirichletBC(subspace, zero_vector, boundaries, marker))

        boundary_dofs = [key for b in bc for key in b.get_boundary_values().keys()]
        all_dofs = set(range(func.vector().size()))
        interior_dofs = all_dofs - set(boundary_dofs)  

        boundary_dict = {}; interior_dict = {}
        for space_key, dofs in self.ver2dofmap(W, self.species_keys).items():
            boundary_dict[space_key] = [dof for dof in dofs if dof in boundary_dofs]
            interior_dict[space_key] = [dof for dof in dofs if dof in interior_dofs]

        return list(interior_dofs), list(boundary_dofs), interior_dict, boundary_dict


    def GetVertexChem(self):
        interior_dofs, boundary_dofs, _, _ = self.dof_ind
    
        interior_set = set(interior_dofs)
        boundary_set = set(boundary_dofs)
    
        ref_key = self.species_keys[0] if hasattr(self, "species_keys") else next(iter(self.listver2dof))
        ref_dofs = self.listver2dof[ref_key]
    
        vertex_inner_index = [i for i, dof in enumerate(ref_dofs) if dof in interior_set]
        vertex_boundary_index = [i for i, dof in enumerate(ref_dofs) if dof in boundary_set]
    
        return vertex_inner_index, vertex_boundary_index

    def GetVertex(self,W,nkey):
        interior_dofs,boundary_dofs,interior_dict,boundary_dict = self.GetDoF(W)
        vertex_boundary_index = [i for i, dof in enumerate(next(iter(self.ver2dofmap(W,nkey).values()))) if dof in boundary_dofs]
        vertex_inner_index = [i for i, dof in enumerate(next(iter(self.ver2dofmap(W,nkey).values()))) if dof in interior_dofs]
    
        return vertex_inner_index, vertex_boundary_index  
  
    def Solver2(self,u, du,Fi,bcs):
         # Solver parameters
           snes_solver_parameters = {"nonlinear_solver": "snes",
                                      "snes_solver": {"linear_solver": "mumps",
                                         "maximum_iterations": 20,
                                                "report": True,
                                                "error_on_nonconvergence": True}}
           J = derivative(Fi, u, du)
           problem = NonlinearVariationalProblem(Fi, u, bcs, J)
           solver = NonlinearVariationalSolver(problem)
           solver.parameters.update(snes_solver_parameters)
           solver.solve()
           return u
    

    def Solverfun(self, A, l, delta, solvertype, ArgsDic):
        a_mat = assemble(A)
        b_vec = assemble(l)
    
        if solvertype == 'DirectDG':
            e_k = []
            solve(A == l, delta)
            u_k = Function(delta.function_space()) 
            u_k.assign(delta) 
            return e_k, delta
        if solvertype == "IterativeCGDG":
            solver = PETScKrylovSolver("gmres", "ilu")
            solver.parameters["relative_tolerance"] = 1e-8
            solver.parameters["absolute_tolerance"] = 1e-12
            solver.parameters["maximum_iterations"] = 2000
            solver.parameters["monitor_convergence"] = True
            solver.solve(a_mat, delta.vector(), b_vec)
    
            e_k, u_k = delta.split(True)
            return e_k, u_k
    
        elif solvertype == "DirectCGDG":
            solver = LUSolver(a_mat, "mumps")   # or "umfpack" / "default"
            solver.solve(delta.vector(), b_vec)
    
            e_k, u_k = delta.split(True)
            return e_k, u_k

    def BoundaryConditions_pressure(self, W, p):
        uD = self.problem.pressure()['uD_pressure']
        boundaries,_ = BoundaryCreator(self.p['mesh'],self.p).create_boundaries()
        DBC = []
        for bd, value in uD.items():
            DBC.append(DirichletBC(W, Constant(value), boundaries, int(bd)))
        return DBC
    
    def getF(self, u_, v, u_n_, p, ArgsDic):

        F_i     = 0
        kappa   = ArgsDic['kappa']
        f       = self.problem.forcing_term(p)       
        beta_raw = ArgsDic.get('beta_vec', 0)
        beta_vec = None if beta_raw == 0 else beta_raw    
        
        scheme   = self.p['TIM']
        dt       = Constant(p['dt'])

        for i, sp in enumerate(self.species_keys):
            
            if self.p['Unsteady']:
                if scheme == 'BE':#self.problem['Temporal']
                    th1 = ArgsDic['theta_n1']; th0 = ArgsDic['theta_n']
                    F_i += (1.0/dt) * v[i] * (th1*u_[i] - th0*u_n_[i]) * dx
    
            F_i += kappa[sp] * dot(grad(u_[i]), grad(v[i])) * dx    
            if beta_vec is not None:
                F_i += dot(beta_vec, grad(u_[i])) * v[i] * dx
    
            F_i += - f[sp] * v[i] * dx
    
        return F_i
    
    def getBoundaryMeasures(self, mesh, W, gD_tags, boundaries,p):

         ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
         dS = Measure('dS', domain=mesh, subdomain_data=boundaries)
    
         dsD = []
         num_subspaces = W.num_sub_spaces()
        
         for i, value in enumerate(self.species_keys):
             gD_esp = gD_tags[value]
             if len(gD_esp) == 0:
               dsD.append(None)
             else:
               dsDi = ds(gD_esp[0])
               for tag in gD_esp[1:]:
                   dsDi += ds(tag)
               dsD.append(dsDi)
               
         dsN = none_list = [None] * num_subspaces  #No tags for Neumann have been implemented yet
        
         return ds,dS,dsD,dsN

    def _dorfler_mark_modified(self, mesh, E, g, params, cell_markers):
        if params['REF_TYPE'] != 1:
            return False

        rgind = np.argsort(g)[::-1]
        g0 = params['REFINE_RATIO']**2 * E**2
        Ntot = mesh.num_cells()
    
        h_chem = params['h_chem']
        h_min_allowed = params.get('h_transport_min_factor', 0.5) * h_chem
        chem_refine_tol = params.get('chem_refine_tol', 0.5)

        hvals = np.array([cell.h() for cell in cells(mesh)])
    
        eligible_mask = hvals > h_min_allowed
    
        g_total = np.sum(g)
        g_eligible = np.sum(g[eligible_mask])
    
        if g_total <= 0.0:
            return False
    
        eligible_ratio = g_eligible / g_total
    
        if eligible_ratio < chem_refine_tol:
            return False
    
        scum = 0.0
        cutoff = None
        marked_any = False
    
        for nj in range(Ntot):
            cid = rgind[nj]
    
            if not eligible_mask[cid]:
                continue
    
            scum += g[cid]
            cell_markers[cid] = True
            cutoff = g[cid]
            marked_any = True
    
            if scum >= g0:
                for nk in range(nj + 1, Ntot):
                    cid2 = rgind[nk]
    
                    if not eligible_mask[cid2]:
                        continue
    
                    if cutoff - (1 + params['tolref']) * g[cid2] < 0:
                        cell_markers[cid2] = True
                    else:
                        break
                break
    
        if not marked_any:
            print("[refine] stop: no eligible cells could be marked.")
            return False
    
        return True
    
    def refinement_multi_species_union_modified(self, mesh, E_list, g_list, params,
                                       scale_theta=False):
    
        nspecies = len(g_list)
        assert nspecies == len(E_list)
    
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
    
        theta_global = params['REFINE_RATIO']
    
        if scale_theta:
            theta_s = theta_global / float(nspecies)
        else:
            theta_s = theta_global
    
        any_marked = False
    
        for E, g in zip(E_list, g_list):
            params['REFINE_RATIO'] = theta_s
            marked_this_species = self._dorfler_mark_modified(mesh, E, g, params, cell_markers)
            any_marked = any_marked or marked_this_species
    
        params['REFINE_RATIO'] = theta_global
    
        if not any_marked:
            print("[refine] no cells marked, returning same mesh.")
            return mesh, cell_markers
    
        max_nodes = params.get("max_nodes", 150000)
        max_nodes_stop_fraction = params.get("max_nodes_stop_fraction", 0.9)
    
        n_nodes = mesh.num_vertices()
        node_threshold = int(max_nodes_stop_fraction * max_nodes)
    
        if n_nodes >= node_threshold:
            print(f"[refine] stop: mesh has {n_nodes} nodes, "
                  f"which is above the buffer threshold {node_threshold} "
                  f"(max_nodes = {max_nodes}).")
            return mesh, cell_markers
    
        meshr = refine(mesh, cell_markers)
        return meshr, cell_markers

    def error_compute(self,Eudis,pos,ArgsDic):
          #Local Error criteria
          V = Eudis.function_space()
          mesh = V.mesh()

          n = FacetNormal(mesh)
          h = CellDiameter(mesh)
    
          PC2 = FunctionSpace(mesh,"DG", 0)
          c  = TestFunction(PC2)             # p.w constant fn
          g_plot = Function(PC2)
    
          Eu_trial = Eudis ;  Eu_test= Eudis  
    
          g = assemble(self.gh(Eu_trial,Eu_test,grad(Eu_trial),grad(Eu_test),n,h,c,pos,ArgsDic))
    
          g = np.abs(g*1)
          g_plot.vector()[:] = g
          Ee = np.sum(np.array(g))
          E = sqrt(Ee)
                      
          return E,g,g_plot
      

    def get_TestTrial(self,V,ArgsDic):   
        if ArgsDic['Solver'] == 'Iterative':
            V1 = V.sub(0).collapse() 
            V2 = V.sub(1).collapse() 
     
            e = TrialFunction(V1)  ;  u = TrialFunction(V2)
            w = TestFunction(V1)   ;  v = TestFunction(V2)       
        else:
            e,  u = TrialFunctions(V);  w,  v = TestFunctions(V)
        return e,u,v,w
         
    def bh_prima(self,du,v,n,h,pos,ArgsDic):   
        val = 0
        C2 = 1
        if ArgsDic['Unsteady']: 
           val +=   ArgsDic['theta_n1']*ArgsDic['C1']* inner(du,v)*ArgsDic['dx']
           C2  =   ArgsDic['C2']
        val += C2 * self.bh_dar(du,v,n,h,pos,ArgsDic)
        return val

    def bh_dar(self,u,w,n,h,pos,ArgsDic):
        gamma =    ArgsDic['gammac']
        beta_vec = ArgsDic['beta_vec'];
        kappa = ArgsDic['kappa'][self.species_keys[pos]];
        val=0;
        if gamma:      val += inner(u, gamma*w)*ArgsDic["dx"] 
        #if kappa:      val += self.bh_sip(u,w,n,h,ArgsDic,pos,Constant(kappa))
        if kappa:      val += self.bh_sip(u,w,n,h,ArgsDic,pos,kappa)

        if beta_vec:   val += self.bh_upw(u,w,n,h,ArgsDic,pos,beta_vec)
        if kappa== 0 and  beta_vec== 0:
           raise Exception('you must specify at least one, kappa or beta_vec')
        return val

    def rhs(self,w,f,n,h,pos,ArgsDic):  #fully rhs form
        val = 0;  
        if ArgsDic['Unsteady']:
           Crhs2 = ArgsDic["Crhs2"] ; Crhs3 = ArgsDic["Crhs3"] 
           #val += Crhs2 * inner(ArgsDic['theta_n']*ArgsDic['u_n'],w)*ArgsDic["dx"] 
           val += Crhs2 * inner(ArgsDic['theta_n']*ArgsDic['u_n'],w)*ArgsDic["dx"] 

        val +=  Crhs3 * inner(f,w)*ArgsDic["dx"]  #Source term
        val +=  Crhs3 * self.rhs_dar(w,n,h,pos,ArgsDic)
        return val

    def bh_sip(self,u,v,n,h,ArgsDic,pos,kappa):
        #SWIP
        eta = ArgsDic['eta']/(h**ArgsDic['psuperc']);    
        dsD   = ArgsDic['dsD'][pos];
        dS    = ArgsDic["dS"];
        ds    = ArgsDic["ds"];
        dx = ArgsDic["dx"] 
     
        val = inner(kappa*grad(u), grad(v))*ArgsDic["dx"] 
        val += -inner(dot(self.avgK(kappa,n,u), n('+')), jump(v))*ArgsDic["dS"] #consistency
        
        val += ArgsDic['epsilon']*inner(jump(u), dot(self.avgK(kappa,n, v), n('+')))*ArgsDic["dS"] #adjoint consistency
        val += inner(avg(eta)*self.gamma_K_i(kappa,n)*jump(u), jump(v))*ArgsDic["dS"] #penalty
    
        val += -inner(v, dot(kappa*grad(u), n))*dsD #consistency
        val += ArgsDic['epsilon']*inner(u, dot(kappa*grad(v), n))*dsD #adjoint consistency
        val += inner(eta*self.gamma_K_b(kappa,n)*u, v)*dsD  #penalty
        
        return val
    
    def bh_upw(self,u,v,n,h,ArgsDic,pos,beta_vec):
        #UPW
        dS    = ArgsDic["dS"];
        ds    = ArgsDic["ds"];
        dx    = ArgsDic["dx"];
        dsD   = ArgsDic['dsD'][pos];

        val = inner(dot(beta_vec,grad(u)), v)*dx 
        val += 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*v, u)*dsD
        val -= inner(dot(beta_vec('+'),n('+'))*avg(v), jump(u))*dS
        val += 0.5*inner(abs(dot(beta_vec('+'),n('+')))*jump(u), jump(v))*dS   
        return val
                  
    def rhs_dar(self,w,n,h,pos,ArgsDic):
        #kappa = Constant(ArgsDic['kappa'][self.species_keys[pos]]);
        kappa = ArgsDic['kappa'][self.species_keys[pos]];

        beta_vec = ArgsDic['beta_vec'];        
        val = 0  # Add Neumann contribution if required
    
        if kappa:     val += self.rhs_sip(w,n,h,ArgsDic,pos,kappa)  #SIP RHS contribution
        if beta_vec:  val += self.rhs_adv(w,n,ArgsDic,pos,beta_vec)  #Upwinding RHS contribution
        if kappa==0 and beta_vec==0:
           raise Exception('you must specify at least one, kappa or beta_vec')
        return val

    def rhs_sip(self,w,n,h,ArgsDic,pos,kappa):
        eta = ArgsDic['eta']/(h**ArgsDic['psuperc']);
        dsD   = ArgsDic['dsD'][pos]; uD = ArgsDic['uD'][self.species_keys[pos]];
        gD = next(iter(uD.values())) #fix if there is more thatn one identifier (more than one DBC)
        val   = 0;
        val   = ArgsDic['epsilon']*inner(dot(kappa*grad(w),n), gD)*dsD
        val  += inner(self.gamma_K_b(kappa,n)*eta*w, gD)*dsD
        return val
    
    def rhs_adv(self,w,n,ArgsDic,pos,beta_vec):
        ds   = ArgsDic['ds']; uD = ArgsDic['uD'][self.species_keys[pos]];
        dsD   = ArgsDic['dsD'][pos]
        gD = next(iter(uD.values())) #fix if there is more thatn one identifier (more than one DBC)
        val = 0.5*inner((abs(dot(beta_vec,n))-dot(beta_vec,n))*w, gD)*dsD    
        return val

    def TimeintegratorMethod(self,TIM,params):
       if TIM == 'BE':
          C1=1;  C2 = params['dt']
          Crhs1= 0 ; Crhs2 = 1; Crhs3 = params['dt']
       else:
           raise Exception('Time integrator Method not accepted')
       return [C1,C2,Crhs1,Crhs2,Crhs3]


    # DG norms
    # ----------------------------------------------------------------------#
    
    def gh(self,e,w,grade,gradw,n,h,c,pos,ArgsDic): #fully discrete gh primer norm
         val = 0
         C2 = 1
         if ArgsDic['Unsteady']:
             val =   ArgsDic['C1']*self.ghV0(e,w,c,ArgsDic)   #revisar
             C2  =   ArgsDic['C2']
         val +=  C2*self.gh_dar(e,w,grade,gradw,n,h,c,pos,ArgsDic)
         
         return val
    
    def ghV0(self,e,w,c,ArgsDic):
          val = inner(e,w*c)*ArgsDic["dx"]
          return val

    #space-dependent norms
    def gh_dar(self,e,w,grade,gradw,n,h,c,pos,ArgsDic):
     #norm sip
        val =0
        kappa = ArgsDic['kappa_norm'][self.species_keys[pos]];

        if ArgsDic['gamma_norm']:
           val += (ArgsDic['gamma_norm'])*inner(e, w)*c*ArgsDic["dx"] 
        if kappa:  val += self.gh_sip(e,w,grade,gradw,c,n,h,pos,ArgsDic)
        if ArgsDic['beta_norm']:
            val += self.gh_upw(e,w,grade,gradw,c,n,h,ArgsDic)
        if ArgsDic['kappa_norm'] == 0 and ArgsDic['beta_norm'] == 0:
            raise Exception('you must specify at least one, kappa or beta_vec')
          
        return val
    
    def gh_upw(self,e,w,grade,gradw,c,n,h,ArgsDic):
     #norm upw
        beta_vec = ArgsDic['beta_norm'];  beta_inf = ArgsDic['beta_inf'];
        val = 0.5*inner(abs(dot(beta_vec,n))*e, w*c)*ArgsDic["ds"]  #semi norm
        val += ArgsDic['eta_adv']*0.5*inner(abs(dot(beta_vec('+'),n('+')))*jump(e), jump(w)*avg(c))*ArgsDic["dS"]  # upw
        val += h/beta_inf*inner(dot(beta_vec, grade), c*dot(beta_vec, gradw))*ArgsDic["dx"]  #inf sup
        val += beta_inf*inner(e,w*c)*ArgsDic["dx"] 
        return val
    
    
    def gh_sip(self,e,v,grade,gradv,c,n,h,pos,ArgsDic):
       # kappa = Constant(ArgsDic['kappa_norm'][self.species_keys[pos]]);
        kappa = ArgsDic['kappa_norm'][self.species_keys[pos]];

        eta = ArgsDic['eta']/(h**ArgsDic['psuperc']); 
        val = inner(kappa*grade, gradv*c)*ArgsDic["dx"] 
        val += inner(self.gamma_K_i(kappa,n)*avg(eta)*jump(e), jump(v)*avg(c))*ArgsDic["dS"]
        val += inner((self.gamma_K_b(kappa,n)*eta)*e, v*c)*ArgsDic["ds"]  #penalty
    
        return val

    # ----------------------------------------------------------------------#
    # Auxiliary functions for DG norms
    # ----------------------------------------------------------------------#
    #Stabilization constant for diffusion
    def opt_sta(self,mesh,dx,ds,dS):
        PC  = FunctionSpace(mesh,"DG",0)
        c   = TestFunction(PC)
        sta = Function(PC)
        AK  = np.array(assemble(2*Constant(1)*avg(c)*dS + Constant(1)*c*ds))
        VK  = np.array(assemble(Constant(1)*c*dx))
        sta.vector()[:]=AK/(VK)
        del AK,VK
    
        return sta
    
    def gamma_K_b(self,kappa,n):
        gamma_K_b = dot(n, kappa*n)
        return gamma_K_b
        
    def gamma_K_i(self,kappa,n):
        delta_Kn_plus = dot(n('+'), kappa('+')*n('+'))
        delta_Kn_min = dot(n('-'), kappa('-')*n('-'))
    
        gamma_K_i = delta_Kn_plus*delta_Kn_min/(delta_Kn_plus + delta_Kn_min)
        return gamma_K_i
    
    def avgK(self,kappa,n, v):
        delta_Kn_plus = dot(n('+'), kappa('+')*n('+'))
        delta_Kn_min = dot(n('-'), kappa('-')*n('-'))
    
        omega_min = delta_Kn_plus/(delta_Kn_plus + delta_Kn_min)
        omega_plus = delta_Kn_min/(delta_Kn_plus + delta_Kn_min)    
        val = omega_min*(kappa*grad(v))('-') + omega_plus*(kappa*grad(v))('+')
        return val

    def system_Matrices_DG(self,u,w,p,pos,ArgsDic):
        # ----------------------------------------------------------------------#
        # Matrix Formulation - DG
        # ----------------------------------------------------------------------#
        # # Trial / Test  functions
        # u    --- Trial V1 //  w    --- Test V1
        # ----------------------------------------------------------------------#
        n = FacetNormal(p['mesh'])
        h = CellDiameter(p['mesh'])
        
        f = self.problem.forcing_term(p)[self.species_keys[pos]]
        F = self.rhs(w,f,n,h,pos,ArgsDic)  
        Bh= self.bh_prima(u,w,n,h,pos,ArgsDic)
        return Bh, F

    def system_Matrices_ASFEM(self,e,u,w,v,p,pos,ArgsDic):

        # ----------------------------------------------------------------------#
        # Matrix Formulation - Primal problem ASFEM
        # ----------------------------------------------------------------------#
        # # Trial / Test  functions
        # e   --- Trial V1 //  w    --- Test V1
        # du  --- Trial V2 //  v    --- Test V2
        # ----------------------------------------------------------------------#
        n = FacetNormal(p['mesh'])
        h = CellDiameter(p['mesh'])
        
        f = self.problem.forcing_term(p)[self.species_keys[pos]]
        F = self.rhs(w,f,n,h,pos,ArgsDic)  

        # Grand matrix form
        Gh   =  self.gh(e,w,grad(e),grad(w),n,h,Constant(1),pos,ArgsDic)
        Bh   =  self.bh_prima(u,w,n,h,pos,ArgsDic)
        Bht  =  self.bh_prima(v,e,n,h,pos,ArgsDic)  

        A = Gh+Bh+Bht
    
        return A,F
    
    def run_dar(self,un,ts):
                
        n_esp = len(self.species_keys)
        iterR=0;
        refi =1;
        mesh0 = self.p['mesh']
         
        if self.p["ASFEM"]==0 and self.p['DG']==0 :  
                 self.p["FEM"]=1; 
        #         self.p["MAX_REF"]=1
        elif self.p["ASFEM"]==0 and self.p['DG']==1: 
                 self.p["FEM"]=0; 
        #         self.p["MAX_REF"]=1
        else:
                 self.p["FEM"]=0; self.p['DG']==0 
                      
        W  = self.Functionspace(n_esp,self.p) #chem species + pot    
        Wdg  = self.FunctionspaceDG(n_esp,self.p) #chem species + pot  
        W1,W2 = self.get_FunctionSpaces(n_esp,self.p) #W1 discontionus, W2 contionus

        diffusivities = self.problem.diffusionCoeff()
        
        if self.p['meshdomain'] == 'Furrowdomain' or self.p['meshdomain'] == 'Leachingdomain':
            kappac = { key: self.p['D_tensor_rich'] for key in diffusivities }
            water_content_n1 = self.p['water_content_n1']
            water_content_n = self.p['water_content_n'] 
            beta_vec = self.p['velocity_rich'] 
        else:
            kappac = diffusivities
            beta_vec = self.problem.AdvectiveCoeff(self.p)
            water_content_n1 = Constant(1)#self.funporo  self.p['water_content_n1']#Constant(1)
            water_content_n = Constant(1)#self.funporo#Constant(1)#self.p['water_content_n1']#Constant(1)
            
        beta_inf = 1
        beta_norm = beta_vec

        uD, uN = self.problem.BoundaryValues_species(0, self.p) # check bcs function lines above
        boundaries,boundarynames = BoundaryCreator(self.p['mesh'],self.p).create_boundaries()
        
        if self.p['plotboundaries']:  Postprocess(self.p,self.problem).plot_boundaries(self.p['mesh'], boundaries, boundarynames)

        gD_tags = {species: [int(tag) for tag in uD[species].keys()] for species in self.species_keys}
        dx = Measure('dx', domain=self.p['mesh']) 
        ds,dS,dsD,dsN = self.getBoundaryMeasures(self.p['mesh'],W, gD_tags, boundaries,self.p)
        eta = (self.p['pdegree']+1)*(self.p['pdegree']+ self.p['dim'])/self.p['dim'] * self.opt_sta(self.p['mesh'], dx , ds, dS) 
        
        ArgsDic = {'mesh':self.p['mesh'],'dx':dx,'dS':dS,'ds':ds,'dsD':dsD,'dsN': dsN,
                   'uD':uD,'uN':uN,
                   'kappa':kappac,'kappa_norm':kappac,'gammac':0,'gamma_norm':0,
                   'beta_inf':beta_inf,'beta_vec':beta_vec,'beta_norm':beta_norm,
                   'theta_n1':water_content_n1, 'theta_n':water_content_n, 
                   'eta':eta,'epsilon':self.p['epsilon'],'eta_adv':self.p['eta_adv'],
                   'psuperc':self.p['superpen'],
                   'Solver':self.p['SolverASFEM']}
                
        self.problem.Temporal(self.p)
        
        v_ = split(TestFunction(W)) if n_esp > 1 else TestFunction(W) 

              
        if self.p["Unsteady"]:
            self.p['t'] = self.p['tini']
        else:    
            self.p['dt']=1; self.p['t'] = 0; self.p['T']=self.p['dt']

        C1be,C2be,Crhs1be,Crhs2be,Crhs3be = self.TimeintegratorMethod('BE',self.p) #it starts with BE to compute the first time step
        UnsteadyDic = {'Unsteady':self.p['Unsteady'],'dt':self.p['dt'],\
                       'C1': C1be,'C2': C2be,'Crhs1': Crhs1be,\
                       'Crhs2': Crhs2be,'Crhs3': Crhs3be,\
                       'u_n': un, 'TIM':'BE', 't':self.p['tini']}
        ArgsDic.update(UnsteadyDic)  #check u_n=u_n otherwise u_n:u_k0 check V2  problems for non_linear cases

        u_n_ = un.split()

        u_DG = Function(Wdg)
        u_CGproj = Function(W)
        sol_ASFEM = Function(W)
        u_n = Function(W)
        E_val=[];g_val=[]

        if self.p['DG']:
            for i, value in enumerate(self.species_keys):
                Vh = Wdg.sub(i).collapse()
                VhCG = W.sub(i).collapse()

                uDG_n= project(u_n_[i],Vh)
                ArgsDic.update({'u_n':uDG_n})
                u = TrialFunction(Vh)
                w = TestFunction(Vh)
                uk  = Function(Vh)

                adg,ldg = self.system_Matrices_DG(u,w,self.p,i,ArgsDic)
                _,solDG = self.Solverfun(adg,ldg,uk,self.p['SolverDG'],ArgsDic)
                #_,solDG = self.Solverfun(adg,ldg,uk,'DirectDG',ArgsDic)

                u_CGproj = project(solDG,VhCG)
                assign(u_DG.sub(i), solDG)
                assign(u_n.sub(i), u_CGproj)

        if self.p['FEM']:
                u  = Function(W)
                u_ = split(u)

                du = TrialFunction(W)     
                F = self.getF(u_, v_, u_n_, self.p, ArgsDic)      
                bcs = self.BoundaryConditions_species(W,self.p)
                u_n = self.Solver2(u, du,F,bcs)               
                    
        if self.p['ASFEM']:
                
                E_val=[]; g_val=[] 
                for i, value in enumerate(self.species_keys):
                    Vh1 = W1.sub(i).collapse()
                    Vh2 = W2.sub(i).collapse()
                    mixed_element = MixedElement([Vh1.ufl_element(), Vh2.ufl_element()])
                    Vh = FunctionSpace(Vh1.mesh(), mixed_element)
                    u_n_[i].set_allow_extrapolation(True)  # Important to allow extrapolation

                    uDG_n= project(u_n_[i],Vh1)
                    ArgsDic.update({'u_n':uDG_n})
                    
                    e,u,v,w = self.get_TestTrial(Vh,ArgsDic)
                    delta = Function(Vh)
                 
                    a,l= self.system_Matrices_ASFEM(e,u,w,v,self.p,i,ArgsDic)
                    e_k,solCG  = self.Solverfun(a,l,delta,self.p['SolverASFEM'],ArgsDic)
                    assign(u_n.sub(i), solCG)
                    
                    [E,g,g_plot] = self.error_compute(e_k,i,ArgsDic)
                    E_val.append(E)
                    g_val.append(g)
                    
       
        return u_n,[E_val,g_val]