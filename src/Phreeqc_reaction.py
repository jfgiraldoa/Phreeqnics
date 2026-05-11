"""
Script Name: Phreeqc + Fenics Reactive Transport. Built on top of Phreeqpy scripts
Author: Juan Giraldo
Email: juan.giraldo@csiro.au
Created: 2024-04-30
Version: Version 1.0
"""

import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
import copy
import os
from Postprocess import Postprocess

class ReactionModel(object):

    def __init__(self, nvertices,verticesind,initial_conditions, processes,problem,params):
        if processes > nvertices:
            raise ValueError('Number of processes needs to be less or equal '
                             'than number of cells. %d processes %d cells.'
                             % (processes, nvertices))
        if processes < 1:
            raise ValueError('Need at least one process got %d' % processes)
        self.problem = problem
        self.params = params
        self.parallel = False
        if processes > 1:
            self.parallel = True
        self.nvertices = nvertices
        self.verticesind = verticesind
        self.initial_conditions = initial_conditions
        self.processes = processes
        self.inflow_conc = {}
        self.init_conc = {}
        self.conc = {}
        self.component_names = []
        self.calculators = []
        self.cell_ranges = []
        self._init_calculators()
        self.make_initial_state()
        self.permeability ={}
        self.porosity ={}

    def _init_calculators(self):
        root_calculator = PhreeqcCalculator(self.nvertices,self.verticesind,self.initial_conditions,self.problem,self.params)
        self.calculators = [root_calculator]
        self.cell_ranges = [(0, self.nvertices)]

    def make_initial_state(self):
        self.inflow_conc = self.calculators[0].inflow_conc
        self.init_conc = self.calculators[0].init_conc
        self.component_names = self.calculators[0].component_names
        self.conc = self.calculators[0].conc
        self.permeability = self.calculators[0].permeability
        self.porosity = self.calculators[0].porosity_ini


    def modify(self, new_conc, ts, v_index):
        mode = self.params.get('chemistry_mesh_mode', 'fixed')

        if mode == 'fixed':
            self.conc = {}
            for name in self.component_names:
                self.conc[name] = []

            for cell_range, calculator in zip(self.cell_ranges, self.calculators):
                i0, i1 = cell_range
                current_conc = {
                    name: value[i0:i1] for name, value in new_conc.items()
                }
                calculator.modify(current_conc, ts, v_index)

            for calculator in self.calculators:
                conc = calculator.get_modified()
                for name in self.component_names:
                    self.conc[name].extend(conc[name])

            return conc

        else:
            raise NotImplementedError(
                f"chemistry_mesh_mode='{mode}' is not implemented. "
                "Use 'fixed' or 'transport_full'."
            )

    def modify_inflow(self, new_conc, ts, v_index):
        mode = self.params.get('chemistry_mesh_mode', 'fixed')

        if mode == 'fixed':
            self.inflow_conc = {}
            for name in self.component_names:
                self.inflow_conc[name] = []

            for cell_range, calculator in zip(self.cell_ranges, self.calculators):
                i0, i1 = cell_range
                current_conc = {
                    name: value[i0:i1] for name, value in new_conc.items()
                }
                calculator.modify_inflow(current_conc, ts, v_index)

            for calculator in self.calculators:
                inflowconc = calculator.get_modified_inflow()
                for name in self.component_names:
                    self.inflow_conc[name].extend(inflowconc[name])

            return inflowconc

        else:
            raise NotImplementedError(
                f"chemistry_mesh_mode='{mode}' is not implemented. "
                "Use 'fixed'"
            )
  
    def properties_udpate(self,perm):    
        self.conc['permeability'] = perm[0]
        self.inflow_conc['permeability'] = perm[1]
        self.porosity = self.conc['porosity']
        
class ScriptGenerator:
    def __init__(self, problem, params):
        self.problem = problem
        self.params = params
        
    def new_phases_definition(self):
        phases = self.problem.phases_definition()
        return phases.strip().split("\n") if phases else []
   

    def generate_solution_script(self, cell, content, idx=None, ph_val=None, ph_charge=None):
        conditions = self.problem.Conditions()
        lines = [f"SOLUTION {cell}"]
    
        if ph_charge is None:
            charge_flag = conditions.get('pH_charge', False)
        else:
            charge_flag = ph_charge
    
        if ph_val is None:
            ph_value = conditions.get('pH', None)
            if isinstance(ph_value, list) and idx is not None:
                ph_used = ph_value[idx]
            else:
                ph_used = ph_value
        else:
            ph_used = ph_val
    
        if ph_used is not None:
            if charge_flag:
                lines.append(f"pH {ph_used} charge")
            else:
                lines.append(f"pH {ph_used}")
    
        for key, value in conditions.items():
            if key in ('pH', 'pH_charge'):
                continue
            val = value[idx] if (isinstance(value, list) and idx is not None) else value
            lines.append(f"{key} {val}")
    
        if content:
            lines.append(content)
    
        return "\n".join(lines) + "\n"

    def ini_solid_solution(self):  
        solid_solution_moles = self.problem.solid_precipitation(0)[1]
        
        if not solid_solution_moles:
            return []
        
        script_lines = ["SOLID_SOLUTIONS 0"]
        
        for solid_solution, components in solid_solution_moles.items():
            script_lines.append(f"    {solid_solution}")
            script_lines.extend(f"        -comp {comp} {value}" for comp, value in components.items())
        
        return "\n".join(script_lines).strip().split("\n")


    def boundary_conditions(self):
        boundary_values, _ = self.problem.BoundaryValues_species(0, 0)
        extras = self.problem.extra_species()
        positions = list(next(iter(boundary_values.values())))
        script_lines = []
    
        for i, position in enumerate(positions):
            content_lines = []
            for sp in boundary_values:
                value = boundary_values[sp][position]
                content_lines.append(f"{sp}    {value}")
            for sp, val in extras.items():
                content_lines.append(f"{sp}    {val}")
                    
            content = "\n".join(content_lines)
    
            script_line = self.generate_solution_script(
                cell=i,
                content=content,
                idx=0
            )
            script_lines.append(script_line)
    
        return script_lines


    def ini_conditions(self):
        species = self.problem.chem_species()
        (values, properties) = self.problem.Initial_condition(0)
        extras = self.problem.extra_species()
    
        ph_prop        = properties.get('pH', None)
        ph_charge_prop = properties.get('pH_charge', False)
    
        first_value = values[species[0]]
        num_components = len(first_value) if isinstance(first_value, list) else 1
    
        scripts = []
    
        for i in range(num_components):
            content_lines = [
                f"{sp}    {values[sp][i] if num_components > 1 else values[sp]}"
                for sp in species
            ]
            content_lines += [f"{sp}    {val}" for sp, val in extras.items()]
            content = "\n".join(content_lines)
    
            if isinstance(ph_prop, list):
                ph_i = ph_prop[i]
            else:
                ph_i = ph_prop
    
            if isinstance(ph_charge_prop, list):
                ph_charge_i = ph_charge_prop[i]
            else:
                ph_charge_i = ph_charge_prop
    
            script = self.generate_solution_script(
                cell=10 + i,
                content=content,
                idx=i,
                ph_val=ph_i,
                ph_charge=ph_charge_i
            ).strip().split("\n")
    
            scripts.append(script)
    
        return scripts[0] if num_components == 1 else scripts

    
    def mechanisms(self):
        equilibrium_solid = self.problem.mechanism()['equilibrium_solid']
        kinetic_solid = self.problem.mechanism()['kinetic_solid']
        exchange = self.problem.mechanism()['exchange']
        solid_volume = self.problem.solid_precipitation(0)[0]
        dissolve_only = self.problem.solid_precipitation(0)[2]
        
        script_lines = []
        
        db_text_getter = None
        if hasattr(self.problem, "phreeqc_database_text") and callable(getattr(self.problem, "phreeqc_database_text")):
            db_text_getter = self.problem.phreeqc_database_text
        
        if db_text_getter:
            db_text = db_text_getter()
            if db_text and str(db_text).strip():
                script_lines.append(str(db_text).strip())
        
        if equilibrium_solid:
            first_value = next(iter(solid_volume.values()))
            is_vector = isinstance(first_value, list)
            num_components = len(first_value) if is_vector else 1
            for i in range(num_components):
                phase_id = i + 1
                script_lines.append(f"\nEQUILIBRIUM_PHASES {phase_id}")
                for solid in equilibrium_solid:
                    vol = solid_volume.get(solid, [0]*num_components)[i] if is_vector else solid_volume.get(solid, 0)
                    flag = " dissolve_only" if solid in dissolve_only else ""
                    script_lines.append(f"  {solid} 0 {vol}{flag}")
        
        if kinetic_solid:
                #time = self.params['tini']
                time = self.problem.Temporal(self.params).get('dt')
                script_lines.append(f"\nKINETICS 1")
                for solid in kinetic_solid:
                    kinmech = self.problem.kinetics_mechanism()[solid]
                    m_init = kinmech.get("m_init", 0.0)
            
                    script_lines.append(f"  {solid}")
                    script_lines.append(f"  -m {m_init}")          
                    script_lines.append(f"  -m0 {m_init}")        
                    script_lines.append(f"  -parms {time}")      
            
                script_lines.append(f"\nRATES")
                for solid in kinetic_solid:
                    script_lines.append(f"  {solid}")
                    kin_lines = self.problem.kinetics_mechanism()[solid]["rate_lines"]
                    script_lines.append("\n".join(kin_lines))


        if exchange:
              script_lines.append("\nEXCHANGE 1")
              script_lines.append("  equilibrate 10")        
              script_lines.append(f"  X {exchange[0]}")      
        return script_lines

    def knobs(self):     
        knobs = self.problem.knobs()
        return knobs.strip().split("\n") if knobs else []
                
    def generate_script(self):
        script_lines = []
        script_lines += self.new_phases_definition()
        script_lines += self.ini_solid_solution()       
        script_lines += self.boundary_conditions()
        ini = self.ini_conditions()
        if isinstance(ini[0], list):
           for sub in ini:  
               script_lines += sub
               script_lines.append("") 

        else:
           script_lines += ini
        script_lines += self.mechanisms()   
        script_lines += self.knobs()   
        scriptjoined = "\n".join(script_lines)        
        return scriptjoined
    
class PhreeqcCalculator(object):

    def __init__(self, nvertices,verticesind, initial_conditions,problem,params):
        self.p = params
        self.problem = problem
        self.nvertices = nvertices
        self.verticesind = verticesind
        self.initial_conditions = initial_conditions
        self.inflow_conc = {}
        self.init_conc = {}
        self.conc = {}
        self.usol = {}
        self.phreeqc = phreeqc_mod.IPhreeqc()
        self.phreeqc.load_database(self.p['DBpath'])
        self.components = []
        self.component_names = []
        self.species_keys = problem.chem_species()
        self._make_initial_state()
        self.solid_heading = []
        self.solscript = ScriptGenerator(self.problem,self.p) 

    def _make_initial_state(self): 
        self.components = self.species_keys
        self.equilibrium_solid = self.problem.mechanism()['equilibrium_solid']
        self.kinetic_solid = self.problem.mechanism()['kinetic_solid']
        self.exchange = self.problem.mechanism()['exchange']
        self.extra = self.problem.add_output_species()
        self.por = self.problem.Initial_condition(0)[1]['porosity']

        boundary_values, _ = self.problem.BoundaryValues_species(0, 0)

        uDval = self.verticesind[1]

        self.phreeqc.run_string(self.initial_conditions)

        code = ''
        for element in uDval:  code += f"COPY solution 0 {element}\n"
        for element in uDval:  code += f"COPY solid_solutions 0 {element}\n"
        if self.equilibrium_solid: 
          for element in uDval:  code += f"COPY equilibrium_phases 1 {element}\n"
        code += "END\n"
        code += "RUN_CELLS; -cells " + " ".join(str(element) for element in uDval) + "\n"   
        code += self.make_selected_output(self.components,self.equilibrium_solid,self.kinetic_solid,self.extra)
        self.phreeqc.run_string(code)
        inflow0 = self.get_selected_output()
        self.inflow_conc = inflow0
 
        self.phreeqc.run_string(self.initial_conditions)
        
        mesh = self.p['mesh']
        coords = mesh.coordinates()    
       
        ini = self.verticesind[0]
        
        if hasattr(self.problem, 'subdivide_domains'):
            subdivided = self.problem.subdivide_domains(mesh, ini)

            if subdivided == 0 or subdivided is None:
                iniv = [ini]
                labels = ['Domain']
            else:
                iniv = list(subdivided.values())
                labels = list(subdivided.keys())
        else:
            iniv = [ini]
            labels = ['Domain']
        
        if self.p['view_png']: Postprocess(self.p, self.problem).plot_vertex_groups(mesh, iniv, uDval, labels)

        code = ''
        
        if isinstance(iniv[0], int):
            iniv = [iniv]
        
        for idx, group in reversed(list(enumerate(iniv))):
            sol_id = 10 + idx
            for element in group:
                code += f"COPY solution {sol_id} {element}\n"
        
        if self.equilibrium_solid:
            for idx, group in reversed(list(enumerate(iniv))):
                phase_id = idx + 1  # Phases follow group index (but reversed order)
                for element in group:
                    code += f"COPY equilibrium_phases {phase_id} {element}\n"
        
        if self.kinetic_solid:
            for element in ini:
                code += f"COPY kinetics 1 {element}\n"
        
        if self.exchange:
            for element in ini:
                code += f"COPY exchange 1 {element}\n"

        code += "END\n"        
        code += "RUN_CELLS; -cells " + " ".join(str(element) for element in ini) + "\n"       
        code += self.make_selected_output(self.components,self.equilibrium_solid,self.kinetic_solid,self.extra)
        self.phreeqc.run_string(code)
        self.conc = self.get_selected_output()
        all_names = self.conc.keys()
        self.component_names = [name for name in all_names if name not in
                                ('cb', 'H', 'O')]
        for k in ("SR_Calcite", "SI_Calcite", "driving_Calcite"):
            if k in self.conc:
                print(f"{k} (first 10):", self.conc[k][:10])
        
        solidcomponents = self.equilibrium_solid+self.kinetic_solid
        self.permeability = self.conc['permeability']
        self.porosity_ini = self.conc['porosity']

        if self.problem.Initial_condition(0)[1]['compute_por']:
            self.porosity_ini = self.conc['porosity']
            self.inflow_conc['porosity'] = [self.por + v for v in self.inflow_conc['delta-porosity']]
            self.conc['porosity']       = [self.por + v for v in self.conc['delta-porosity']]
        
    def modify_dictionaries_ini(self,val,val_mod):        
        val_copy = copy.deepcopy(val)
        for key, constant in val_mod.items():
            if key in val_copy:
                val_copy[key] = [constant]* len(val_copy[key])
        return val_copy 

    def temporal_boundary_extraction(self,inletvec) : 
       inflow_or = {}
       for outer_key, inner_dict in inletvec.items():
         for inner_key, value in inner_dict.items():
           inflow_or[outer_key] = value
       return inflow_or 
   
    def split_conc_by_state(self,conc):
        i_soln_dict = {key: [] for key in conc}
        react_dict = {key: [] for key in conc}
    
        for i, state in enumerate(conc['state']):
            if state == 'i_soln':
                for key in conc:
                    i_soln_dict[key].append(conc[key][i])
            elif state == 'react':
                for key in conc:
                    react_dict[key].append(conc[key][i])
    
        return i_soln_dict, react_dict

    def modify(self, new_conc, time_step, v_index):
        import copy
    
        modify = []
        ini = v_index[0]  # interior vertices on CURRENT mesh
    

        def _clip_nonnegative_aqueous(d):
            out = {}
            for k, vals in d.items():
                if k in self.species_keys and isinstance(vals, (list, tuple)):
                    tmp = []
                    for x in vals:
                        if x is None:
                            tmp.append(0.0)
                        else:
                            try:
                                tmp.append(max(0.0, x))
                            except Exception:
                                tmp.append(x)
                    out[k] = tmp
                else:
                    out[k] = vals
            return out
    
        def _safe_at(dic, key, idx, default=None):
            try:
                if dic is not None and key in dic and dic[key] is not None:
                    return dic[key][idx]
            except Exception:
                pass
            return default
    
        def _safe_float(x, default=0.0):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default
    
        def _state_value(name, idx, default=0.0):
            """
            Priority for interior chemistry update:
              1) merged conc (self.conc + new_conc)
              2) self.conc
              3) default
            """
            v = _safe_at(conc, name, idx, None)
            if v is not None:
                return v
            v = _safe_at(self.conc, name, idx, None)
            if v is not None:
                return v
            return default

        new_conc = _clip_nonnegative_aqueous(new_conc)
    
        conc = copy.deepcopy(self.conc)
        conc.update(new_conc)
    
        val2 = {key: conc[key] for key in self.species_keys if key in conc}
    
        mechanism = self.problem.mechanism()
        equilibrium_solid = mechanism.get('equilibrium_solid', []) or []
        kinetic_solid = mechanism.get('kinetic_solid', []) or []
    
        if self.species_keys:
            rep_key = self.species_keys[0]
            n_ini = len(ini)
            n_vals = len(conc.get(rep_key, []))
            if n_vals != n_ini:
                print(
                    f"[PhreeqcCalculator.modify] WARNING: len(ini)={n_ini}, "
                    f"len(rep='{rep_key}')={n_vals}. "
                    f"This can happen if chemistry/transport mesh mapping changed."
                )

        if self.p['Solution_modify']:
            for index, cell in enumerate(ini):
                modify.append("SOLUTION_MODIFY %d" % cell)
    
                # cb / H / O (if available)
                cb_val = _state_value('cb', index, None)
                h_val  = _state_value('H', index, None)
                o_val  = _state_value('O', index, None)
    
                if cb_val is not None:
                    modify.append("\t-cb      %e" % _safe_float(cb_val, 0.0))
                if h_val is not None:
                    modify.append("\t-total_h %e" % _safe_float(h_val, 0.0))

                if o_val is not None:
                    modify.append("\t-total_o %e" % _safe_float(o_val, 0.0))
    
                modify.append("\t-totals")
                for name in self.species_keys:
                    v = _state_value(name, index, None)
                    if v is not None:
                        modify.append("\t\t%s\t%e" % (name, _safe_float(v, 0.0)))
    
                # Solids update (optional)
                update_solid_amount = False
                try:
                    update_solid_amount = bool(self.problem.pressure().get('update_solid_amount', False))
                except Exception:
                    update_solid_amount = False
    
                if update_solid_amount:

                    if equilibrium_solid:
                        modify.append("EQUILIBRIUM_PHASES_MODIFY %d" % cell)
                        for solid in equilibrium_solid:
                            m_eq = _safe_float(_state_value(solid, index, 0.0), 0.0)
                            modify.append("\t-component %s" % solid)
                            modify.append("\t  -si 0")
                            modify.append("\t  -m %e" % m_eq)

                    if kinetic_solid:
                        modify.append("KINETICS_MODIFY %d" % cell)
    
                        try:
                            time = _safe_float(self.problem.Temporal(self.p).get('dt', time_step), 0.0)
                        except Exception:
                            time = _safe_float(time_step, 0.0)
                            
                        for solid in kinetic_solid:  
                                time = self.problem.Temporal(self.p)['dt']
                                for solid in kinetic_solid:
                                      m_in = _state_value(solid, index, 0.0)
                                      m0 = self.problem.kinetics_mechanism()[solid].get("m_init", m_in)

                                      modify.append("\t-component %s" %solid)
                                      modify.append("\t  -m %e" % (m_in))
                                      modify.append("\t  -d_params")
                                      modify.append("\t\t%g" % time)    

    
        else:
            for index, cell in enumerate(ini):
                content = "\n    ".join(
                    [
                        f"{key} {val2[key][index]}"
                        for key in self.species_keys
                        if key in val2 and index < len(val2[key])
                    ]
                )
                modify.append(self.solscript.generate_solution_script(cell, content))

        if ini:
            run_cells = [str(cell) for cell in ini]
            modify.append("RUN_CELLS; -cells " + " ".join(run_cells) + "\n")
    
        code = '\n'.join(modify)
        
        self.phreeqc.run_string(code)
        self.conc = self.get_selected_output()

        if self.p['Solution_modify'] == 0:
            i_soln_dict, react_dict = self.split_conc_by_state(self.conc)
            listupdate = list(self.solid_properties(self.p)[7])  # copy
            if self.problem.Initial_condition(0)[1]['compute_por']:
                listupdate.append('delta-porosity')
    
            sub_react_dict = {name: react_dict[name] for name in listupdate if name in react_dict}
            self.conc = i_soln_dict.copy()
            self.conc.update(sub_react_dict)
    
        if self.problem.Initial_condition(0)[1]['compute_por']:
            if ('porosity' in self.conc) and ('delta-porosity' in self.conc):
                porosity_ini = copy.deepcopy(self.conc['porosity'])
                for index, cell in enumerate(ini):
                    if index < len(self.conc['porosity']):
                        dp = 0.0
                        if index < len(self.conc['delta-porosity']):
                            try:
                                dp = 0.0 if self.conc['delta-porosity'][index] is None else self.conc['delta-porosity'][index]
                            except Exception:
                                dp = 0.0
                        self.conc['porosity'][index] = porosity_ini[index] + dp

    def modify_inflow(self, new_conc, time_step, v_index):
        import copy
    
        modify = []
        uDval = v_index[1]

        def _clip_nonnegative_dict(d):
            out = {}
            for k, vals in d.items():
                if isinstance(vals, (list, tuple)):
                    tmp = []
                    for x in vals:
                        if x is None:
                            tmp.append(0.0)
                        else:
                            try:
                                tmp.append(max(0.0, x))
                            except Exception:
                                tmp.append(x)
                    out[k] = tmp
                else:
                    out[k] = vals
            return out
    
        def _safe_at(dic, key, idx, default=0.0):
            """Return dic[key][idx] safely, else default."""
            try:
                if key in dic and dic[key] is not None:
                    return dic[key][idx]
            except Exception:
                pass
            return default
    
        def _state_value(name, idx, default=0.0):
            """
            Priority:
              1) merged inflowconc (current inflow + new transported updates)
              2) self.inflow_conc
              3) self.conc
              4) default
            """
            v = _safe_at(inflowconc, name, idx, None)
            if v is not None:
                return v
            v = _safe_at(self.inflow_conc, name, idx, None)
            if v is not None:
                return v
            v = _safe_at(self.conc, name, idx, None)
            if v is not None:
                return v
            return default
    
        def _safe_float(x, default=0.0):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default
    

        new_conc = _clip_nonnegative_dict(new_conc)
    
        inflowconc = copy.deepcopy(self.inflow_conc)
        inflowconc.update(new_conc)
    
        val2 = {key: inflowconc[key] for key in self.species_keys if key in inflowconc}
    
        mechanism = self.problem.mechanism()
        equilibrium_solid = mechanism.get('equilibrium_solid', []) or []
        kinetic_solid = mechanism.get('kinetic_solid', []) or []
    
        update_solid_amount = False
        try:
            update_solid_amount = bool(self.problem.pressure().get('update_solid_amount', False))
        except Exception:
            update_solid_amount = False

        if self.p['Solution_modify']:
            for index, cell in enumerate(uDval):  # check index / cell mapping if needed
                modify.append("SOLUTION_MODIFY %d " % cell)
                modify.append("\t-cb      %e" % _safe_float(_state_value('cb', index, 0.0), 0.0))
                modify.append("\t-total_h %s" % _state_value('H', index, 0.0))
                modify.append("\t-total_o %s" % _state_value('O', index, 0.0))
                modify.append("\t-totals")
    
                for name in self.species_keys:
                    modify.append("\t\t%s\t%s" % (name, _state_value(name, index, 0.0)))
    
                if update_solid_amount:

                    if equilibrium_solid:
                        modify.append("EQUILIBRIUM_PHASES_MODIFY %d" % cell)
                        for solid in equilibrium_solid:
                            m_eq = _safe_float(_state_value(solid, index, 0.0), 0.0)
                            modify.append("\t-component %s" % solid)
                            modify.append("\t  -si 0")
                            modify.append("\t  -m %e" % m_eq)
    

                    if kinetic_solid:
                        modify.append("KINETICS_MODIFY %d" % cell)
                        time = self.problem.Temporal(self.p)['dt']

    
                        for solid in kinetic_solid:
                            m_in = _safe_float(_state_value(solid, index, 0.0), 0.0)

                            sr_val = None
                            sr_candidates = [f"SR_{solid}", "SR"]
                            for sk in sr_candidates:
                                v = _safe_at(inflowconc, sk, index, None)
                                if v is None:
                                    v = _safe_at(self.inflow_conc, sk, index, None)
                                if v is None:
                                    v = _safe_at(self.conc, sk, index, None)
                                if v is not None:
                                    sr_val = v
                                    break
                            if sr_val is None:
                                sr_val = 0.0
                            sr_val = _safe_float(sr_val, 0.0)
    
                            modify.append("\t-component %s" % solid)
                            modify.append("\t  -m %e" % m_in)
                            modify.append("\t  -d_params %f\t%e" % (time, sr_val))

    
        else:
            for index, cell in enumerate(uDval):
                content = "\n    ".join(
                    [f"{key} {val2[key][index]}" for key in self.species_keys if key in val2]
                )
                modify.append(self.solscript.generate_solution_script(cell, content))
    
        if uDval:
            modify.append("RUN_CELLS; -cells " + " ".join(str(element) for element in uDval) + "\n")
    
        code = '\n'.join(modify)
        self.phreeqc.run_string(code)
        self.inflow_conc = self.get_selected_output()

        if self.p['Solution_modify'] == 0:
            i_soln_dict, react_dict = self.split_conc_by_state(self.conc)
            listupdate = self.solid_properties(self.p)[7]  # solid headings
            listupdate = list(listupdate)
            if self.problem.Initial_condition(0)[1]['compute_por']:
                listupdate.append('delta-porosity')
    
            i_soln_dict_inf, react_dict_inf = self.split_conc_by_state(self.inflow_conc)
            sub_react_dict = {name: react_dict_inf[name] for name in listupdate if name in react_dict_inf}
            self.inflow_conc = i_soln_dict_inf.copy()
            self.inflow_conc.update(sub_react_dict)
    
        if self.problem.Initial_condition(0)[1]['compute_por']:
            if ('porosity' in self.inflow_conc) and ('delta-porosity' in self.inflow_conc):
                porosity_ini = copy.deepcopy(self.inflow_conc['porosity'])
                for index, cell in enumerate(uDval):
                    try:
                        dp = self.inflow_conc['delta-porosity'][index]
                        if dp is None:
                            dp = 0.0
                    except Exception:
                        dp = 0.0
                    self.inflow_conc['porosity'][index] = porosity_ini[index] + dp

    def get_modified(self):
        return self.conc
    def get_modified_inflow(self):
        return self.inflow_conc
    
    def solid_properties(self, p): 
        solid_equilibrium_moles, solid_solution_moles, _, _ = self.problem.solid_precipitation(p)
        molar_volumes = self.problem.molar_vol()
        solid_components_dict = self.problem.generate_solid_components_dict()   
        mechanism_data = self.problem.mechanism()    
        solid_solution_components = [comp for components in mechanism_data['solid_solution'].values() for comp in components] if mechanism_data['solid_solution'] else []
        solution_phases = list(mechanism_data['solid_solution'].keys()) if mechanism_data['solid_solution'] else []
        equisolid = mechanism_data['equilibrium_solid'] 
        kinsolid = mechanism_data['kinetic_solid'] 
        solid_solution_comp_moles = {comp: value   for components in solid_solution_moles.values() for comp, value in components.items()}
        solid_component_heading=['s_' + comp for comp in solid_solution_components]
        solid_phase_heading=['s_' + comp for comp in solution_phases]
        solid_heading = equisolid +  kinsolid + solid_component_heading + solid_phase_heading
        return solid_equilibrium_moles, solid_solution_comp_moles, molar_volumes, solid_solution_components, \
               solution_phases, equisolid, solid_components_dict,solid_heading
        
    def compute_Vb(self): #compute bulk volume/kgw per cell            
        solid_mols = {**self.solid_properties(self.p)[0], **self.solid_properties(self.p)[1]}
        molar_volumes = self.solid_properties(self.p)[2]
        _, properties = self.problem.Initial_condition(self.problem)

        return sum(molar_volumes[component] * solid_mols[component] for component in solid_mols) / (1 - properties['porosity'] - self.problem.solid_precipitation(self.p)[3]['Inert_volume_frac'])
    

    def make_selected_output(self, components, equisolid, kinsolid, extra):

        properties_key = self.problem.properties_key()
    
        (equilibrium_solid_moles,
         solid_solution_moles,
         molar_vol,
         solid_solution_components,
         solution_phases,
         equisolid,
         solid_components_dict,
         solid_heading) = self.solid_properties(self.p)
    
        solid_solution = self.problem.mechanism()['solid_solution']
    
        por0 = self.problem.Initial_condition(0)[1].get('porosity', 0.0)
        perm0 = self.problem.Initial_condition(0)[1].get('permeability', 0.0)
        if perm0 is None:
            perm0 = 0.0

        cols = []     
        punch = []        
        lino = 20        
    
        def add_col(name, expr):
            """Add one output column with its matching PUNCH expression."""
            nonlocal lino
            cols.append(name)
            punch.append(f"{lino} PUNCH {expr}")
            lino += 10
    
        def add_stmt(stmt):
            """Add a BASIC statement line (no heading)."""
            nonlocal lino
            punch.append(f"{lino} {stmt}")
            lino += 10
    

        add_col("cb", "CHARGE_BALANCE")
        add_col("H", 'TOTMOLE("H")')
        add_col("O", 'TOTMOLE("O")')
    
        for c in components:
            add_col(c, f'TOT("{c}")')
    
        for c in extra:
            add_col(c, f'MOL("{c}")')
    
        for s in equisolid:
            add_col(s, f'EQUI("{s}")')

        for s in (kinsolid or []):
            add_col(s, f'KIN("{s}")')                
    
            # SR_<s>
            add_stmt(f'sr_{s} = SR("{s}")')
            cols.append(f"SR_{s}")
            punch.append(f"{lino} PUNCH sr_{s}")
            lino += 10
            
            add_stmt(f'sr_{s} = SI("{s}")')
            cols.append(f"SI_{s}")
            punch.append(f"{lino} PUNCH si_{s}")
            lino += 10

        for c in solid_solution_components:
            add_col(c, f'S_S("{c}")')
    
        if solid_solution:
            for phase, comps in solid_solution.items():
                expr = " + ".join([f'S_S("{cc}")' for cc in comps])
                add_col(phase, expr)

        add_stmt("delta_porosity_val = 0")
        
        cols.append("delta-porosity")
        punch.append(f"{lino} PUNCH delta_porosity_val")
        lino += 10
        
        cols.append("porosity")
        punch.append(f"{lino} PUNCH {por0:.15g}")
        lino += 10
        
        cols.append("permeability")
        punch.append(f"{lino} PUNCH {perm0:.15g}")
        lino += 10
            
    
        if properties_key.get("pH", False):
            add_col("pH", '-LA("H+")')
    
        headings = "-headings    " + "\t".join(cols)
    
        out = self.problem.selected_output()
        out += "USER_PUNCH \n"
        out += headings + "\n"
        out += "\n".join(punch) + "\n"
    
        return out


    def get_selected_output(self):
        """Return calculation result as dict.

        Header entries are the keys and the columns
        are the values as lists of numbers.
        """
        output = self.phreeqc.get_selected_output_array()
        header = output[0]
        conc = {}
        for head in header:
            conc[head] = []
        for row in output[1:]:
            for col, head in enumerate(header):
                conc[head].append(row[col])
        return conc