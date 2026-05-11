"""
Script Name: Phreeqc + Fenics Reactive Transport V1 
Author: Juan Giraldo
Email: jfgiraldoa@gmail.com
"""
import numpy as np
from dolfin import *


class Test_Cases(object):

    def __init__(self, wa):
        self.wa = wa
        self.case = self.wa.get("case", "Engeesgard")
        self.asfem = bool(self.wa.get("ASFEM", False))
        self.dim = self.wa.get("dim", 1)
        self.res = 1
        self.ny = 1
        self.snaplist = [100, 200, 400, 600]

        self._set_case_parameters()

        if hasattr(self, "vel_x") and hasattr(self, "vel_y"):
            self.vel = (self.vel_x**2 + self.vel_y**2)**0.5

        dffalpha = self.diff + self.vel * self.alpha_disp
        self.Pe = self.vel * self.lsupx / (self.nx_trans + 1) / dffalpha
        self.Pe_global = self.vel * self.lsupx / dffalpha

        self.D_BE = self.vel**2 * self.dt / 2
        self.Pe_eff = (self.vel * self.lsupx / (self.nx_trans + 1)) / (dffalpha + self.D_BE)
        self.Cou = self.vel * self.dt / (self.lsupx / (self.nx_trans + 1))
        self.relPE = self.Pe_eff / self.Pe

        print(f"Case      = {self.case}")
        print(f"ASFEM     = {self.asfem}")
        print(f"Pe_global = {self.Pe_global:.6g}")
        print(f"Pe        = {self.Pe:.6g}")
        print(f"D_BE      = {self.D_BE:.6g}")
        print(f"Pe_eff    = {self.Pe_eff:.6g}")
        print(f"Cou       = {self.Cou:.6g}")
        print(f"relPE     = {self.relPE:.6g}")
        print("")

    def is_engeesgard_case(self):
        return self.case == "Engeesgard"

    def is_2d_case(self):
        return self.case == "Case_2D"

    def is_exchange_case(self):
        return self.case in ["Pe100", "Pe1000", "Pe10000"]


    def _set_case_parameters(self):
        engeesgard = {
            False: {  # FEM
                "lsupx": 0.5,
                "lsupy": 0.5,
                "nx": 100 * self.res,
                "nx_trans": 100 * self.res,
                "ny": 1,
                "alpha_disp": 0.0,
                "dt": 533,
                "nshifs": 39 * self.res,
                "diff": 6.28125e-8,
                "max_ref": 1,
                "snaplist": self.snaplist,
            },
            True: {   # ASFEM
                "lsupx": 0.5,
                "lsupy": 0.5,
                "nx": 100 * self.res,
                "nx_trans": 100 * self.res,
                "ny": 1,
                "alpha_disp": 0.0,
                "dt": 533,
                "nshifs": 39 * self.res,
                "diff": 6.28125e-8,
                "max_ref": 1,
                "snaplist": self.snaplist,
            },
        }

        exchange = {
            False: {  # FEM
                "lsupx": 0.08,
                "lsupy": 0.08,
                "nx": 750 * self.res,
                "nx_trans": 200 * self.res,
                "ny": 1,
                "alpha_disp": 0.0,
                "vel": 1.25e-4,
                "dt": 5.43216e-1,
                "nshifs": 900,
                "max_ref": 1,
                "snaplist": self.snaplist,
            },
            True: {   # ASFEM
                "lsupx": 0.08,
                "lsupy": 0.08,
                "nx": 750 * self.res,
                "nx_trans": 4 * self.res,
                "ny": 1,
                "alpha_disp": 0.0,
                "vel": 1.25e-4,
                "dt": 5.43216e-1,
                "nshifs": 900,
                "max_ref": 30,
                "snaplist": self.snaplist,
            },
        }

        case_2d = {
            False: {  # FEM
                "lsupx": 0.5,
                "lsupy": 0.5,
                "nx": 50 * self.res,
                "nx_trans": 50 * self.res,
                "ny": 50 * self.res,
                "ny_trans": 50 * self.res,
                "alpha_disp": 0.0,
                "diff": 1e-10,
                "vel_x": 1e-6,
                "vel_y": -1e-6,
                "dt": 538.46,
                "nshifs": 100,
                "max_ref": 1,
                "snaplist": self.snaplist,
            },
            True: {   # ASFEM
                "lsupx": 0.5,
                "lsupy": 0.5,
                "nx": 100 * self.res,
                "nx_trans": 5 * self.res,
                "ny": 100 * self.res,
                "ny_trans": 5 * self.res,
                "alpha_disp": 0.0,
                "diff": 7.07107e-10,
                "vel_x": 1.0e-5,
                "vel_y": -1.0e-5,
                "dt": 538.46,
                "nshifs": 60,
                "max_ref": 23,
                "snaplist": self.snaplist,
            },
        }

        configs = {
            False: {
                "Engeesgard": engeesgard[False],
                "Case_2D": case_2d[False],
                "Pe100": {**exchange[False], "diff": 1e-7},
                "Pe1000": {**exchange[False], "diff": 1e-8},
                "Pe10000": {**exchange[False], "diff": 1e-9},
            },
            True: {
                "Engeesgard": engeesgard[True],
                "Case_2D": case_2d[True],
                "Pe100": {**exchange[True], "diff": 1e-7},
                "Pe1000": {**exchange[True], "diff": 1e-8},
                "Pe10000": {**exchange[True], "diff": 1e-9},
            },
        }

        try:
            cfg = configs[self.asfem][self.case]
        except KeyError:
            valid = list(configs[self.asfem].keys())
            raise ValueError(
                f"Invalid case '{self.case}' for ASFEM={self.asfem}. Valid options: {valid}"
            )

        for key, value in cfg.items():
            setattr(self, key, value)

        self.ny = getattr(self, "ny", 1)
        self.ny_trans = getattr(self, "ny_trans", 1)

        # Global refinement controls requested by case dimension
        self.h_transport_min_factor = 0.24
        self.chem_refine_tol = 0.5 if self.case == "Case_2D" else 0.1

        if self.is_engeesgard_case():
            self.vel = self.lsupx / self.nx / self.dt

    def chem_species(self):
        if self.is_engeesgard_case() or self.is_2d_case():
            return ['Ca', 'C', 'Cl', 'Mg']
        elif self.is_exchange_case():
            return ['Ca', 'Cl', 'Na', 'K', 'N']
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def mechanism(self):
        if self.is_engeesgard_case():
            return {
                'equilibrium_solid': ['Calcite'],
                'kinetic_solid': ['Dolomite'],
                'solid_solution': [],
                'exchange': []
            }
        elif self.is_2d_case():
            return {
                'equilibrium_solid': ['Calcite', 'Dolomite'],
                'kinetic_solid': [],
                'solid_solution': [],
                'exchange': []
            }
        elif self.is_exchange_case():
            return {
                'equilibrium_solid': [],
                'kinetic_solid': [],
                'solid_solution': [],
                'exchange': [0.0011]
            }
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def add_output_species(self):
        if self.is_engeesgard_case() or self.is_2d_case():
            return ['CO3-2']
        elif self.is_exchange_case():
            return []
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def extra_species(self):
        return {}

    def Domain(self, p):
        p['Linfx'] = 0
        p['Lsupx'] = self.lsupx
        if p['dim'] == 2:
            p['Linfy'] = 0
            p['Lsupy'] = self.lsupy
        return p

    def Meshinfo(self, p):
        if p['dim'] == 1:
            p['Nx'] = self.nx
            p['Nx_transp'] = self.nx_trans
            p['Ny'] = 1
            p['Ny_transp'] = self.nx
        elif p['dim'] == 2:
            p['Nx'] = self.nx
            p['Ny'] = self.ny
            p['Nx_transp'] = self.nx_trans
            p['Ny_transp'] = self.ny_trans
        else:
            raise ValueError("p['dim'] must be 1 or 2")

        p['Nx_chem'] = self.nx
        p['mesh_type'] = 'unstructured'
        p['Constant_tag_DBC'] = False
        return p

    def Temporal(self, p):
        p['Unsteady'] = True
        p['TIM'] = 'BE'
        p['tini'] = 0
        p['time_steps'] = self.nshifs
        p['dt'] = self.dt
        p['T'] = p['dt'] * p['time_steps']
        return p

    def diffusionCoeff(self):
        if self.dim == 1:
            diffcoeff = Constant(self.diff + self.vel * self.alpha_disp)
        else:
            diffcoeff = Constant(self.diff)
        return {sp: diffcoeff for sp in self.chem_species()}

    def kinetics_mechanism(self):
        if self.is_engeesgard_case():
            return {
                "Dolomite": {
                    "m_init": 0,
                    "rate_lines": [
                        "-start",
                        "14 logK25 = -3.19",
                        "15 ny = 0.5",
                        "16 mech_a = (10^logK25) * ACT(\"H+\")^ny",
                        "24 logK25 = -7.53",
                        "26 mech_b = (10^logK25)",
                        "30 rate = mech_a + mech_b",
                        "40 teta = 1",
                        "41 eta = 1",
                        "42 Area = .001",
                        "70 rate = Area * rate * (1 - SR(\"Dolomite\")^teta)^eta",
                        "80 deltamoles = rate * (PARM(1))",
                        "100 SAVE deltamoles",
                        "-end",
                    ],
                }
            }
        return {}

    def solid_precipitation(self, p):
        if self.is_engeesgard_case():
            cal_val = 1.220005e-4
            solid_equilibrium_moles = {
                'Calcite': cal_val,
                'Dolomite': cal_val * 0.0526
            }
            other_volumes_frac = {'Inert_volume_frac': 0.45}
            return [solid_equilibrium_moles, {}, {}, other_volumes_frac]

        elif self.is_2d_case():
            solid_equilibrium_moles = {'Calcite': 2.14e-5, 'Dolomite': 0}
            return [solid_equilibrium_moles, {}, {}, {}]

        elif self.is_exchange_case():
            return [{}, {}, {}, {}]

        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def generate_solid_components_dict(self):
        mechanism_data = self.mechanism()
        equilibrium_solid = mechanism_data['equilibrium_solid']
        kinetic_solid = mechanism_data['kinetic_solid']

        solid_solution_comp = (
            [comp for components in mechanism_data['solid_solution'].values() for comp in components]
            if mechanism_data['solid_solution'] else []
        )

        solid_components = equilibrium_solid + kinetic_solid + solid_solution_comp
        return {component: component for component in solid_components}

    def molar_vol(self):
        if self.is_engeesgard_case() or self.is_2d_case():
            return {
                'Calcite': 3.69e-5,
                'Dolomite': 6.49e-5
            }
        elif self.is_exchange_case():
            return {}
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def phases_definition(self):
        return []

    def knobs(self):
        return []

    def selected_output(self):
        return """
                   SELECTED_OUTPUT
                           -reset false
               """

    def AdvectiveCoeff(self, p):
        dim = p.get('dim', 1)

        if dim == 1:
            adv_x = self.vel
            try:
                if float(adv_x) == 0.0:
                    return None
            except Exception:
                pass

            ax = adv_x if hasattr(adv_x, "ufl_shape") else Constant(adv_x)
            return as_vector([ax])

        elif dim == 2:
            adv_x = getattr(self, "vel_x", self.vel)
            adv_y = getattr(self, "vel_y", 0.0)

            try:
                if float(adv_x) == 0.0 and float(adv_y) == 0.0:
                    return None
            except Exception:
                pass

            ax = adv_x if hasattr(adv_x, "ufl_shape") else Constant(adv_x)
            ay = adv_y if hasattr(adv_y, "ufl_shape") else Constant(adv_y)
            return as_vector([ax, ay])

        else:
            raise ValueError("p['dim'] must be 1 or 2")

    def Conditions(self):
        if self.is_engeesgard_case():
            return {
                'pH': 7,
                'pH_charge': True,
                'Temp': 25,
                'units': 'mol/kgw',
            }
        elif self.is_2d_case():
            return {
                'pH': 7,
                'pH_charge': True,
                'Temp': 25,
                'units': 'mol/kgw',
                'pe': 4
            }
        elif self.is_exchange_case():
            return {
                'pH': 7,
                'pH_charge': True,
                'Temp': 25,
                'units': 'mol/kgw',
                'pe': 12.5
            }
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def BoundaryValues_species(self, t, p):
        if self.is_engeesgard_case():
            uD = {
                'Ca': {'1': 1e-10},
                'C': {'1': 1e-10},
                'Cl': {'1': 0.002},
                'Mg': {'1': 0.001},
            }

        elif self.is_2d_case():
            uD = {
                'Ca': {'2': 1e-10, '1': 1e-10},
                'C': {'2': 1e-10, '1': 1e-10},
                'Cl': {'2': 0.002, '1': 0.002},
                'Mg': {'2': 0.001, '1': 0.001},
            }

        elif self.is_exchange_case():
            uD = {
                'Ca': {'1': 0.6e-3},
                'Cl': {'1': 1.2e-3},
                'Na': {'1': 0.0},
                'K': {'1': 0.0},
                'N': {'1': 0.0},
            }

        else:
            raise ValueError(f"Unknown case '{self.case}'")

        uN = Constant(0)
        return [uD, uN]

    def Initial_condition(self, p):
        if self.is_engeesgard_case():
            Ini_spe = {
                'Ca': 1.22e-4,
                'C': 1.22e-4,
                'Cl': 1e-12,
                'Mg': 1e-12
            }
            properties = {
                'pH': 7,
                'pH_charge': True,
                'porosity': 0.32,
                'permeability': 1.186e-11,
                'compute_por': False
            }

        elif self.is_2d_case():
            Ini_spe = {
                'Ca': 1.22e-4,
                'C': 1.22e-4,
                'Cl': 1e-12,
                'Mg': 1e-12
            }
            properties = {
                'pH': 7,
                'pH_charge': True,
                'porosity': 0.3,
                'permeability': 1e-12,
                'compute_por': False
            }

        elif self.is_exchange_case():
            Ini_spe = {
                'Ca': 0,
                'Cl': 0,
                'Na': 1.0e-3,
                'K': 0.2e-3,
                'N': 1.2e-3
            }
            properties = {
                'pH': 7,
                'pH_charge': True,
                'porosity': 0.3,
                'permeability': 1e-12,
                'compute_por': False
            }

        else:
            raise ValueError(f"Unknown case '{self.case}'")

        return Ini_spe, properties

    def pressure(self):
        return {
            'Compute_velocity': False,
            'update_solid_amount': True,
            'uD_pressure': {'1': 70e3, '3': 0},
            'source_term': Constant(1e-15),
        }

    def forcing_term(self, p):
        val = Constant(0)
        return {sp: val for sp in self.chem_species()}

    def ExactSoltution(self):
        return []

    def Colordic(self):
        if self.is_engeesgard_case() or self.is_2d_case():
            return {
                'Ca': 'orange',
                'C': 'black',
                'Cl': 'green',
                'Mg': 'blue',
                'CO3-2': 'red',
                'Calcite': 'purple',
                'Dolomite': 'magenta',
            }
        elif self.is_exchange_case():
            return {
                'Ca': 'red',
                'Cl': 'orange',
                'Na': 'blue',
                'K': 'green',
                'N': 'black'
            }
        else:
            raise ValueError(f"Unknown case '{self.case}'")

    def color_scale(self):
        return {'color-mod': False}

    def Components_plot(self):
        return [
            [
                self.chem_species() + self.add_output_species(),
                self.mechanism()['equilibrium_solid'],
                self.mechanism()['kinetic_solid']
            ],
            self.chem_species() + self.add_output_species() +
            self.mechanism()['equilibrium_solid'] +
            self.mechanism()['kinetic_solid']
        ]

    def get_scaling_factor(self, species_type):
        por = self.Initial_condition(0)[1].get('porosity', 0.32)
        rho_w = 1000.0
        Sw = 1.0

        unit_factor = (1.0 - por) / (rho_w * por * Sw)
        scale = 1.0 / unit_factor

        scaling_factors = {
            'Calcite': scale,
            'Dolomite': scale,
        }
        return scaling_factors.get(species_type, 1000.0)

    def get_axis_limit(self, axis_name):
        if self.is_engeesgard_case():
            axis_limits = {
                'lim-mod': True,
                'y_lim_species': (0, 2.1),
                'y_lim_solid_equilibrium': (0, 0.06),
                'y_lim_solid_kinetic': (0, 0.0015)
            }
        else:
            axis_limits = {
                'lim-mod': False,
                'y_lim_species': (0, 2.1),
                'y_lim_solid_equilibrium': (0, 0.06),
                'y_lim_solid_kinetic': (0, 0.0015)
            }

        return axis_limits.get(axis_name, (0, 1))

    def properties_key(self):
        compute_vel = False
        try:
            compute_vel = bool(self.pressure().get('Compute_velocity', False))
        except Exception:
            compute_vel = False

        return {
            'delta-porosity': compute_vel,
            'porosity': compute_vel,
            'permeability': compute_vel,
            'pH': False,
            'SR_Calcite': False,
            'SI_Dolomite': False,
        }
    def save_outputs(self, p, data, accum, accumadapta, dofstotal):
        import os
        from Postprocess import Postprocess
        from Exchange_outputs import save_exchange_outputs_h5, save_exchange_outputs_h5_adaptive
        if self.is_exchange_case() and p['dim'] == 1 and (p["ASFEM"] is False) and (p["DG"] is False) and p.get('accumulate_solution', False):
            save_exchange_outputs_h5(
                "../results/Exchange-outputs/Exchange_PoreVolume_phreeqnics-FEM100-Pe" + str(int(self.Pe_global)) + "-dofs200" + ".h5",
                "../results/Exchange-outputs/Exchange_Distance_phreeqnics-FEM100-Pe" + str(int(self.Pe_global)) + "-dofs200" + ".h5",
                accum,
                p,
                ts=int(self.nshifs) - 1,
                to_mmol=True
            )
    
        if self.is_exchange_case() and p['dim'] == 1 and (p["ASFEM"] is False) and (p["DG"] is True) and p.get('accumulate_solution', False):
            save_exchange_outputs_h5(
                "../results/Exchange-outputs/Exchange_PoreVolume_phreeqnics-DG100-Pe" + str(int(self.Pe_global)) + "-dofs200" + ".h5",
                "../results/Exchange-outputs/Exchange_Distance_phreeqnics-DG100-Pe" + str(int(self.Pe_global)) + "-dofs200" + ".h5",
                accum,
                p,
                ts=int(self.nshifs) - 1,
                to_mmol=True
            )
    
        if self.is_exchange_case() and p['dim'] == 1 and (p["ASFEM"] is True) and p.get('accumulate_solution_adapta', False):
            save_exchange_outputs_h5_adaptive(
                "../results/Exchange-outputs/Exchange_Timesolution_phreeqnics-ref-adapta2-Pe" + str(int(self.Pe_global)) + ".h5",
                "../results/Exchange-outputs/Exchange_Snaps_phreeqnics-ref-adapta2-Pe" + str(int(self.Pe_global)) + ".h5",
                accumadapta,
                p,
                ts=int(self.nshifs) - 1,
                to_mmol=True
            )
    
        if self.is_engeesgard_case():
            outdir = '../results/Engeesgard'
            os.makedirs(outdir, exist_ok=True)
    
            Postprocess(p, self).save_profiles_h5(
                os.path.join(outdir, 'Calcite-Dolomite-Phreeqnics-B1.h5'),
                data
            )
    
