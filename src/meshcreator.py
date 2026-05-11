
"""
Created on Wed Apr 23 18:37:46 2025

@author: Juan Giraldo
"""
from dolfin import IntervalMesh, Measure, RectangleMesh, BoxMesh, Point, Mesh, near, MeshFunction, SubDomain, XDMFFile
import meshio
import gmsh
from pathlib import Path
import numpy as np

class MeshGenerator:
    def __init__(self, params):
        self.p = params
    def Create_mesh(self, problem):
        problem.Domain(self.p)
        problem.Meshinfo(self.p)
        mesh = self._create_mesh_internal()
        self.p['mesh'] = mesh
        return mesh

    def Create_mesh_new(self, problem):
        mesh = self._create_mesh_internal()
        self.p['mesh'] = mesh
        return mesh
    
    def Create_mesh_new2(self, problem):
        mesh = self._create_mesh_internal()
        return mesh

    def _create_mesh_internal(self):
        if self.p['dim'] == 1 and self.p['meshdomain'] == 'Unitdomain':
            mesh = IntervalMesh(self.p['Nx'], self.p['Linfx'], self.p['Lsupx'])
        elif self.p['dim'] == 2 and self.p['meshdomain'] == 'Unitdomain':
            mesh = RectangleMesh(Point(self.p['Linfx'], self.p['Linfy']),
                                  Point(self.p['Lsupx'], self.p['Lsupy']),
                                  self.p['Nx'], self.p['Ny'])
        elif self.p['dim'] == 2:
          if self.p['meshdomain'] == 'Furrowdomain' or self.p['meshdomain'] == 'Leachingdomain':
            mesh = self.Create_geometry_with_mesh(self.p['Nx']) 
        else:
            raise Exception('TODO other dim')
        return mesh


    def Create_geometry_with_mesh(self, Nini):
        import gmsh
        import meshio
        from dolfin import Mesh, XDMFFile
    
        gmsh.initialize()
    
        if self.p['meshdomain'] == 'Furrowdomain':
            
            meshpath="../MeshesAux/Furrow_geometry"
            gmsh.model.add(meshpath)

            points = [(0, 0), (0, 0.7), (0.08, 0.7), (0.26, 0.85), (0.74, 0.85), (0.74, 0)]
            gmsh_points = [gmsh.model.occ.addPoint(x, y, 0) for x, y in points]
            lines = []
            for i in range(len(gmsh_points) - 1):
                lines.append(gmsh.model.occ.addLine(gmsh_points[i], gmsh_points[i+1]))
            lines.append(gmsh.model.occ.addLine(gmsh_points[-1], gmsh_points[0]))
            curve_loop = gmsh.model.occ.addCurveLoop(lines)
            gmsh.model.occ.addPlaneSurface([curve_loop])
    
    
        if self.p['meshdomain'] == 'Furrowdomain':
            points = [(0, 0),(0, 0.7),(0.08, 0.7),(0.26, 0.85),(0.74, 0.85),(0.74, 0)]
            #points = [[(0, 0),(0, 0.85), (0.15, 0.85), (0.30, 1.00),(0.70, 1.00), (0.85, 0.85), (1.00, 0.85), (1.00, 0)]
            gmsh_points = [gmsh.model.occ.addPoint(x, y, 0) for x, y in points] 
            lines = []
            for i in range(len(gmsh_points) - 1):
                lines.append(gmsh.model.occ.addLine(gmsh_points[i], gmsh_points[i+1]))
            lines.append(gmsh.model.occ.addLine(gmsh_points[-1], gmsh_points[0]))  # Close loop  
            curve_loop = gmsh.model.occ.addCurveLoop(lines)
            gmsh.model.occ.addPlaneSurface([curve_loop])
            gmsh.model.occ.synchronize()

    
        if self.p['meshdomain'] == 'Leachingdomain':
            
            meshpath="../MeshesAux/Tailling_geometry"
            gmsh.model.add(meshpath)
         
            points = [(0, 0), (0, 0.24), (0.50, 0.24), (0.54, 0.232), (0.80, 0.12),
                      (1.00, 0.059), (1.00, 0.04), (1.05, 0.04), (1.05, 0)]
            gmsh_points = [gmsh.model.occ.addPoint(x, y, 0) for x, y in points]
            lines = []
            for i in range(len(gmsh_points) - 1):
                lines.append(gmsh.model.occ.addLine(gmsh_points[i], gmsh_points[i+1]))
            lines.append(gmsh.model.occ.addLine(gmsh_points[-1], gmsh_points[0]))
            curve_loop = gmsh.model.occ.addCurveLoop(lines)
            gmsh.model.occ.addPlaneSurface([curve_loop])
    
            gmsh.model.occ.synchronize()
    
            gmsh.model.mesh.field.add("Box", 1)
            gmsh.model.mesh.field.setNumber(1, "VIn", Nini * 0.3)  # Fine mesh
            gmsh.model.mesh.field.setNumber(1, "VOut", Nini)       # Coarse mesh outside
            gmsh.model.mesh.field.setNumber(1, "XMin", 0.04)
            gmsh.model.mesh.field.setNumber(1, "XMax", 0.26)
            gmsh.model.mesh.field.setNumber(1, "YMin", 0.2)       # Refine top band
            gmsh.model.mesh.field.setNumber(1, "YMax", 0.25)
            gmsh.model.mesh.field.setNumber(1, "Thickness", 1e-2)  # Ignored in 2D but required
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
    
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", Nini)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(meshpath+'.msh')
        gmsh.finalize()
    
        mesh_from_gmsh = meshio.read(meshpath+'.msh')
        meshio.write(meshpath+'.xdmf', meshio.Mesh(
            points=mesh_from_gmsh.points[:, :2],
            cells={"triangle": mesh_from_gmsh.cells_dict["triangle"]},
        ))
    
        fenics_mesh = Mesh()
        with XDMFFile(meshpath+'.xdmf') as infile:
            infile.read(fenics_mesh)
    
        return fenics_mesh


    
class BoundaryCreator:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p

    def create_boundaries(self):
        mesh, p = self.mesh, self.p
    
        # Initialize facet markers
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
    
        # List of (boundary class, marker ID)
        boundary_classes = []
    
        if p.get('Constant_tag_BC', False):
            boundaries.set_all(1)
            marker_to_name = {1: 'Constant BC'}
        else:
            # CASE 1: Standard rectangle
            if p['meshdomain'] == 'Unitdomain':
                
                if p['boundarydomain'] == 'CalciteDolomite2D-adaptive':
                    class Left(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[0], p['Linfx']) and on_boundary and (0.4 <= x[1] <= 0.5)
                    class Right(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[0], p['Lsupx']) and on_boundary   
                    class Top(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[1], p['Lsupy']) and on_boundary and (x[0] <= 0.1)
                    class Bottom(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[1], p['Linfy']) and on_boundary                    
                    boundary_classes = [(Left, 1),(Top, 2),(Right, 3),(Bottom, 4)]
                    
                else:                    
                    class Left(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[0], p['Linfx']) and on_boundary #and (0.4 <= x[1] <= 0.5)
                    class Right(SubDomain):
                        def inside(self, x, on_boundary):
                            return near(x[0], p['Lsupx']) and on_boundary   
                    boundary_classes = [(Left, 1),(Right, 3)]
                    if p['dim']>1:
                        class Top(SubDomain):
                            def inside(self, x, on_boundary):
                                return near(x[1], p['Lsupy']) and on_boundary #and (x[0] <= 0.1)
                        class Bottom(SubDomain):
                            def inside(self, x, on_boundary):
                                return near(x[1], p['Linfy']) and on_boundary
                        boundary_classes += [(Top, 2),(Bottom, 4)]
    
            # CASE 2: Furrow domain
            elif p['meshdomain'] == 'Furrowdomain':
                class BottomBoundary(SubDomain):
                    def inside(self, x, on_boundary):
                        return near(x[0], 0)
    
                class LeftBoundary(SubDomain):
                    def inside(self, x, on_boundary):
                        return near(x[0], 0)
    
                class FurrowSurface(SubDomain):
                    def inside(self, x, on_boundary):
                        #return on_boundary and x[0] < 0.3 and 0.85 <= x[1] < 0.91
                        return on_boundary and x[0] < 0.26 and 0.7 <= x[1] < 0.76

    
                boundary_classes = [(BottomBoundary, 3), (LeftBoundary, 4), (FurrowSurface, 1)]
    
            # CASE 2: Leaching domain
            elif p['meshdomain'] == 'Leachingdomain':
                        
                tol               = 1e-6
                xmin, xmax        = 0.0, 1.05
                ymin, ymax        = 0.0, 0.24
                water_table_left  = 0.12
                water_table_right = 0.04
                x1,x2,x3 = 1.0, 0.25, 0.05
                y1 =0.059
                
                    
                class LeftBelow(SubDomain):
                    def inside(self, x, on_boundary):
                        return (on_boundary
                                and near(x[0], xmin, tol)
                                and x[1] <= water_table_left + tol)
                
                class LeftAbove(SubDomain):
                    def inside(self, x, on_boundary):
                        return (on_boundary
                                and near(x[0], xmin, tol)
                                and x[1] >= water_table_left - tol)
                
                class RightBelow(SubDomain):
                    def inside(self, x, on_boundary):
                        return (
                            on_boundary
                            and x[0] >  x1 - tol         # right edge
                            and x[1] > ymin - tol       # include y=0 corner
                        )
                               
                class BC3(SubDomain):  # originally     x >  0.25
                        def inside(self, x, on_boundary):
                            return (on_boundary 
                                and x[1] > y1 - tol
                                and x[0] > xmin -tol 
                                #and (x[0] >= x2+tol or x[0] < x3 + tol)
                                )    
                
                class BC2(SubDomain):  # originally 0.05 < x <= 0.25
                        def inside(self, x, on_boundary):
                            return (on_boundary 
                                    and near(x[1], ymax)
                                    and  x[0] >  x3 +tol
                                    and  x[0] <  x2 +tol)
                
                class BottomBoundary(SubDomain):
                        def inside(self, x, on_boundary):
                            return (on_boundary and near(x[1], ymin, tol)) 
    
                    
                boundary_classes = [(RightBelow, 4), (BC3, 6),(BC2, 5),(BottomBoundary, 1),(LeftAbove, 3),(LeftBelow, 2)]
            else:
                raise Exception("Unknown domain type")
    
            # Automatically mark and create the marker_to_name map
            marker_to_name = {}
            for boundary_class, marker in boundary_classes:
                boundary_class().mark(boundaries, marker)
                marker_to_name[marker] = boundary_class.__name__
    
        return boundaries, marker_to_name

