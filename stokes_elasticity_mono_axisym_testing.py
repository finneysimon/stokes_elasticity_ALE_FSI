"""
This is a script to test the monolithic solver. The implementation 
is steady, 2D axisymmetric and is formulated in a moving
frame that travels with the particle.

To test the solid solver we define d_mag such that the particle 
attains the radius alpha(1+d_mag). We fix epsilon small such that
the particle is effectively stiff to the fluid stress. We can then 
predict the particle velocity of the inflated radius to test the
fluid and ALE solvers. The code works by initially solving the problem
with a small d_mag (deformation magnitude) and gradually ramping up 
the value of d_mag. If convergence is not obtained then the code 
tries again using a smaller value of d_mag.

"""

from dolfin import *
from multiphenics import *
from helpers import *
import matplotlib.pyplot as plt
from csv import writer
import numpy as np


# writing to csvfile
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def write_list_as_row(file_name, list_of_elem):
    with open(file_name, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


# Python function for axisymmetric divergence
def diva(vec, r):
    return div(vec) + vec[1] / r


# ---------------------------------------------------------------------
# Setting up file names and paramerers
# ---------------------------------------------------------------------
# directory for file output
dir = '/home/simon/data/fenics/elastic_particle_ALE/'


def generate_output_files(rad):
    output_s = XDMFFile(dir + "validation_solid_0-"
                        + str(float('%.2g' % rad))[2:] + ".xdmf")
    output_s.parameters['rewrite_function_mesh'] = False
    output_s.parameters["functions_share_mesh"] = True
    output_s.parameters["flush_output"] = True

    output_f = XDMFFile(dir + "validation_fluid_0-"
                        + str(float('%.2g' % rad))[2:] + ".xdmf")
    output_f.parameters['rewrite_function_mesh'] = False
    output_f.parameters["functions_share_mesh"] = True
    output_f.parameters["flush_output"] = True

    return output_s, output_f


def get_mesh(rad):
    meshname = 'channel_sphere_' + str(float('%.1g' % rad))[2:]
    mesh = Mesh('mesh/' + meshname + '.xml')
    subdomains = MeshFunction("size_t", mesh, 'mesh/' + meshname + '_physical_region.xml')
    bdry = MeshFunction("size_t", mesh, 'mesh/'
                        + meshname + '_facet_region.xml')
    return mesh, subdomains, bdry


"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["ghost_mode"] = "shared_facet"

# loop over radii
for ii in range(1):
    rad = (2*ii + 6) / 20
    print("doing rad = ", 2 * rad)

    output_data_file = dir + 'validation_data_0-' + str(2 * rad)[2:] + '.csv'
    print(output_data_file)
    write_list_as_row(output_data_file, ['d_mag', 'vel', 'disp'])

    output_s, output_f = generate_output_files(2 * rad)

    # mesh has been created with Gmsh
    mesh, subdomains, bdry = get_mesh(2 * rad)

    """
        Physical parameters
    """

    # stiff particle such that it is rigid to fluid stress
    eps_try = 1e-10
    eps = Constant(eps_try)

    # Lame's first parameter for compressible particle
    lambda_s = 1  # nu_s = lambda_s/(2(lambda_s + mu_s))

    # deformation magnitude for inflation
    d_mag_try = 0.1
    d_mag_max = 0.5
    d_mag_it = 0.1
    d_mag = Constant(d_mag_try)
    d_mag_it_min = 1e-3

    # define the boundaries (values from the gmsh file)
    circle = 1
    fluid_axis = 2
    inlet = 3
    outlet = 4
    wall = 5
    solid_axis = 6

    # define the domains
    fluid = 10
    solid = 11

    Of = generate_subdomain_restriction(mesh, subdomains, fluid)
    Os = generate_subdomain_restriction(mesh, subdomains, solid)
    Sig = generate_interface_restriction(mesh, subdomains, {fluid, solid})

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)
    dS = dS(circle)

    # normal and tangent vectors
    nn = FacetNormal(mesh);
    tt = as_vector((-nn[1], nn[0]))

    ez = as_vector([1, 0])
    ex = as_vector([0, 1])

    # define the surface differential on the circle
    x = SpatialCoordinate(mesh)
    r = x[1]

    # ---------------------------------------------------------------------
    # elements, function spaces, and test/trial functions
    # ---------------------------------------------------------------------
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    DGT = VectorElement("CG", mesh.ufl_cell(), 1)
    P0 = FiniteElement("R", mesh.ufl_cell(), 0)

    """
    Setting up the elements and solution: here is the notation:
    
    u_f: fluid velocity from Stokes equations
    p_f: fluid pressure from Stokes equations
    
    u_s: solid displacement from nonlinear elasticity
    p_s: solid pressure from nonlinear elasticity
    
    f_0: Lagrange multiplier corresponding to the force needed to pin the
    solid in place (should end up being zero since particle is in
    equilibrium with Poiuseuille flow and applied body force)
    
    U_0: the translational velocity of the solid
    
    lam: Lagrange multiplier corresponding to the fluid traction 
    acting on the solid
    
    lam_p: Lagrange multiplier to ensure the mean fluid pressure is zero
    
    u_a: fluid "displacement" from the ALE method (e.g. how to deform
    the fluid geometry)
    
    lam_a: Lagrange multiplier to ensure continuity of fluid and solid
    displacement (ensures compatibility between fluid/solid domains)
    
    """

    mixed_element = BlockElement(P2, P1, P2, P0, P0, DGT, P0, P2, DGT)
    V = BlockFunctionSpace(mesh, mixed_element, restrict=[Of, Of, Os, Os, Of, Sig, Of, Of, Sig])

    X = BlockFunction(V)
    (u_f, p_f, u_s, f_0, U_0, lam, lam_p, u_a, lam_a) = block_split(X)

    # unknowns and test functions
    Y = BlockTestFunction(V)
    (v_f, q_f, v_s, g_0, V_0, eta, eta_p, v_a, eta_a) = block_split(Y)

    Xt = BlockTrialFunction(V)

    # Placeholder for the last converged solution
    X_old = BlockFunction(V)

    # ---------------------------------------------------------------------
    # boundary conditions
    # ---------------------------------------------------------------------

    """
    Physical boundary conditions
    """

    # Far-field fluid velocity - Poiseuille flow
    far_field = Expression(('(1 - x[1] * x[1] / 0.25) * t', '0'), degree=0, t=1)

    # impose the far-field fluid velocity upstream and downstream
    bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
    bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)

    # impose no vertical fluid flow at the centreline axis
    bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)

    # impose no-slip and no-penetration at the channel wall
    bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), bdry, wall)

    # impose test displacement on solid axis
    solid_def = Expression(('d_mag * x[0]', '0.0'), degree=0, d_mag=d_mag)
    bc_solid_axis = DirichletBC(V.sub(2), solid_def, bdry, solid_axis)

    """
    Boundary conditions for the ALE problem for fluid 
    displacement.  These are no normal displacements
    """
    ac_inlet = DirichletBC(V.sub(7).sub(0), Constant((0)), bdry, inlet)
    ac_outlet = DirichletBC(V.sub(7).sub(0), Constant((0)), bdry, outlet)
    ac_fluid_axis = DirichletBC(V.sub(7).sub(1), Constant((0)), bdry, fluid_axis)
    ac_wall = DirichletBC(V.sub(7).sub(1), Constant((0)), bdry, wall)

    # Combine all BCs together
    bcs = BlockDirichletBC(
        [bc_inlet, bc_outlet, bc_fluid_axis, bc_wall, bc_solid_axis,
         ac_inlet, ac_outlet, ac_fluid_axis, ac_wall])

    # surface traction required to generate the test displacement
    solid_traction = Expression(('1/eps*(1+d_mag-(1/(1+d_mag))+pow(1+d_mag, 2)*(pow(1+d_mag, 3)-1)*lambda_s) * x[0] / rad',
                                 '1/eps*(1+d_mag-(1/(1+d_mag))+pow(1+d_mag, 2)*(pow(1+d_mag, 3)-1)*lambda_s) * x[1] / rad'),
                                degree=0, eps=eps, lambda_s=lambda_s, d_mag=d_mag, rad=rad)
    # ---------------------------------------------------------------------
    # Define the model
    # ---------------------------------------------------------------------

    I = Identity(2)

    """
    Solids problem
    """
    # deformation gradient tensor
    F = I + grad(u_s)
    H = inv(F.T)

    # (non-dim) compressible PK1 stress tensor
    def F_func(u_s):
        return I + grad(u_s)

    def H_func(u_s):
        return inv(F_func(u_s).T)

    def J_s_func(u_s):
        return det(F_func(u_s)) * (1 + u_s[1] / r)

    def Sigma_s_func(u_s, eps):
        return 1 / eps * (F_func(u_s) - H_func(u_s)) + lambda_s / eps * (J_s_func(u_s) - 1) * J_s_func(u_s) * H_func(u_s)

    J_s = det(F) * (1 + u_s[1] / r)
    Sigma_s = 1 / eps * (F - H) + lambda_s / eps * (J_s - 1) * J_s * H

    """
    Fluids problem: mapping the current configuration to the 
    initial configuration leads a different form of the
    incompressible Stokes equations
    """

    # Deformation gradient for the fluid
    F_a = I + grad(u_a)
    H_a = inv(F_a.T)

    # Jacobian for the fluid
    J_a = det(F_a) * (1 + u_a[1] / r)

    # PK1 stress tensor and incompressibility condition for the fluid
    Sigma_f = J_a * (-p_f * I + grad(u_f) * H_a.T + H_a * grad(u_f).T) * H_a
    ic_f = diva(J_a * inv(F_a) * u_f, r)

    """
    ALE problem: there are three different versions below
    """

    # Laplace
    # sigma_a = grad(u_a)

    # linear elasticity
    nu_a = Constant(0.1)
    E_a = 0.5 * (grad(u_a) + grad(u_a).T)
    sigma_a = nu_a / (1 + nu_a) / (1 - 2 * nu_a) * diva(u_a, r) * I + 1 / (1 + nu_a) * E_a

    # nonlinear elasticity
    # nu_a = Constant(0.48)
    # E_a = 0.5 * (F_a.T * F_a - I)
    # sigma_a = F_a * (nu_a / (1 + nu_a) / (1 - 2 * nu_a) * tr(E_a) * I + 1 / (1 + nu_a) * E_a)

    # ---------------------------------------------------------------------
    # build equations
    # ---------------------------------------------------------------------

    # Stokes equations for the fluid
    FUN1 = (-inner(Sigma_f, grad(v_f)) * r * dx(fluid)
            - J_a * (2 * u_f[1] * v_f[1] / (r + u_a[1]) ** 2 - p_f * v_f[1] / (r + u_a[1])) * r * dx(fluid)
            + inner(lam("+"), v_f("+")) * r("+") * dS)
            # + inner(solid_traction("+"), v_f("+")) * r("+") * dS)

    # Incompressibility for the fluid
    FUN2 = ic_f * q_f * r * dx(fluid) + lam_p * q_f * r * dx(fluid)

    # Nonlinear elasticity for the solid - compressible
    FUN3 = (-inner(Sigma_s, grad(v_s)) * r * dx(solid) - 1 / eps * (1 + u_s[1] / r) * v_s[1] * dx(solid)
            + (1 / eps - lambda_s / eps * (J_s - 1) * J_s) * v_s[1] * r / (r + u_s[1]) * dx(solid)
            # + inner(solid_pressure, v_s) * r * dx(solid)  # test body force
            + inner(solid_traction("-"), v_s("-")) * r("-") * dS  # test traction
            + inner(as_vector([f_0, 0]), v_s) * r * dx(solid)
            - inner(lam("-"), v_s("-")) * r("-") * dS)

    # Continuity of fluid velocity at the solid
    FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("+"), 0])) * r("+") * dS

    # No total axial traction on the solid (ez . sigma_s . n = 0)
    FUN6 = dot(ez, lam("+")) * V_0("+") * r("+") * dS

    # ALE bulk equation
    FUN7 = (-inner(sigma_a, grad(v_a)) * r * dx(fluid)
            - 1 / (1 + nu_a) * u_a[1] * v_a[1] / r * dx(fluid)
            - nu_a / (1 + nu_a) / (1 - 2 * nu_a) * diva(u_a, r) * v_a[1] * dx(fluid)
            + inner(lam_a("+"), v_a("+")) * r("+") * dS)

    # Continuity of fluid and solid displacement
    FUN8 = inner(avg(eta_a), u_a("+") - u_s("-")) * r("+") * dS

    # mean axial solid displacement is zero
    FUN9 = dot(ez, u_s) * g_0 * r * dx(solid)

    # mean fluid pressure is zero
    FUN10 = p_f * eta_p * r * dx(fluid)

    # Combine equations and compute Jacobian
    FUN = [FUN1, FUN2, FUN3, FUN5, FUN6, FUN7, FUN8, FUN9, FUN10]

    JAC = block_derivative(FUN, X, Xt)

    # ---------------------------------------------------------------------
    # set up the solver
    # ---------------------------------------------------------------------

    # Initialize solver
    problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
    solver = BlockPETScSNESSolver(problem)
    solver.parameters.update(snes_solver_parameters["snes_solver"])

    # extract solution components
    (u_f, p_f, u_s, f_0, U_0, lam, lam_p, u_a, lam_a) = X.block_split()

    # ---------------------------------------------------------------------
    # Set up code to save solid quanntities only on the solid domain and
    # fluid quantities only on the fluid domain
    # ---------------------------------------------------------------------

    """
        Separate the meshes
    """
    mesh_f = SubMesh(mesh, subdomains, fluid)
    mesh_s = SubMesh(mesh, subdomains, solid)

    # Create function spaces for the velocity and displacement
    Vf = VectorFunctionSpace(mesh_f, "CG", 1)
    Vs = VectorFunctionSpace(mesh_s, "CG", 1)
    P1v = VectorFunctionSpace(mesh, "DG", 0)
    # stress
    cs = x[0] / sqrt(x[0] ** 2 + x[1] ** 2)
    sn = x[1] / sqrt(x[0] ** 2 + x[1] ** 2)
    A = as_tensor([[cs, sn], [-sn, cs]])

    u_f_only = Function(Vf)
    u_a_only = Function(Vf)
    u_s_only = Function(Vs)


    # Python function to save solution for a given value
    # of epsilon and d_mag
    def save(eps, d_mag):
        u_f_only = project(u_f, Vf)
        u_a_only = project(u_a, Vf)
        u_s_only = project(u_s, Vs)

        MhP = BlockFunctionSpace([P1v], restrict=[Os])
        sig_s = project(eps * (A * Sigma_s_func(u_s, eps) * A.T) * as_vector([1, 0]), MhP.sub(0))
        VhSD = VectorFunctionSpace(mesh_s, 'DG', 0)
        sigma_s = project(sig_s, VhSD)

        u_f_only.rename("u_f", "u_f")
        u_a_only.rename("u_a", "u_a")
        u_s_only.rename("u_s", "u_s")
        sigma_s.rename("sigma", "sigma")

        output_f.write(u_f_only, d_mag)
        output_f.write(u_a_only, d_mag)
        output_s.write(u_s_only, d_mag)
        output_s.write(sigma_s, d_mag)

        # save boundary data
        output_bdry_file = dir + 'validation_bdry_0-' + str(2 * rad)[2:] + '_0-' + str(d_mag)[2:] + '.csv'
        print(output_bdry_file)
        write_list_as_row(output_bdry_file, ['z', 'r', 'u_z', 'u_r'])

        coor = mesh_s.coordinates()
        disp = 0
        for xi in coor:
            # calcuate L2 Norm
            disp += abs(np.linalg.norm(u_s(xi)) - d_mag * np.linalg.norm(xi))
            if near(xi[0] ** 2 + xi[1] ** 2, rad ** 2, DOLFIN_EPS):
                append_list_as_row(output_bdry_file, [float('%.6g' % xi[0]),
                                                      float('%.6g' % xi[1]),
                                                      float('%.6g' % u_s(xi)[0]),
                                                      float('%.6g' % u_s(xi)[1])])
        # Average L2 Norm and save 
        disp = disp / len(coor)
        append_list_as_row(output_data_file, [d_mag, U_0.vector()[0], disp])
        print('d_mag =', d_mag)
        print('Translational speed of the particle: U_0 =', U_0.vector()[0])
        print('disp = ', disp)

    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------

    n = 0

    # last converged value of d_mag
    d_mag_conv = 0

    # increment d_mag and solve
    while d_mag_conv < d_mag_max:

        print('-------------------------------------------------')
        print(f'attempting to solve problem with d_mag = {float(d_mag):.4e}')

        # make a prediction of the next solution based on how the solution
        # changed over the last two increments, e.g. using a simple
        # extrapolation formula
        if n > 1:
            X.block_vector()[:] = X_old.block_vector()[:] + (d_mag_try - d_mag_conv) * dX_dd_mag

        (its, conv) = solver.solve()

        # if the solver converged...
        if conv:

            n += 1
            # update value of eps_conv and save
            d_mag_conv = float(d_mag)
            eps_conv = float(eps)
            save(eps_conv, d_mag_conv)

            # update the value of d_mag to try to solve the problem with
            d_mag_try += d_mag_it

            # copy the converged solution into old solution
            block_assign(X_old, X)

            # print some info to the screen
            print('Translational speed of the particle: U_0 =', U_0.vector()[0])

            # approximate the derivative of the solution wrt d_mag
            if n > 0:
                dX_dd_mag = (X.block_vector()[:] - X_old.block_vector()[:]) / d_mag_it

        # if the solver diverged...
        if not (conv):
            # halve the increment in d_mag
             if not at smallest value
            # and use the previously converged solution as the initial
            # guess
            if d_mag_it > d_mag_it_min:
                d_mag_it /= 2
                d_mag_try = d_mag_conv + d_mag_it
                block_assign(X, X_old)
            else:
                print('min increment reached...aborting')
                save(eps_try, d_mag_try)
                break

        # update the value of d_mag
        d_mag.assign(d_mag_try)
