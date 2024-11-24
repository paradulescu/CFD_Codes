import numpy as np
import jax.numpy as jnp
import math
import jax
from flux import FluxFunction
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys

#Ensure Jax runs smoothly
jax.config.update("jax_enable_x64", True)

#Create a struct to store the info of the cell
##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
#Load mesh and compute normals and all the info needed from cells
##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
def constructMesh(nodefile,elemfile,connectfile):
    #Input
    #Node File: the file that contains the node information. List of nodes and their xy posisitons
    #elemfile: The file that contains the element, has the information on the nodes that form each of the cell triangle.
    #connectfile: File that has the connection information 

    #Output
    #
    node = jnp.load(nodefile)

    elem = jnp.load(elemfile)

    # Prepare line segments for each triangle
    triangle_segments = []
    centroids = []

    for tri in elem:
        # Get the nodes in the triangle
        p1, p2, p3 = node[tri[0]], node[tri[1]], node[tri[2]]
        # Define the lines for the triangle
        triangle_segments.append([p1, p2])
        triangle_segments.append([p2, p3])
        triangle_segments.append([p3, p1])
        centroids.append(jnp.mean(node[tri], axis=0))

    connect = jnp.load(connectfile)
    triangle_segments = []

    return node,elem,connect,centroids,triangle_segments

def cell_data(node,elem):
    #Input:
    #Cell info

    #Output:

    elemx = node[elem, 0]
    elemy = node[elem, 1]

    dx_face0=elemx[:, 1] - elemx[:, 2]; dx_face1=elemx[:, 2] - elemx[:, 0]; dx_face2=elemx[:, 0] - elemx[:, 1]
    dy_face0=elemy[:, 1] - elemy[:, 2]; dy_face1=elemy[:, 2] - elemy[:, 0]; dy_face2=elemy[:, 0] - elemy[:, 1]

    # dx_face0=elemx[:, 2] - elemx[:, 0]; dx_face1=elemx[:, 1] - elemx[:, 2]; dx_face2=elemx[:, 0] - elemx[:, 1]
    # dy_face0=elemy[:, 2] - elemy[:, 0]; dy_face1=elemy[:, 1] - elemy[:, 2]; dy_face2=elemy[:, 0] - elemy[:, 1]

    dx = jnp.column_stack([dx_face0, dx_face1, dx_face2])
    dy = jnp.column_stack([dy_face0, dy_face1, dy_face2])

    sideLengths=jnp.sqrt(dx**2 + dy**2)

    perimeters = jnp.sum(sideLengths, axis=1, keepdims=True)
    perimeters = perimeters.squeeze()

    normal_vec_x=-dy/sideLengths
    normal_vec_y=dx/sideLengths

    semiperm=perimeters/2
    areas = jnp.sqrt(semiperm*(semiperm-sideLengths[:, 0])*(semiperm-sideLengths[:, 1])*(semiperm-sideLengths[:, 2]))

    return sideLengths,normal_vec_x,normal_vec_y,perimeters,areas

##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
#Flux Functions
##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
def wallFlux(uplus,n,gamma):
    #Input
    #uplus: state coming in
    #Fplus: Incoming Flux 
    #n: Normal vector of the elem
    #gamma: gamma of the flow

    #Output:
    #Fluxb: The boundary Flux
    #wavespeed: The wavespeed, used for CFL calcs

    rhoplus=uplus[0]
    Eplus=uplus[3]/rhoplus

    vplus = jnp.array([uplus[1]/rhoplus, uplus[2]/rhoplus])

    vb=vplus-(jnp.dot(vplus,n)*n)

    pb=(gamma-1)*(rhoplus*Eplus-1/2*rhoplus*jnp.sqrt(vb[0]**2+vb[1]**2)**2)
    Fluxb=jnp.array([0,pb*n[0],pb*n[1],0])

    ub=[rhoplus,rhoplus*vb[0],rhoplus*vb[1],uplus[3]]

    flowspeed=jnp.sqrt(vb[0]**2+vb[1]**2)
    c=jnp.sqrt(gamma*pb/rhoplus)
    
    wavespeed=flowspeed+c

    return Fluxb,wavespeed

def inflowFlux(uplus,n,gamma,a0,rho0,alpha):
    #Input
    #uplus: state coming in
    #Fplus: Incoming Flux 
    #n: Normal vector of the elem
    #gamma: gamma of the flow
    #Tt: Total temp inflow
    #pt: Total Pressure
    #alpha: Flow angle

    #Output:
    #Fluxb: The boundary Flux
    #wavespeed: The wavespeed, used for CFL calcs

    a0=1 
    R=287

    pt=rho0*a0**2/gamma
    Tt=pt/(rho0*R)
   
    rhoplus = uplus[0]
    vplus = jnp.array([uplus[1]/rhoplus, uplus[2]/rhoplus])

    pplus=(gamma-1)*(uplus[3]-1/2*rhoplus*(vplus[0]**2+vplus[1]**2))
    cplus=jnp.sqrt(gamma*pplus/rhoplus)
    # Mplus=(jnp.sqrt(vb[0]**2+vb[1]**2))/cplus
    # Tplus=Tt/(1+(gamma-1)/2*Mplus**2)

    vb=vplus-jnp.dot(vplus,n)*n
    #This may be wrong with how I am doing my uplus normal
    Jplus=jnp.dot(vplus,n)+2*cplus/(gamma-1)

    nin=jnp.array([jnp.cos(alpha),jnp.sin(alpha)])
    dn=jnp.dot(nin,n)

    #Find the Mach for boundary
    #Coefficents of the system 
    c2=gamma*R*Tt*dn**2-(gamma-1)/2*Jplus**2
    c1=(4*gamma*R*Tt*dn/(gamma-1))
    c0=4*gamma*R*Tt/((gamma-1)**2)-Jplus**2
    # cs=jnp.array([c2,c1,c0])

    # Use quadratic formula to solve
    Mb1 = (-c1+jnp.sqrt(c1**2 - 4*c2*c0))/(2*c2)
    Mb2 = (-c1-jnp.sqrt(c1**2 - 4*c2*c0))/(2*c2)

    Mb = jnp.where((Mb1 >= 0) & (Mb2 >= 0), jnp.minimum(Mb1, Mb2), 
    jnp.where(Mb1 >= 0, Mb2, jnp.where(Mb2 >= 0, Mb2, jnp.maximum(Mb1, Mb2))))

    Mb=Mb.real             

    Tb=Tt/(1+.5*(gamma-1)*Mb**2)
    pb=pt*(Tb/Tt)**(gamma/(gamma-1))

    rhob=pb/(R*Tb)

    cb=jnp.sqrt(gamma*pb/rhob)

    vb=Mb*cb*nin
    rhoEb=pb/(gamma-1)+0.5*rhob*(vb[0]**2+vb[1]**2)
    Eb=rhoEb/rhob
    H=Eb+pb/rhob
    
    Fluxbx=jnp.array([rhob*vb[0],rhob*vb[0]**2+pb,rhob*vb[0]*vb[1],rhob*vb[0]*H])
    Fluxby=jnp.array([rhob*vb[1],rhob*vb[1]*vb[0],rhob*vb[1]**2+pb,rhob*vb[1]*H])
    
    Fluxb=jnp.array([[Fluxbx[0],Fluxby[0]],[Fluxbx[1],Fluxby[1]],[Fluxbx[2],Fluxby[2]],[Fluxbx[3],Fluxby[3]]])
    Fluxb=jnp.dot(Fluxb,n)
    
    ub=[rhob,rhob*vb[0],rhob*vb[1],rhoEb]
    flowspeed=jnp.sqrt(vb[0]**2+vb[1]**2)
    c=jnp.sqrt(gamma*pb/rhob)
    wavespeed=flowspeed+c

    return Fluxb,wavespeed

def outflowFlux(uplus,n,gamma,pb):
    #Input
    #uplus: state coming in
    #n: Normal vector of the elem
    #gamma: gamma of the flow
    #pb: boundary pressure

    #Output:
    #Fluxb: The boundary Flux
    #wavespeed: The wavespeed, used for CFL calcs

    rhoplus=uplus[0]
    Eplus=uplus[3]/rhoplus
    vplus=jnp.array([uplus[1]/rhoplus,uplus[2]/rhoplus])

    pplus=(gamma-1)*(rhoplus*Eplus-1/2*rhoplus*jnp.sqrt(vplus[0]**2+vplus[1]**2)**2)
    cplus=jnp.sqrt(gamma*pplus/rhoplus)

    Splus=pplus/(rhoplus**gamma)
    rhob=(pb/Splus)**(1/gamma)

    cb=jnp.sqrt(gamma*pb/rhob)
    Jplus=jnp.dot(vplus,n)+2*cplus/(gamma-1)

    ubn=Jplus-2*cb/(gamma-1)

    vb=vplus-(jnp.dot(vplus,n)*n)+ubn*n

    rhoEb=pb/(gamma-1)+1/2*rhob*(vb[0]**2+vb[1]**2)
    Eb=rhoEb/rhob
    H=Eb+pb/rhob

    Fluxbx=jnp.array([rhob*vb[0],rhob*vb[0]**2+pb,rhob*vb[0]*vb[1],rhob*vb[0]*H])
    Fluxby=jnp.array([rhob*vb[1],rhob*vb[1]*vb[0],rhob*vb[1]**2+pb,rhob*vb[1]*H])
    
    Fluxb=jnp.array([[Fluxbx[0],Fluxby[0]],[Fluxbx[1],Fluxby[1]],[Fluxbx[2],Fluxby[2]],[Fluxbx[3],Fluxby[3]]])
    Fluxb=jnp.dot(Fluxb,n)

    ub=[rhob,rhob*vb[0],rhob*vb[1],rhoEb]
    flowspeed=jnp.sqrt(vb[0]**2+vb[1]**2)
    c=jnp.sqrt(gamma*pb/rhob)
    wavespeed=flowspeed+c

    return Fluxb,wavespeed

##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
#Finite Elemenet Solver
##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
def initial_state(M,rho0,a0,alpha):
    #Creates the initial states of the function
    #Input:
    #M,rho0,a0,alpha: Mach, stagnation densitity, stagnation speed of sound, and angle of attack of the flow

    #Output:
    #u: state

    #Output
    g=1.4

    rho=rho0*(1+(g-1)/2*M**2)**(-1/(g-1))
    a=a0*(jnp.sqrt((1+((g-1)/2)*M**2)**-1))
    p=a**2/g*rho
    speed=M/a
    u=jnp.cos(alpha)*speed
    v=jnp.sin(alpha)*speed
    rhoE=p/(g-1)+1/2*rho*speed**2

    u=jnp.array([rho,rho*u,rho*v,rhoE])

    return u

def calc_flux(UL,n,FaceRightElem,a0,rho0,alpha,pb,sideLengths,face_num):
    #use conditional logic to get the right function, calcs flux for just one face
    #Input:
    #UL: State on the left side of the face
    #n: Normal vector for the current face
    #neighbor: vector of neighbor cell
    #a0, rho0: stagnation speed of sound and density, used for boundary conditions
    #alpha, pb: The angle of attack and boundary pressure boundary condition values
    #sidelengths: Vector of the length of all the edges on the currecnt face
    #face_num: The face number that is currently having the flux being calced, ex 0,1,2

    #Output:
    #Flux:Flux vector for the current face for all the cells
    #Wavespeed: The wavespeed for the current face for all the cells

    ##FREE STREAM TEST ONLY

    #vmap the functions to work on vectors
    roe = jax.vmap(FluxFunction, in_axes=(0, 0, None, 0))
    inflow = jax.vmap(inflowFlux, in_axes=(0, 0, None, None, None,None))
    outflow = jax.vmap(outflowFlux, in_axes=(0, 0, None,None))
    wall = jax.vmap(wallFlux, in_axes=(0, 0,None))

    g=1.4

    #Find index of the elements
    index_roe=jnp.where(FaceRightElem>=0)[0]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    index_inflow=jnp.where(FaceRightElem==-1)[0]
    index_outflow=jnp.where(FaceRightElem==-2)[0]

    Flux=jnp.zeros((len(sideLengths),4))
    wavespeed=jnp.zeros(len(sideLengths))

    UR=UL[FaceRightElem,:]
    # UR=u0

    #Roe Flux
    roe_flux,wavespeed_roe=roe(UL[index_roe,:],UR[index_roe,:],g,n[index_roe,:])
    Flux = Flux.at[index_roe,:].set(roe_flux)
    wavespeed = wavespeed.at[index_roe].set(wavespeed_roe)
    
    if len(index_wall)>0: #Wall Flux
        flux_wall,wavespeed_wall=wall(UL[index_wall,:],n[index_wall,:],g)
        Flux = Flux.at[index_wall,:].set(flux_wall)
        wavespeed = wavespeed.at[index_wall].set(wavespeed_wall)

    if len(index_outflow)>0: #Outflow Flux
        flux_outflow,wavespeed_outflow=outflow(UL[index_outflow,:],n[index_outflow,:],g,pb)
        Flux = Flux.at[index_outflow,:].set(flux_outflow)
        wavespeed = wavespeed.at[index_outflow].set(wavespeed_outflow)

    if len(index_inflow)>0: #Inflow Flux
        flux_inflow,wavespeed_inflow=inflow(UL[index_inflow,:],n[index_inflow,:],g,a0,rho0,alpha)
        Flux = Flux.at[index_inflow,:].set(flux_inflow)
        wavespeed = wavespeed.at[index_inflow].set(wavespeed_inflow)

    wavespeed=wavespeed*sideLengths[:,face_num]

    return Flux, wavespeed

def solver(connect,sideLengths,normal_vec_x,normal_vec_y,areas,u0,a0,rho0,alpha,tol=10**-5,CFL=1,maxIter=10000):
    #input:
    #connect: Cell information about neighbors and BC
    #normal_vec_x,normal_vec_y: Normal vectors
    #sideLengths,areas: Side lengths [Face0,Face1,Face2] and areas of the cells
    #u0: intial state
    #a0,rho0,alpha: Boundary Condtion values, stagnation speed of sound, pressure, and angle of attack
    #tol=10**-5,CFL=1,maxIter=10000: Preset defult values for solver, tolerance of residual, CFL #, and Max number of iterations allowed

    #output:
    #u: Final state of mesh
    #Res_hist,k: History of the residual vector, and the number of iterations (scalar)

    u=u0 #set the initial state to be the 
    Res_hist = []                           
    number_cells=len(areas)
    k=0
    pb=0.8*p0

    while (k<maxIter):
        #create vectors of the residuals and wavespeed
        Residual=jnp.zeros((number_cells,4))
        wavespeed_iter=np.zeros((len(sideLengths)))

        for face_num in range(3): #loops through the faces

            n = jnp.column_stack([normal_vec_x[:,face_num],normal_vec_y[:,face_num]])

            FaceRightElem=connect[:,face_num]
            
            flux,wavespeed=calc_flux(u,n,FaceRightElem,a0,rho0,alpha,pb,sideLengths,face_num)
            wavespeed_iter=wavespeed_iter+wavespeed

            Residual = Residual + flux*sideLengths[:, face_num][:, None]

        R_total = jnp.sum(jnp.sum(jnp.abs(Residual)))

        Res_hist.append(R_total)

        if math.isnan(R_total):
            print("Iteration breaking",k)
            return
        
        delt = (2*CFL*areas)/wavespeed_iter
        u = u - delt[:, None]*Residual / areas[:, None]

        if R_total<tol:
            print("Converged")
            return u,Res_hist,k
        
        if k%100==0:
            print("Iteration",k)
            print("Residual",R_total)
        
        k+=1
        
    return u,Res_hist,k

def calc_flux_freestream(UL,n,neighbor,sideLengths,face_num):
    #FREE STREAM ONLY
    #Input:
    #UL: State on the left side of the face
    #n: Normal vector for the current face
    #neighbor: vector of neighbor cell
    #a0, rho0: stagnation speed of sound and density, used for boundary conditions
    #alpha, pb: The angle of attack and boundary pressure boundary condition values
    #sidelengths: Vector of the length of all the edges on the currecnt face
    #face_num: The face number that is currently having the flux being calced, ex 0,1,2

    #Output:
    #Flux:Flux vector for the current face for all the cells
    #Wavespeed: The wavespeed for the current face for all the cells

    ##FREE STREAM TEST ONLY
    g=1.4
    
    roe = jax.vmap(FluxFunction, in_axes=(0, 0, None, 0))

    Flux=jnp.zeros((len(sideLengths),4))
    wavespeed=jnp.zeros(len(sideLengths))

    index_roe=jnp.where(neighbor>=0)[0]
    index_wall=jnp.where(neighbor==-3)[0]
    index_inflow=jnp.where(neighbor==-1)[0]
    index_outflow=jnp.where(neighbor==-2)[0]

    UR=UL[neighbor,:]
    
    roe_flux,wavespeed_roe=roe(UL[index_roe,:],UR[index_roe,:],g,n[index_roe,:])
    Flux = Flux.at[index_roe,:].set(roe_flux)
    wavespeed = wavespeed.at[index_roe].set(wavespeed_roe)

    if index_wall.size>0:
        flux_wall,wavespeed_wall=roe(UL[index_wall,:],UR[index_wall,:],g,n[index_wall,:])
        Flux = Flux.at[index_wall,:].set(flux_wall)
        wavespeed = wavespeed.at[index_wall].set(wavespeed_wall)

    if index_outflow.size>0:
        flux_outflow,wavespeed_outflow=roe(UL[index_outflow,:],UR[index_outflow,:],g,n[index_outflow,:])
        Flux = Flux.at[index_outflow,:].set(flux_outflow)
        wavespeed = wavespeed.at[index_outflow].set(wavespeed_outflow)

    if index_inflow.size>0:
        flux_inflow,wavespeed_inflow=roe(UL[index_inflow,:],UR[index_inflow,:],g,n[index_inflow,:])
        Flux = Flux.at[index_inflow,:].set(flux_inflow)
        wavespeed = wavespeed.at[index_inflow].set(wavespeed_inflow)

    wavespeed=wavespeed*sideLengths[:,face_num]

    return Flux, wavespeed

def solver_freestream(connect,sideLengths,normal_vec_x,normal_vec_y,areas,u0,CFL=1,maxIter=1000):
    #FUNCTION ONLY FOR FREESTREAM
    
    #input:
    #connect: Cell information about neighbors and BC
    #normal_vec_x,normal_vec_y: Normal vectors
    #sideLengths,areas: Side lengths [Face0,Face1,Face2] and areas of the cells
    #u0: intial state
    #a0,rho0,alpha: Boundary Condtion values, stagnation speed of sound, pressure, and angle of attack
    #tol=10**-5,CFL=1,maxIter=10000: Preset defult values for solver, tolerance of residual, CFL #, and Max number of iterations allowed

    #output:
    #u: Final state of mesh
    #Res_hist,k: History of the residual vector, and the number of iterations (scalar)

    u=u0 #set the initial state to be the 
    Res_hist = []                           
    number_cells=len(areas)
    k=0
    pb=0.8*p0

    while (k<maxIter):
        #create vectors of the residuals and wavespeed
        Residual=jnp.zeros((number_cells,4))
        wavespeed_iter=np.zeros((len(sideLengths)))

        for face_num in range(3): #loops through the faces

            n = jnp.column_stack([normal_vec_x[:,face_num],normal_vec_y[:,face_num]])

            neighbor=connect[:,face_num]
            
            flux,wavespeed=calc_flux_freestream(u,n,neighbor,sideLengths,face_num)
            wavespeed_iter=wavespeed_iter+wavespeed

            Residual = Residual + flux*sideLengths[:, face_num][:, None]

        R_total = jnp.sum(jnp.sum(jnp.abs(Residual)))

        Res_hist.append(R_total)

        if math.isnan(R_total):
            print("Iteration breaking",k)
            return
        
        delt = (2*CFL*areas)/wavespeed_iter
        u = u - delt[:, None]*Residual / areas[:, None]
        
        if k%100==0:
            print("Iteration",k)
            print("Residual",R_total)
        
        k+=1
        
    return u,Res_hist,k

##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
#Testing
##---------------------------------------------------------------------------------------------------##
##---------------------------------------------------------------------------------------------------##
#Unit test Flux
def testFlux():
    #set up state
    rho=1
    vel=1
    alpha=0.785398
    gamma=1.4
    n0=jnp.array([0,1])
    n90=jnp.array([1,0])
    n45=jnp.array([jnp.sqrt(2)/2,jnp.sqrt(2)/2])

    u_wall_0=[rho,rho*vel,0,1]
    u_wall_45=[rho,rho*vel*jnp.cos(alpha),rho*vel*jnp.cos(alpha),1]
    u_wall_90=[rho,0,rho*vel,1]

    flux_wall_0,dump=wallFlux(u_wall_0,-n0,gamma);flux_wall_45,dump=wallFlux(u_wall_45,n45,gamma)
    flux_wall_90,dump=wallFlux(u_wall_90,n90,gamma)

    print("Wall Flux, 0 Degrees",flux_wall_0,"45 Degrees",flux_wall_45,"90 Degrees",flux_wall_90)

    a0=1
    rho0=1
    alpha=0
    flux_inflow_0,dump=inflowFlux(u_wall_0,-1*n90,gamma,a0,rho0,alpha)
    flux_inflow_45,dump=inflowFlux(jnp.array([1,jnp.cos(0.785398),jnp.cos(0.785398),1]),jnp.array([-1,0]),gamma,a0,rho0,0.785398)

    print("Inflow Flux, 0 Degrees",flux_inflow_0,np.shape(flux_inflow_0))
    print("Inflow Flux, 45 Degrees",flux_inflow_45,np.shape(flux_inflow_45))

    flux_outflow_0,dump=outflowFlux(u_wall_0,n90,gamma,1)
    flux_outflow_45,dump=outflowFlux(jnp.array([1,jnp.cos(0.785398),jnp.cos(0.785398),1]),jnp.array([1,0]),gamma,2)

    print("outflow Flux,pb=1 ",flux_outflow_0)
    print("outflow Flux,pb=2,v=[.707,.707]",flux_outflow_45)

# testFlux()

R=287.; g=1.4
M=0.1 
rho0=1
a0=1
alpha=jnp.pi/180*5

u_initial=initial_state(M,rho0,a0,alpha)
rho=u_initial[0]; vel=[u_initial[1]/rho,u_initial[2]/rho]
p=(g-1)*(u_initial[3]-1/2*rho*jnp.sqrt(vel[0]**2+vel[1]**2)**2)
T=p/(rho*R)
Tt=T/(jnp.sqrt((1+((g-1)/2)*M**2)**-1))
p0=rho0*a0**2/g
pb=0.8*p0


# # # temp_inflow=inflowFlux(u_initial,jnp.array([-1,0]),g,a0,rho0,alpha)
node,elem,connect,centroids,triangle_segments=constructMesh('airfoil0.node.npy','airfoil0.elem.npy','airfoil0.connect.npy')
# #node,elem,connect,centroids,triangle_segments=constructMesh('airfoil1.node.npy','airfoil1.elem.npy','airfoil1.connect.npy')
# #node,elem,connect,centroids,triangle_segments=constructMesh('airfoil2.node.npy','airfoil2.elem.npy','airfoil2.connect.npy')


# #print("node",np.shape(node),"elem",np.shape(elem),"centroids",np.shape(centroids),"traingle",np.shape(triangle_segments),"normal")
sideLengths,normal_vec_x,normal_vec_y,perimeter,areas=cell_data(node,elem)
# print("Loaded Mesh stuff")

# numelems=len(areas)
# u0=np.zeros((numelems,4))
# u0[:,0]=u_initial[0];u0[:,1]=u_initial[1];u0[:,2]=u_initial[2];u0[:,3]=u_initial[3]
# u0=jnp.array(u0)

# u_final,rhistory,k=solver(connect,sideLengths,normal_vec_x,normal_vec_y,areas,u0,a0,rho0,alpha)

# np.save('Airfoil0_state.npy',u_final)
# np.save('Airfoil0_residual.npy',rhistory)

