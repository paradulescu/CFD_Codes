import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

#Ensure Jax runs smoothly
jax.config.update("jax_enable_x64", True)

#find the derivatives of the respective values

#v^n+1/2 solver
#(vn_onehalf-vn)/dt+ (v*dv/dx+u*du/dy)=nu*(d^2u/dx^2+d^2v/dv)
#Solves the momentum eqn for velocity ignoring presure for projection method
def Velocity_Predictor(u,v,dt,dx,dy,F,G,Hx,Hy,Nx,Ny):
#     #Input:
#     #u,v: The u and v direction velocity for each of the cells [Nx+2,Ny+2] arrays
#     #dt,dx,dy: Time step, x step size, y step size
#     #F,G: Horizontal and vertical momentum
#     #Hx,Hy: vectical and horizontal flux of the momentum 
#     #Nx,Ny: Number of nodes in x and y respectivly

    #Define BC indices
    UpperWall=1; LowerWall=Ny+1
    LeftWall=1; RightWall=Nx+1

    u_old=np.copy(u)
    u_new=np.copy(u)
    v_old=np.copy(v)
    v_new=np.copy(v)

    for i in range(LeftWall,RightWall+1):
        for j in range(UpperWall,LowerWall+1):
            
            dF_dx=(F[j, i] - F[j, i-1])/(dx)
            dHx_dy=(Hx[j, i] - Hx[j+1, i])/(dy)
            u_new[j,i]=u_old[j,i] - dt*(dF_dx+dHx_dy)

    for i in range(LeftWall,RightWall+1):
        for j in range(UpperWall,LowerWall+1):

            dG_dy=(G[j-1, i] - G[j, i])/(dy)
            dHy_dx=(Hy[j, i+1] - Hy[j, i])/(dx)
            v_new[j,i]=v_old[j,i] - dt*(dG_dy+dHy_dx)

    return u_new,v_new

def ppe_jax_func(u,v,p,dt,dx,dy,Nx,Ny,leftWall_step,lowerWall_step,max_iters=2000):
    #Jacobi Poission iteration
   
    def body_fun(i, val):
        p=val

        p_old= jnp.copy(p)
        
        UpperWall=1; LowerWall=Ny
        LeftWall=1; RightWall=Nx

        du_dx=(u[1:-1,2:-1] - u[1:-1,1:-2])/(dx)
        dv_dy=(v[1:-2, 1:-1] - v[2:-1, 1:-1])/(dy)

        RHS=(du_dx+dv_dy)/dt
        #set interior values to the 
        p=p.at[1:Ny+1,1:Nx+1].set((RHS-(p[1:-1,2:]+p[1:-1,:-2])/(dx**2)-(p[2:, 1:-1]+p[:-2, 1:-1])/(dy**2))/(-2/dx**2-2/dy**2))
        
        #set the BC pressures to the right values
        p=p.at[:,RightWall].set(0) #outlet; 
        p=p.at[:,RightWall+1].set(0); 

        # p=p.at[:,LeftWall-1].set(p[:,LeftWall]) #inlet

        # p=p.at[LowerWall,:].set(p[LowerWall-1,:]); 
        # p=p.at[UpperWall-1,:].set((p[UpperWall,:]))
        # p=p.at[lowerWall_step,1:leftWall_step+1].set(p[lowerWall_step-1,1:leftWall_step+1])
        # p=p.at[LowerWall+1:LowerWall+1,leftWall_step].set(p[LowerWall+1:LowerWall+1,leftWall_step+1])

        # p=p.at[lowerWall_step+1:LowerWall+4,0:leftWall_step].set(0)

        residual = jnp.linalg.norm(p - p_old)

        return p

    # Using jax.lax.fori_loop for the loop
    p = jax.lax.fori_loop(0, max_iters, body_fun, (p))
    
    return p

def Velocity_Correction(u,v,p,dt,dx,dy,Nx,Ny): #think this is good
    #Input:
    #u,v: The u and v direction velocity for each of the cells [Nx+3,Ny+3] arrays
    #p: The pressure of each of the cells [Nx+2,Ny+2] array
    #dt,dx,dy: Time step, x step, and y step
    #rho: density of the air
    #Nx,Ny: Number of nodes in x and y respectivly
    #Uwall: BC Wall speed

    #Output:
    #u_corrected: the corrected u velocity for all the cells [Nx+3,Ny+3] array
    #v_corrected: the corrected v velocity for all the cells [Nx+3,Ny+3] array

    #define the BC (THIS IS NOT RIGHT FOR FINAL PROJECT)
    LowerWall=Ny+1; UpperWall=1
    LeftWall=1; RightWall=Nx+1
   
    u_corrected=np.copy(u)
    v_corrected=np.copy(v)

    for i in range(LeftWall, RightWall+1):
        for j in range(UpperWall, LowerWall+1):

            dp_dx=(p[j, i] - p[j, i-1])/dx
            dp_dy=(p[j-1, i] - p[j, i])/dy

            u_corrected[j,i]=u[j,i]-dt*(dp_dx)
            v_corrected[j,i]=v[j,i]-dt*(dp_dy)

    return u_corrected,v_corrected
    
def flux(u,v,Nx,Ny,dx,dy,nu,Ls,H): #I think this is good
    #input:
    #u,v: The fluid speed in the x or y direction
    #Nx,Ny: Number of cells in each direction

    #Output:
    #fluxs: The corresponding flux for the given q and phi for all the cells
    #Key: flux=q*phi
    #F : uu = transport of u in the x-direction
    #G : vv = transport of v in the y-direction
    #Hx: vu = transport of u in the y-direction
    #Hy: uv = transport of v in the x-direction
    #Define BC for greater box
    LeftWall=1; RightWall=Nx+1 #LHS edge of the wall
    LowerWall=Ny+1; UpperWall=1 #Lower edge of the wall

    leftWall_step=int(LeftWall+Ls/dx)
    lowerWall_step=int(UpperWall+H/2/dy)

    F=np.zeros((Ny+2,Nx+2)); G=np.zeros((Ny+2,Nx+2))
    Hx=np.zeros((Ny+3,Nx+3)); Hy=np.zeros((Ny+3,Nx+3))

    #phi i,j is i-1/2 and j-1/2
    for j in range(0, Ny+2):
        for i in range(1,Nx+1):
            
            #F first
            ux=(u[j, i+1] - u[j, i])/(dx)

            q_F=(u[j,i+1]+u[j,i])/2

            if q_F>0:
                phi_F=(3*u[j,i+1]+6*u[j,i]-u[j,i-1])/8
            else:
                phi_F=(3*u[j,i]+6*u[j,i+1]-u[j,i+2])/8

            F[j,i]=q_F*phi_F-nu*ux

    for j in range(0, Ny+2):
        for i in range(1,Nx+2):

            #G #need to find this too at inflow and outflow
            vy=(v[j, i] - v[j+1, i])/(dy)

            q_G=(v[j+1,i]+v[j,i])/2
            if q_G>0:
                phi_G=(3*v[j-1,i]+6*v[j,i]-v[j+1,i])/8
            else:
                phi_G=(3*v[j,i]+6*v[j-1,i]-v[j-2,i])/8

            G[j,i]=q_G*phi_G-nu*vy
            
    #current location i or j is technically i+1/2 or j+1/2 for u and v
    for j in range(1, Ny):
        for i in range(1,Nx+2):

            #Hx
            uy=(u[j-1, i] - u[j, i])/(dy)

            q_Hx=(v[j,i]+v[j,i-1])/2
            
            if q_Hx>0:
                phi_Hx=(3*u[j,i]+6*u[j+1,i]-u[j+2,i])/8
            else:
                phi_Hx=(3*u[j+1,i]+6*u[j,i]-u[j-1,i])/8

            Hx[j,i]=q_Hx*phi_Hx-nu*uy

    #find top and bottom wall Hx
    # Hx[0,:]=-nu*2*u[0,:]/dy
    Hx[1,:]=-nu*(u[0,:]- u[1,:])/dy #top wall
    uy_bottom=(u[Ny,:]- u[Ny+1,:])/dy
    Hx[Ny+1,:]=-nu*uy_bottom

    uy_close_bottom=(u[Ny-1,1:Nx+2]- u[Ny,1:Nx+2])/dy
    qHx_close_bottom=(v[Ny,1:Nx+2] + v[Ny,0:Nx+1])/2
    phiHx_closeBottom=(3*u[Ny+1,1:Nx+2]+6*u[Ny,1:Nx+2]-u[Ny-1,1:Nx+2])/8
    Hx[Ny,1:Nx+2]=qHx_close_bottom*phiHx_closeBottom-nu*uy_close_bottom

    Hx[lowerWall_step:Ny+3,leftWall_step+1]=0 #set the left vertical wall to zero
    Hx[lowerWall_step:Ny+3,leftWall_step+2]=0
    Hx[lowerWall_step:Ny+3,leftWall_step]=0

    # uy_inlet=(u[0:lowerWall_step,1] - u[1:lowerWall_step+1,1])/(dy)

    # q_Hx_inlet=(v[1:lowerWall_step+1,1]+v[1:lowerWall_step+1,0])/2
    # phi_Hx_inlet_pos=(3*u[1:lowerWall_step+1,1]+6*u[2:lowerWall_step+2,1]-u[3:lowerWall_step+3,1])/8
    # phi_Hx_inlet_neg=(3*u[2:lowerWall_step+2,1]+6*u[1:lowerWall_step+1,1]-u[0:lowerWall_step,1])/8
    # phi_Hx_inlet=np.where(q_Hx_inlet>0,phi_Hx_inlet_pos,phi_Hx_inlet_neg)

    # Hx[1:lowerWall_step+1,1]=q_Hx_inlet*phi_Hx_inlet-nu*uy_inlet

    # for j in range(1, Ny+1):
    #     for i in range(1,Nx):

    #         #Hy
    #         vx=(v[j, i] - v[j, i-1])/(dx)

    #         q_Hy=(u[j,i]+u[j-1,i])/2

    #         if q_Hy>0:
    #             phi_Hy=phi_G=(3*v[j,i]+6*v[j,i-1]-v[j,i-2])/8
    #         else:
    #             phi_Hy=(3*v[j,i-1]+6*v[j,i]-v[j,i+1])/8

    #         Hy[j,i]=q_Hy*phi_Hy-nu*vx

    vx=(v[2:-2,2:-1]-v[2:-2,1:-2])/dx
    q_Hy=(u[2:-1,2:-2]+u[1:-2,2:-2])/2

    phi_Hy_pos=(3*v[2:-2,2:-1]+6*v[2:-2,1:-2]-v[2:-2,:-3])/8
    phi_Hy_neg=(3*v[2:-2,1:-2]+6*v[2:-2,2:-1]-v[2:-2,3:])/8
    phi_Hy=np.where(q_Hy>0,phi_Hy_pos,phi_Hy_neg)
    Hy[2:-2,2:-2]=q_Hy*phi_Hy-nu*vx

    # #forcfully find the outflow values
    # vx_outflow=(v[2:-2,Nx-1]-v[2:-2,Nx-2])/dx
    # q_Hy_outflow=(u[2:-1,Nx-1]+u[1:-2,Nx-1])/2
    # phi_Hy_pos_outflow=(3*v[2:-2,Nx-1]+6*v[2:-2,Nx-2]-v[2:-2,Nx-3])/8
    # phi_Hy_neg_outflow=(3*v[2:-2,Nx-2]+6*v[2:-2,Nx-1]-v[2:-2,Nx-3])/8
    # phi_Hy_outflow=np.where(q_Hy_outflow>0,phi_Hy_pos_outflow,phi_Hy_neg_outflow)
    # Hy[2:-2,Nx-1]=phi_Hy_outflow-nu*vx_outflow

    #force Hy of the wall to be the right values
    #set inflow and outflow values
    vx_inout=0
    q_Hy_out=(u[1:Ny+1,Nx+1]+u[0:Ny,Nx+1])/2
    phi_Hy_pos_out=(3*v[1:Ny+1,Nx+1]+6*v[1:Ny+1,Nx]-v[1:Ny+1,Nx-1])/8
    # phi_Hy_neg_out=(3*v[1:Ny+1,Nx]+6*v[1:Ny+1,Nx+1]-v[1:Ny+1,Nx+2])/8
    # phi_Hy_out=np.where(q_Hy_out>0,phi_Hy_pos_out,phi_Hy_neg_out)
    Hy[1:Ny+1,Nx+1]=q_Hy_out*phi_Hy_pos_out

    q_Hy_in=(u[1:Ny+1,1+1]+u[0:Ny,2+1])/2
    phi_Hy_pos_in=(3*v[1:Ny+1,2]+6*v[1:Ny+1,2-1]-v[1:Ny+1,2-2])/8
    phi_Hy_neg_in=(3*v[1:Ny+1,2-1]+6*v[1:Ny+1,2]-v[1:Ny+1,2+1])/8
    phi_Hy_int=np.where(q_Hy_in>0,phi_Hy_pos_in,phi_Hy_neg_in)
    Hy[1:Ny+1,1]=q_Hy_out*phi_Hy_int

    #horz walls
    Hy[Ny+2,:]=0
    Hy[Ny+1,:]=0
    Hy[Ny,:]=0
    Hy[1,:]=0
    Hy[lowerWall_step,1:leftWall_step+2]=0
    Hy[lowerWall_step+1,1:leftWall_step+1]=0

    #vertical wall
    vx_vert_wall=(v[lowerWall_step:LowerWall+2,lowerWall_step+1]-v[lowerWall_step:LowerWall+2,lowerWall_step])/dx
    Hy[lowerWall_step:LowerWall+2,leftWall_step]=-nu*vx_vert_wall

    return F, G, Hx, Hy

def Residuals(p,F,G,Hx,Hy,Nx,Ny,dx,dy,lowerwall_step,leftwall_step):
    #input:
    #p: pressure of each of the cells
    #F,G: Horizontal and vertical momentum
    #Hx,Hy: vectical and horizontal flux of the momentum 
    #Nx,Ny: Number of cells in each direction

    #output:
    #Ri, Residual for R[i+1/2,j]
    #Rj, Residual for Ri[i,j+1/2]
    #RL1, L1 norm of the residual

    RL1=0.
    Ls=4

    #Ri first
    for i in range(1, Nx+1):
        for j in range(1,Ny+1):
                if not(i>=Ny/2+2 and (j>=0 and j<=Ls/dx)):
                    Ri=dx*(F[j,i]+p[j,i]-F[j,i-1]-p[j,i-1])+dy*(Hx[j,i]-Hx[j+1,i])
                    RL1+=abs(Ri)

    #Rj
    for i in range(1, Nx+1):
        for j in range(1,Ny+1):
                if not(i>=Ny/2+2 and (j>=0 and j<=Ls/dx)):
                    Rj=dy*(G[j-1,i]+p[j-1,i]-G[j,i]-p[j,i])+dx*(Hy[j,i+1]-Hy[j,i])
                    RL1+=abs(Rj)

    #subtract the empty box part for the RL1
    #Ri first
    # for i in range(1,leftwall_step+1):
    #     for j in range(lowerwall_step,Ny-1):
    #             Ri=dx*(F[j,i]+p[j,i]-F[j,i-1]-p[j,i-1])+dy*(Hx[j,i]-Hx[j+1,i])
    #             RL1-=abs(Ri)

    # #Rj
    # for i in range(1, leftwall_step+1):
    #     for j in range(lowerwall_step,Ny-1):
    #             Rj=dy*(G[j-1,i]+p[j-1,i]-G[j,i]-p[j,i])+dx*(Hy[j,i+1]-Hy[j,i])
    #             RL1-=abs(Rj)     

    return RL1

def Initial_State(dx,dy,L,H,Ubulk):

    Nx=L/dx
    Ny=H/dy
    Nx=int(Nx); Ny=int(Ny)

    u0=np.zeros((Ny+2,Nx+3)) #stored at cell vertical edge
    v0=np.zeros((Ny+3,Nx+2)) #stored at cell horiz edge
    p0=np.zeros((Ny+2,Nx+2)) #stored at cell centers

    y=np.linspace(-H/4,H/4,int(Ny/2))
    Uwall=3/2*Ubulk*(1-(y/(H/2))**2)

    return u0,v0,p0,Nx,Ny,Uwall

#Projection method solver
def ProjectionMethod(Ld,Ls,H,dx,dy,Re,tol=10**-3,Beta=.95,Ubulk=1,MaxIters=5000):
    #input:
    #L,H: Length and height of the 
    ppe_jax=jax.jit(ppe_jax_func, static_argnums=(3,4,5,6,7,8,9,10))
    # ppe_jax=ppe_jax_func

    #initialize the state
    nu=Ubulk*H/Re

    print("nu")

    u0,v0,p0,Nx,Ny,Uwall=Initial_State(dx,dy,Ls+Ld,H,Ubulk)

    R_history=[]
    
    h=np.sqrt(dx**2+dy**2)

    dt=Beta*min(h**2/(nu*4),4*nu/Ubulk**2)
    print("dt",dt)

    RL1=1
    iteration=0
    u=u0; v=v0; p=p0

    #Define BC for greater box
    LeftWall=1; RightWall=Nx+1 #LHS edge of the wall
    LowerWall=Ny+1; UpperWall=1 #Lower edge of the wall

    leftWall_step=int(LeftWall+Ls/dx)
    lowerWall_step=int(UpperWall+H/2/dy)

    #debugging only
    print("Nx",Nx,"Ny",Ny)
    print("leftwall step",leftWall_step)
    print("lowerwall step", lowerWall_step)
    print("LowerWall",LowerWall)
    print("RightWall", RightWall)

    # while RL1>tol:
    while iteration<MaxIters:

        # u=np.random.rand(Ny+2,Nx+3)
        # v=np.random.rand(Ny+3,Nx+2)
        # p=np.random.rand(Ny+2,Nx+2)

        #Set ghost cells for Open ends box Flow
        ##set inflow speed
        u[UpperWall:lowerWall_step,LeftWall]=Uwall
        v[UpperWall:lowerWall_step+1,LeftWall-1]=0

        ##Set outflow speed
        u[:,RightWall]=u[:,RightWall-1]
        u[:,RightWall+1]=u[:,RightWall]
        v[:,RightWall]=v[:,RightWall-1]

        #inflow and outflow BC
        u[:,LeftWall-1]=np.copy(u[:,LeftWall]); 
        u[:,RightWall+1]=np.copy(u[:,RightWall]); 
        u[:,RightWall]=np.copy(u[:,RightWall-1])
        v[:,LeftWall-1]=np.copy(v[:,LeftWall]); 
        v[:,RightWall]=np.copy(v[:,RightWall-1])
        p[:,LeftWall-1]=np.copy(p[:,LeftWall]); 
        p[:,RightWall]=0

        # #cheat outlet
        # v[:,RightWall]=0
        # v[:,RightWall]=0
        
        #Wall of whole box
        u[LowerWall,1:Nx+2]=np.copy(-u[LowerWall-1,1:Nx+2]); 
        u[UpperWall-1,1:Nx+2]=np.copy(-u[UpperWall,1:Nx+2])
        v[LowerWall+1,1:Nx+1]=np.copy(v[LowerWall-1,1:Nx+1]); 
        v[UpperWall-1,1:Nx+1]=np.copy(v[UpperWall+1,1:Nx+1])
        v[LowerWall,1:Nx+1]=0; 
        v[UpperWall,1:Nx+1]=0 #no stick
        p[LowerWall,1:Nx+1]=np.copy(p[LowerWall-1,1:Nx+1]); 
        p[UpperWall-1,1:Nx+1]=np.copy(p[UpperWall,1:Nx+1])

        #Step Lower wall, good
        u[lowerWall_step,1:leftWall_step+2]=np.copy(-u[lowerWall_step-1,1:leftWall_step+2])
        v[lowerWall_step+1,1:leftWall_step+1]=np.copy(v[lowerWall_step-1,1:leftWall_step+1])
        v[lowerWall_step,1:leftWall_step+1]=0
        p[lowerWall_step,1:leftWall_step+1]=np.copy(p[lowerWall_step-1,1:leftWall_step+1])

        #step side wall (this is the half vertical wall)
        u[lowerWall_step:LowerWall+3,leftWall_step]=np.copy(u[lowerWall_step:LowerWall+3,leftWall_step+2])
        u[lowerWall_step:LowerWall+3,leftWall_step+1]=0 #speed of wall is zero
        v[lowerWall_step+1:LowerWall+3,leftWall_step]=np.copy(-v[lowerWall_step+1:LowerWall+3,leftWall_step+1])
        p[lowerWall_step:LowerWall+1,leftWall_step]=np.copy(p[lowerWall_step:LowerWall+3,leftWall_step+1])

        #set the value of p u, and v in the empty part of the step to zero
        u[lowerWall_step+1:Ny+2,0:leftWall_step]=0
        v[lowerWall_step+2:Ny+3,0:leftWall_step]=0
        p[lowerWall_step+1:LowerWall+3,0:leftWall_step]=0

        #get the fluxes of the state
        F, G, Hx, Hy=flux(u,v,Nx,Ny,dx,dy,nu,Ls,H)

        #set fluxes of walls to be correct
        #horizontal walls
        # Hy[lowerWall_step,1:leftWall_step+2]=0; Hy[1,:]=0; Hy[Ny+1,:]=0
        
        #predict the velocity half step thingy
        u_predictor,v_predictor=Velocity_Predictor(u,v,dt,dx,dy,F,G,Hx,Hy,Nx,Ny)

        # # #set u and v to be zero in the empty part of step
        # u_predictor[lowerWall_step+1:Ny+2,0:leftWall_step]=0
        # v_predictor[lowerWall_step+2:Ny+3,0:leftWall_step]=0

        u=np.copy(u_predictor)
        v=np.copy(v_predictor)
        
        #solve PPE for pressure using predicted velocities using Jacobi
        p_jax=jnp.copy(p)
        leftWall_step=int(LeftWall+Ls/dx)
        lowerWall_step=int(UpperWall+H/2/dy)
        u_predictor_jax=jnp.copy(u)
        v_predictor_jax=jnp.copy(v)
        p=ppe_jax(u_predictor_jax,v_predictor_jax,p_jax,dt,dx,dy,Nx,Ny,leftWall_step,lowerWall_step)
        p=np.array(p)
    
        #update velocities using pressures
        u,v=Velocity_Correction(u_predictor,v_predictor,p,dt,dx,dy,Nx,Ny)

        #set the value of p u, and v in the empty part of the step to zero
        u[lowerWall_step:Ny+2,0:leftWall_step-1]=0
        v[lowerWall_step+1:Ny+3,0:leftWall_step-1]=0
        p[lowerWall_step:LowerWall+2,0:leftWall_step-1]=0

        #find the residuals
        RL1=Residuals(p,F,G,Hx,Hy,Nx,Ny,dx,dy,lowerWall_step,leftWall_step)

        ##print out the residual of every x iterations
        if iteration%50==0:
            print("Iteration",iteration)
            print("Residual",RL1)

        #increase iteration and keep track of the residual history
        iteration+=1
        R_history+=[RL1]

        # print("Iteration",iteration)
        # print("Residual",RL1)
        

    return u,v,p,F,G,Hx,Hy,R_history

#testing
Ld=20
Ls=4

L=Ld+Ls
H=2.
Re=200.
dx=0.1
dy=0.1

u,v,p,F,G,Hx,Hy,R_history=ProjectionMethod(Ld,Ls,H,dx,dy,Re)
print("p outlet",p[:,int(L/dx)+1])
print("u Outlet",u[:,int(L/dx)+1])
print("u before step",u[:,int(Ls/dx)])

np.save('Re_200_Mesh1_u.npy',u)
np.save('Re_200_Mesh1_v.npy',v)
np.save('Re_200_Mesh1_p.npy',p)
# np.save('Re_100_Mesh1_R.npy',R_history)

# print("pressure right wall",p[:,1])
# Ubulk=1
# u0,v0,p0,Nx,Ny,Uwall=Initial_State(dx,dy,L,H,Ubulk)

# np.savetxt("u_Mesh2_Re100.csv", u, delimiter=",")
# np.savetxt("v_Mesh2_Re100.csv", v, delimiter=",")
# np.savetxt("p_Mesh2_Re100.csv", p, delimiter=",")

lvls=75
Nx=int(L/dx)
Ny=int(H/dy)
xs=np.linspace(-dx/2,L+dx/2,Nx+2)
ys=np.linspace(H+dy/2,-dy/2,Ny+2)
X,Y=np.meshgrid(xs,ys)

# fig=plt.figure(0)
# ax = fig.add_subplot()
# plt.contourf(X,Y,p,levels=lvls)
# plt.title('Pressure')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_aspect('equal')
# plt.colorbar()

# # fig=plt.figure(1)
# # ax = fig.add_subplot()
# # plt.contourf(X,Y,F,levels=lvls)
# # plt.title('F')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # ax.set_aspect('equal')
# # plt.colorbar()

# # fig=plt.figure(3)
# # ax = fig.add_subplot()
# # plt.contourf(X,Y,G,levels=lvls)
# # plt.title('G')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # ax.set_aspect('equal')
# # plt.colorbar()
# # # plt.xticks(xs) 
# # # plt.yticks(ys)
# # # plt.grid(color='k')

# xs=np.linspace(-dx,L+dx,Nx+3)
# ys=np.linspace(H+dy/2,-dy/2,Ny+2)
# X,Y=np.meshgrid(xs,ys)

# fig=plt.figure(4)
# ax = fig.add_subplot()
# plt.contourf(X,Y,u,levels=lvls)
# plt.title('u')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.xticks(xs) 
# # plt.yticks(ys)
# # plt.grid(color='k')
# ax.set_aspect('equal')
# plt.colorbar()

# xs=np.linspace(-dx/2,L+dx/2,Nx+2)
# ys=np.linspace(H+dy,-dy,Ny+3)
# X,Y=np.meshgrid(xs,ys)

# figtemp=plt.figure(5)
# ax1 = figtemp.add_subplot()
# plt.contourf(X,Y,v,levels=lvls)
# plt.title('v')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.xticks(xs) 
# # plt.yticks(ys)
# # plt.grid(color='k')
# ax1.set_aspect('equal')
# plt.colorbar()

# xs=np.linspace(-dx,L+dx,Nx+3)
# ys=np.linspace(H+dy,-dy,Ny+3)
# X,Y=np.meshgrid(xs,ys)

# # plt.figure(6)
# # ax = fig.add_subplot()
# # plt.contourf(X,Y,Hx,levels=lvls)
# # plt.title('Hx')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # # plt.xticks(xs) 
# # # plt.yticks(ys)
# # # plt.grid(color='k')
# # ax.set_aspect('equal')
# # plt.colorbar()

# # fig=plt.figure(7)
# # ax = fig.add_subplot()
# # plt.contourf(X,Y,Hy,levels=lvls)
# # plt.title('Hy')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # # plt.xticks(xs) 
# # # plt.yticks(ys)
# # # plt.grid(color='k')
# # ax.set_aspect('equal')
# # plt.colorbar()

# ##Streamlines
# psi=np.zeros((Ny,Nx))
# # psiB=psiA-dx*(Va+Vb)/2

# for i in range(1, Nx):
#     for j in range(0, Ny-1):
#         if i == 1:  # For the first row, assume bottom boundary condition
#             psi[j, i] = psi[j-1,i] + u[j,i+1]
#         else:
#             psi[j, i] = psi[j,i-1] - v[j+1,i]

# #Streamlines
# xs=np.linspace(0,L,Nx)
# ys=np.linspace(H,0,Ny)
# X,Y=np.meshgrid(xs,ys)
# fig=plt.figure(8)
# ax = fig.add_subplot()
# plt.contourf(X,Y,psi,levels=lvls)
# plt.contour(X,Y,psi,colors='k')
# plt.title('Stream Function Contour')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()
# ax.set_aspect('equal')

# xs=np.linspace(0,L,Nx)
# ys=np.linspace(H,0,Ny)
# X,Y=np.meshgrid(xs,ys)
# u_center=np.zeros((Ny,Nx))
# v_center=np.zeros((Ny,Nx))

# #plt streamlines
# for i in range(0, Nx):
#     for j in range(0,Ny):
#         u_center[j,i]=(u[j,i]+u[j,i+1])/2
#         v_center[j,i]=(v[j,i]+v[j+1,i])/2

# fig=plt.figure(9)
# ax = fig.add_subplot()
# plt.quiver(X,Y,u_center,v_center,scale=1/0.05)
# plt.title('Stream plot')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_aspect('equal')

# #Analytical
# Ubulk=1
# y=np.linspace(-H/2+dy/2,H/2+dy/2,Ny)
# Uwall=3/4*Ubulk*(1-(y/(H/2))**2)

# #end flow
# u_outflow=u[2:np.shape(u)[0]-1,np.shape(u)[1]-1]
# v_outflow=v[2:np.shape(v)[0]-1,np.shape(v)[1]-1]
# fig=plt.figure(10)
# ax = fig.add_subplot()
# plt.plot(Uwall,ys,label='Analytical u')
# plt.plot(u[1:Ny+1,Nx+1],ys,label='u solver')
# plt.plot(u[1:Ny+1,int(2/dy)],ys,label='u solver x=2')
# plt.plot(u[1:Ny+1,int(5/dy)],ys,label='u solver x=5')
# plt.plot(u[1:Ny+1,int(10/dy)],ys,label='u solver x=10')
# plt.plot(u[1:Ny+1,int(15/dy)],ys,label='u solver x=15')
# # plt.plot(v[1:Ny+1],ys,label='v solver')
# plt.title('Outflow Profile')
# plt.xlabel('speed')
# plt.ylabel('y')
# plt.legend()

# #end flow
# fig=plt.figure(11)
# ax = fig.add_subplot()
# plt.plot(v[1:Ny+1,Nx+1],ys,label='v outlet')
# plt.plot(v[1:Ny+1,int(2/dy)],ys,label='v solver x=2')
# plt.plot(v[1:Ny+1,int(5/dy)],ys,label='v solver x=5')
# plt.plot(v[1:Ny+1,int(10/dy)],ys,label='v solver x=10')
# plt.plot(v[1:Ny+1,int(15/dy)],ys,label='v solver x=15')
# # plt.plot(v[1:Ny+1],ys,label='v solver')
# plt.title('Outflow Profile')
# plt.xlabel('speed')
# plt.ylabel('y')
# plt.legend()

# plt.show()
# plt.close('all')

# fig=plt.figure(3)
# ax = fig.add_subplot()
# plt.quiver(X,Y,u,v,scale=1/0.01,color='k')
# plt.title('velocity')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.xticks(xs) 
# # plt.yticks(ys)
# # plt.grid(color='k')
# ax.set_aspect('equal')
# plt.show()
# plt.close('all')

# plt.plot(xs,np.zeros(len(xs)),'k')
# plt.plot(xs,np.linspace(H,H,len(xs)),'k')
# plt.plot(np.zeros(len(ys)),ys,'k')
# plt.plot(np.linspace(L,L,len(ys)),ys,'k')

#unused code

#v^n+1/2 solver
#(vn_onehalf-vn)/dt+ (v*dv/dx+u*du/dy)=nu*(d^2u/dx^2+d^2v/dv)
#Solves the momentum eqn for velocity ignoring presure for projection method

#solve the pressure Poisson equation (PPE) using Jacobi method
#Laplacian(p^n+1)=-rho*grad_dot(v* dot grad(v*))
def PPE(u,v,p,dt,dx,dy,Nx,Ny,max_iters=1000,tol=10**-6): #THINK THIS IS RIGHT
    #Input:
    #u,v: The u and v direction velocity for each of the cells [Nx+2,Ny+2] arrays
    #p: The pressure of each of the cells [Nx+2,Ny+2] array
    #dt,dx,dy: Time step, x step, and y step
    #rho: density of the air
    #Nx,Ny: Number of nodes in x and y respectivly
    #max_iter: Max number of iterations for Jacobi solver default to 500
    #tol: Tolerance for the residual of the Jacobi solver, default to 10E-6

    #Output:
    #p: new corrected pressure for the problem array of [Nx+2,Ny+2]
    Nx=L/dx
    Ny=H/dy
    Nx=int(Nx); Ny=int(Ny)

    UpperWall=1; LowerWall=Ny
    LeftWall=1; RightWall=Nx

    leftWall_step=int(LeftWall+Ls/dx)
    lowerWall_step=int(UpperWall+H/2/dy)

    #Jacobi Poission iteration
    for iters_p in range(max_iters):  # Poisson iteration
        p_new= np.copy(p)
        p_old = np.copy(p)

        for i in range(LeftWall,RightWall+1):
            for j in range(UpperWall,LowerWall+1):

                #find partials
                du_dx=(u[j, i+1] - u[j, i])/(dx)
                dv_dy=(v[j, i] - v[j+1, i])/(dy)

                #RHS of the PPE
                RHS=(du_dx+dv_dy)/dt

                p_new[j,i]=(RHS-(p[j,i+1]+p[j,i-1])/(dx**2)-(p[j+1,i]+p[j-1,i])/(dy**2))/(-2/dx**2-2/dy**2)
                        
        #set the BC pressures to the right values
        p_new[:,RightWall+1]=0; p_new[:,RightWall]=0 #outlet;  
        p_new[:,LeftWall-1]=np.copy(p_new[:,LeftWall]) #inlet
        p[lowerWall_step,1:leftWall_step+1]=np.copy(p[lowerWall_step-1,1:leftWall_step+1])
        p[LowerWall+1:LowerWall+1,leftWall_step]=np.copy(p[LowerWall+1:LowerWall+1,leftWall_step+1])
        p[lowerWall_step+1:LowerWall+4,0:leftWall_step]=0

        p=np.copy(p_new)

        residual = np.linalg.norm(p - p_old)
        if residual < tol:
            print("Residual of PPE",residual)
            print("iteration when PPE resolved",iters_p)
            return p

    print("Residual of PPE",residual)
    
    return p

def temp(u,v,dt,dx,dy,nu,Nx,Ny):
    #Input:
    #u,v: The u and v direction velocity for each of the cells [Nx+2,Ny+2] arrays
    #dt,dx,dy: Time step, x step, and y step
    #nu: Kinematic Viscosity (given by Re number and IC)
    #Nx,Ny: Number of nodes in x and y respectivly

    #Output:
    #u_predictor: the predicted u velocity for all the cells [Nx+2,Ny+2] array (this is u^(n+1/2))
    #v_predictor: the predicted v velocity for all the cells [Nx+2,Ny+2] array (this is v^(n+1/2))

    #Define BC indices
    #Simple pipe flow (not the actual BC!!)
    LowerWall=1; UpperWall=Ny+1
    LeftWall=1; RightWall=Nx+1

    #create an array for the predictor state
    u_predictor=np.copy(u)
    v_predictor=np.copy(v)

    #iterate through all the cells
    for i in range(LeftWall,RightWall):
        for j in range(LowerWall,UpperWall):

            #get first and 2nd derivative of the u velocity
            d2u_dx2=(u[i+1, j] - 2*u[i, j]+u[i-1, j])/dx**2
            d2u_dy2=(u[i, j+1] - 2*u[i, j]+u[i, j-1])/dy**2
            du_dx=(u[i+1, j] -u[i-1, j])/(2*dx)
            du_dy=(u[i, j+1] -u[i, j-1])/(2*dy)

            #get first and 2nd derivative of the v velocity
            d2v_dy2=(v[i, j+1] - 2*v[i, j]+v[i, j-1])/dy**2
            d2v_dx2=(v[i+1, j] - 2*v[i, j]+v[i-1, j])/dx**2
            dv_dy=(v[i, j+1] -v[i, j-1])/(2*dy)
            dv_dx=(v[i+1, j] -v[i-1, j])/(2*dx)
            
            #solve the momentum eqn ignoring pressure
            u_predictor[i,j]=(nu*(d2u_dx2+d2u_dy2)-(u[i,j]*du_dx+v[i,j]*du_dy))*dt+u[i,j]
            v_predictor[i,j]=(nu*(d2v_dx2+d2v_dy2)-(u[i,j]*dv_dx+u[i,j]*dv_dy))*dt+v[i,j]
            
    return u_predictor,v_predictor

def BC():
    #set BC
        #Vertical Walls
        # for j in range(LowerWall+1,UpperWall):

        #     y=dy*j

        #     du_dy_L=(u[LeftWall, j+1] -u[LeftWall, j-1])/(2*dy)
        #     d2v_dy2_L=(v[LeftWall, j+1] - 2*v[LeftWall, j]+v[LeftWall, j-1])/dy**2

        #     du_dy_R=(u[RightWall, j+1] - u[RightWall, j-1])/(2*dy)
        #     d2v_dy2_R=(v[RightWall, j+1] - 2*v[RightWall, j]+v[RightWall, j-1])/dy**2

        #     u_corrected[LeftWall,j]=Uwall[j]+du_dy_L*y
        #     v_corrected[LeftWall,j]=1/2*d2v_dy2_L*y**2

        #     u_corrected[RightWall,j]=0+du_dy_R*y
        #     v_corrected[RightWall,j]=1/2*d2v_dy2_R*y**2

        # for i in range(LeftWall+1,RightWall):

        #     y_low=LowerWall*j
        #     y_high=LowerWall*j

        #     du_dy_L=(-3*u[i, LowerWall] +4*u[i, LowerWall+1]-u[i, LowerWall+2])/(2*dy)
        #     d2v_dy2_L=(2*v[i, LowerWall] - 5*v[i, LowerWall+1]+4*v[i, LowerWall+2]-v[i,LowerWall+3])/dy**3

        #     du_dy_U=(3*u[i, UpperWall] - 4*u[i, UpperWall-1]+ u[i, UpperWall-2])/(2*dy)
        #     d2v_dy2_U=(2*v[i, UpperWall] - 5*v[i, UpperWall+1]+4*v[i, UpperWall+2]-v[i,UpperWall-3])/dy**3

        #     u_corrected[i,LowerWall]=0+du_dy_L*y_low
        #     v_corrected[i,LowerWall]=1/2*d2v_dy2_L*y_low**2

        #     u_corrected[i,UpperWall]=0+du_dy_U*y_high
        #     v_corrected[i,UpperWall]=1/2*d2v_dy2_U*y_high**2
    return