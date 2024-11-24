import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from project2 import wallFlux
from project2 import inflowFlux
from project2 import outflowFlux
from project2 import constructMesh
from project2 import cell_data

#Ensure Jax runs smoothly
jax.config.update(        "jax_enable_x64", True)

#Testing
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

def Airfoil0():
    g=1.4
    node,elem,connect,centroids,triangle_segments=constructMesh('airfoil0.node.npy','airfoil0.elem.npy','airfoil0.connect.npy')
    u_final=np.load('Airfoil0_state.npy')
    rhistory=np.load('Airfoil0_residual.npy')
    print(np.shape(u_final))

    #Mach number
    rho_final=u_final[:,0]
    u_vel_final=u_final[:,1]/rho_final
    v_vel_final=u_final[:,2]/rho_final
    p_final=(g-1)*(rho_final*u_final[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    c_final=jnp.sqrt(g*p_final/rho_final)
    speed=jnp.sqrt(u_vel_final**2+v_vel_final**2)
    M_final=speed/c_final

    # #Residual plot
    # plt.figure(0)
    # plt.plot(jnp.arange(0,len(rhistory)),rhistory)
    # plt.yscale('log')
    # plt.ylabel('Residual (L1 norm)')
    # plt.xlabel('Iteration')
    # plt.title('Residual History for Solver on Airfoil0')
    # plt.show()
    # plt.close('all')

    #Mach number plot
    plt.figure(1)
    plt.title("Mach for Airfoil 0")
    plt.tripcolor(node[:,0],node[:,1],elem,facecolors=M_final,shading='flat')
    print("Max Mach",jnp.max(M_final))
    print("Min Mach",jnp.min(M_final))
    print("avg Mach",jnp.sum(M_final)/len(M_final))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    plt.close('all')

    # #pressure number plot
    # plt.figure(2)
    # plt.title("Pressure")
    # plt.tripcolor(node[:,0],node[:,1],elem,facecolors=p_final,shading='flat')
    # plt.colorbar()
    # plt.show()

def Airfoil1():
    g=1.4
    node,elem,connect,centroids,triangle_segments=constructMesh('airfoil1.node.npy','airfoil1.elem.npy','airfoil1.connect.npy')
    u_final=np.load('Airfoil1_state.npy')
    rhistory=np.load('Airfoil1_residual.npy')
    print(np.shape(u_final))

    #Mach number
    rho_final=u_final[:,0]
    u_vel_final=u_final[:,1]/rho_final
    v_vel_final=u_final[:,2]/rho_final
    p_final=(g-1)*(rho_final*u_final[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    c_final=jnp.sqrt(g*p_final/rho_final)
    speed=jnp.sqrt(u_vel_final**2+v_vel_final**2)
    M_final=speed/c_final

    # #Residual plot
    # plt.figure(0)
    # plt.plot(jnp.arange(0,len(rhistory)),rhistory)
    # plt.yscale('log')
    # plt.ylabel('Residual (L1 norm)')
    # plt.xlabel('Iteration')
    # plt.title('Residual History for Solver on Airfoil1')
    # plt.show()
    # plt.close('all')

    #Mach number plot
    plt.figure(1)
    plt.title("Mach for Airfoil 1")
    plt.tripcolor(node[:,0],node[:,1],elem,facecolors=M_final)
    print("Max Mach",jnp.max(M_final))
    print("Min Mach",jnp.min(M_final))
    print("avg Mach",jnp.sum(M_final)/len(M_final))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    plt.close('all')

    # #pressure number plot
    # plt.figure(2)
    # plt.title("Pressure")
    # plt.tripcolor(node[:,0],node[:,1],elem,facecolors=p_final)
    # plt.colorbar()
    # plt.show()

def Airfoil2():
    g=1.4
    node,elem,connect,centroids,triangle_segments=constructMesh('airfoil2.node.npy','airfoil2.elem.npy','airfoil2.connect.npy')
    u_final=np.load('Airfoil2_state.npy')
    rhistory=np.load('Airfoil2_residual.npy')
    print(np.shape(u_final))

    #Mach number
    rho_final=u_final[:,0]
    u_vel_final=u_final[:,1]/rho_final
    v_vel_final=u_final[:,2]/rho_final
    p_final=(g-1)*(rho_final*u_final[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    c_final=jnp.sqrt(g*p_final/rho_final)
    speed=jnp.sqrt(u_vel_final**2+v_vel_final**2)
    M_final=speed/c_final

    # #Residual plot
    # plt.figure(0)
    # plt.plot(jnp.arange(0,len(rhistory)),rhistory)
    # plt.yscale('log')
    # plt.ylabel('Residual (L1 norm)')
    # plt.xlabel('Iteration')
    # plt.title('Residual History for Solver on Airfoil2')
    # plt.show()
    # plt.close('all')

    #Mach number plot
    plt.figure(1)
    plt.title("Mach for Airfoil2")
    plt.tripcolor(node[:,0],node[:,1],elem,facecolors=M_final)
    print("Max Mach",jnp.max(M_final))
    print("Min Mach",jnp.min(M_final))
    print("avg Mach",jnp.sum(M_final)/len(M_final))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    plt.close('all')

    #Zoomed in Plot Mach
    plt.figure(1)
    plt.title("Mach for Airfoil2")
    plt.tripcolor(node[:,0],node[:,1],elem,facecolors=M_final)
    print("Max Mach",jnp.max(M_final))
    print("Min Mach",jnp.min(M_final))
    print("avg Mach",jnp.sum(M_final)/len(M_final))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3,0)
    plt.ylim(-1.5,1.5)
    plt.colorbar()
    plt.show()
    plt.close('all')

    # #pressure number plot
    # plt.figure(2)
    # plt.title("Pressure Airfoil2")
    # plt.tripcolor(node[:,0],node[:,1],elem,facecolors=p_final)
    # plt.colorbar()
    # plt.show()

def freestream():
    g=1.4
    node,elem,connect,centroids,triangle_segments=constructMesh('airfoil0.node.npy','airfoil0.elem.npy','airfoil0.connect.npy')
    rhistory=np.load('freestream_residual.npy')
    
    #Residual plot
    plt.figure(0)
    plt.plot(jnp.arange(0,len(rhistory)),rhistory)
    plt.yscale('log')
    plt.ylabel('Residual (L1 norm)')
    plt.xlabel('Iteration')
    plt.title('Residual History for Solver on Freestream')
    plt.show()
    plt.close('all')

def plot_cp():
    #set constants
    g=1.4
    rho0=1
    a0=1
    p0=rho0*a0**2/g
    pout=0.8*p0
    Mout2=2/(g-1)*((p0/pout)**((g-1)/g)-1)
    qout=1/2*g*pout*Mout2

    node_airfoil0,elem_airfoil0,connect_airfoil0,dump,dump=constructMesh('airfoil0.node.npy','airfoil0.elem.npy','airfoil0.connect.npy')
    u_airfoil0=np.load('Airfoil0_state.npy')

    node_airfoil1,elem_airfoil1,connect_airfoil1,dump,dump=constructMesh('airfoil1.node.npy','airfoil1.elem.npy','airfoil1.connect.npy')
    u_airfoil1=np.load('Airfoil1_state.npy')

    node_airfoil2,elem_airfoil2,connect_airfoil2,dump,dump=constructMesh('airfoil2.node.npy','airfoil2.elem.npy','airfoil2.connect.npy')
    u_airfoil2=np.load('Airfoil2_state.npy')

    #Do airfoil0 first
    FaceRightElem=connect_airfoil0[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil0=u_airfoil0[index_wall]
    nodes_wall=elem_airfoil0[index_wall,1]
    xs0=node_airfoil0[nodes_wall,0]
    #Get vals
    rho_final=Wall_airfoil0[:,0]
    u_vel_final=Wall_airfoil0[:,1]/rho_final
    v_vel_final=Wall_airfoil0[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil0[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    cp_airfoil0=(p_final-pout)/qout

    #Airfoil 1
    FaceRightElem=connect_airfoil1[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil=u_airfoil1[index_wall]
    nodes_wall=elem_airfoil1[index_wall,1]
    xs1=node_airfoil1[nodes_wall,0]
    #Get vals
    rho_final=Wall_airfoil[:,0]
    u_vel_final=Wall_airfoil[:,1]/rho_final
    v_vel_final=Wall_airfoil[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    cp_airfoil1=(p_final-pout)/qout

    #Airfoil 2
    FaceRightElem=connect_airfoil2[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil=u_airfoil2[index_wall]
    nodes_wall=elem_airfoil2[index_wall,1]
    xs2=node_airfoil2[nodes_wall,0]
    #Get vals
    rho_final=Wall_airfoil[:,0]
    u_vel_final=Wall_airfoil[:,1]/rho_final
    v_vel_final=Wall_airfoil[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    cp_airfoil2=(p_final-pout)/qout

    plt.figure(0)
    # plt.plot(xs0,cp_airfoil0,color='blue',label='airfoil0')
    # plt.plot(xs1,cp_airfoil1,color='red',label='airfoil1')
    # plt.plot(xs2,cp_airfoil2,color='black',label='airfoil2')
    plt.scatter(xs0,cp_airfoil0,color='blue',label='airfoil0',s=10)
    plt.scatter(xs1,cp_airfoil1,color='red',label='airfoil1',s=10)
    plt.scatter(xs2,cp_airfoil2,color='black',label='airfoil2',s=10)
    plt.xlabel('x')
    plt.ylabel('cp')
    # plt.title('Cp Plot for All Three Airfoils')
    plt.legend()
    plt.show()
    plt.close('all')

def Forces():
    #set constants
    g=1.4
    rho0=1
    a0=1
    p0=rho0*a0**2/g
    pout=0.8*p0
    Mout2=2/(g-1)*((p0/pout)**((g-1)/g)-1)
    qout=1/2*g*pout*Mout2

    node_airfoil0,elem_airfoil0,connect_airfoil0,dump,dump=constructMesh('airfoil0.node.npy','airfoil0.elem.npy','airfoil0.connect.npy')
    sideLengths_airfoil0,normal_vec_x_airfoil0,normal_vec_y_airfoil0,dump,dump=cell_data(node_airfoil0,elem_airfoil0)
    u_airfoil0=np.load('Airfoil0_state.npy')

    node_airfoil1,elem_airfoil1,connect_airfoil1,dump,dump=constructMesh('airfoil1.node.npy','airfoil1.elem.npy','airfoil1.connect.npy')
    sideLengths_airfoil1,normal_vec_x_airfoil1,normal_vec_y_airfoil1,dump,dump=cell_data(node_airfoil1,elem_airfoil1)
    u_airfoil1=np.load('Airfoil1_state.npy')

    node_airfoil2,elem_airfoil2,connect_airfoil2,dump,dump=constructMesh('airfoil2.node.npy','airfoil2.elem.npy','airfoil2.connect.npy')
    sideLengths_airfoil2,normal_vec_x_airfoil2,normal_vec_y_airfoil2,dump,dump=cell_data(node_airfoil2,elem_airfoil2)
    u_airfoil2=np.load('Airfoil2_state.npy')

    #Do airfoil0 first
    FaceRightElem=connect_airfoil0[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil0=u_airfoil0[index_wall]
    nodes_wall=elem_airfoil0[index_wall,1]
    xs0=node_airfoil0[nodes_wall,0]
    normals=jnp.column_stack([normal_vec_x_airfoil0[index_wall,2],normal_vec_y_airfoil0[index_wall,2]])
    #Get vals
    rho_final=Wall_airfoil0[:,0]
    u_vel_final=Wall_airfoil0[:,1]/rho_final
    v_vel_final=Wall_airfoil0[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil0[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    p_final=jnp.array(p_final)
    Fx=p_final*normals[:,0]*sideLengths_airfoil0[index_wall,2]
    Fx=jnp.sum(Fx)
    Fy=p_final*normals[:,1]*sideLengths_airfoil0[index_wall,2]
    Fy=jnp.sum(Fy)
    Cx=Fx/(qout*1);Cy=Fy/(qout*1)

    print("Airfoil 0")
    print("Cx",Cx)
    print("Cy",Cy)

    #Do airfoil1
    FaceRightElem=connect_airfoil1[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil=u_airfoil1[index_wall]
    nodes_wall=elem_airfoil1[index_wall,1]
    normals=jnp.column_stack([normal_vec_x_airfoil1[index_wall,2],normal_vec_y_airfoil1[index_wall,2]])
    #Get vals
    rho_final=Wall_airfoil[:,0]
    u_vel_final=Wall_airfoil[:,1]/rho_final
    v_vel_final=Wall_airfoil[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    p_final=jnp.array(p_final)
    Fx=p_final*normals[:,0]*sideLengths_airfoil1[index_wall,2]
    Fx=jnp.sum(Fx)
    Fy=p_final*normals[:,1]*sideLengths_airfoil1[index_wall,2]
    Fy=jnp.sum(Fy)
    Cx=Fx/(qout*1);Cy=Fy/(qout*1)

    print("Airfoil 1")
    print("Cx",Cx)
    print("Cy",Cy)

    #Do airfoil2
    FaceRightElem=connect_airfoil2[:,2]
    index_wall=jnp.where(FaceRightElem==-3)[0]
    Wall_airfoil=u_airfoil2[index_wall]
    nodes_wall=elem_airfoil2[index_wall,1]
    normals=jnp.column_stack([normal_vec_x_airfoil2[index_wall,2],normal_vec_y_airfoil2[index_wall,2]])
    #Get vals
    rho_final=Wall_airfoil[:,0]
    u_vel_final=Wall_airfoil[:,1]/rho_final
    v_vel_final=Wall_airfoil[:,2]/rho_final
    p_final=(g-1)*(rho_final*Wall_airfoil[:,3]-1/2*rho_final*jnp.sqrt(u_vel_final**2+v_vel_final**2)**2)
    p_final=jnp.array(p_final)
    Fx=p_final*normals[:,0]*sideLengths_airfoil2[index_wall,2]
    Fx=jnp.sum(Fx)
    Fy=p_final*normals[:,1]*sideLengths_airfoil2[index_wall,2]
    Fy=jnp.sum(Fy)
    Cx=Fx/(qout*1);Cy=Fy/(qout*1)

    print("Airfoil 2")
    print("Cx",Cx)
    print("Cy",Cy)

def StateToCSV():
    u_final=np.load('Airfoil0_state.npy')
    np.savetxt("state_Airfoil0.csv", u_final, delimiter=",")

StateToCSV()

# testFlux()

# Airfoil0()

# Airfoil1()

# Airfoil2()

# freestream()

# plot_cp()

# Forces()
