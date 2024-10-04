%% check_OCP_solution
% this code only allows for constant kappa along the prediction horizon
clc;clear all;close all




p.Q = diag([0 1 1 1]);% s,n,alpha,v
%p.R = diag([1 0]);
p.R = diag([1 1]); % v delta
p.P = p.Q;
p.nx = 4;
p.nu = 2;
p.N_sim = 300;
p.useTerminal = false;
p.alpha = Inf;
p.N = 50;%50;%10
p.xmin = [0 -Inf -Inf 0]';
p.xmax = [Inf Inf Inf 1]';
p.umin = [0 -0.5]';%-Inf;
p.umax = [1 0.5]';%Inf;
v_ref = 0.6;
v_com_ref = 0.6;
delta_ref = 0; % not used anymore in the code
p.x_ref=[0 0 0 v_ref]';
p.modelparameter = 1;
p.u_ref = [v_com_ref atan(0.25*p.modelparameter)]';
p.np_model = 1;

p.np_plant = 1;% mumber plant parameter
p.plantparameter = p.modelparameter;% no model-plant-mismatch 0;%plantparameter
x = casadi.SX.sym('x',[p.nx 1]);% #states
u = casadi.SX.sym('u',[p.nu 1]);% #controls
modelparameter = casadi.SX.sym('modelparameter',[p.np_model 1]);%
%x_plus_nom = nominal_dynamics(x,u,modelparameter);
x_plus_nom  = nominal_dynamics_no_beta(x,u,modelparameter);
f_nom = casadi.Function('f_nom',{x,u,modelparameter},{x_plus_nom},{'x','u','p'},{'x_next_nom'});



mpc = Standard_MPC(f_nom,f_nom,p.nx,p.nu,p.N,p.N_sim,p);
% you have to create_OCP for most of the things first
mpc.create_OCP(@StageCost,@TerminalCost);
xinit = [0 0 0 0.6]';
sim_para.u_error = zeros(1,p.N_sim);
mpc = mpc.closedLoopSimulation(xinit,sim_para,1);
mpc.plot_closed_loop()


%%%% open loop simulation with precompensation
x_ol = open_loop_simulation(xinit,[v_com_ref atan(0.25*p.modelparameter)]',p);
% figure(6)
% hold on
% plot()
for k = 1:4
    figure(k)
    hold on
    plot(x_ol(k,:),'r-')
    legend('closed loop mpc','open loop')
end

kappa_list = -2:0.1:2;
u_opt_1= [];
iter_planned = 1;
for k =1:length(kappa_list)
    p.modelparameter = kappa_list(k);
    p.plantparameter = p.modelparameter;
    p.u_ref = [v_com_ref atan(0.25*p.modelparameter)]';
    %x_plus_nom = nominal_dynamics(x,u,modelparameter);
    x_plus_nom  = nominal_dynamics_no_beta(x,u,modelparameter);
    f_nom = casadi.Function('f_nom',{x,u,modelparameter},{x_plus_nom},{'x','u','p'},{'x_next_nom'});
    mpc = Standard_MPC(f_nom,f_nom,p.nx,p.nu,p.N,p.N_sim,p);

    %%%%% use precompensation as reference input
    % delta_ref = 0.25*kappa_list(k);
    % p.u_ref = [v_com_ref delta_ref]';
    % p.R = diag([1 1]);
    %%%%%%%%%%%
    mpc.create_OCP(@StageCost,@TerminalCost);
    % xinit = X_samples(:,k);
    %xinit = [0 0.01 0 0.3]';
    initial_guess_X = repmat(xinit,[1,p.N+1]);
    initial_guess_U = repmat([0.6;atan(0.25*kappa_list(k))],[1,p.N]);%zeros(p.nu,p.N);
    [stats,ocp_sol]=mpc.solveOCP(xinit,initial_guess_X,initial_guess_U);
    %ocp_sol
    u_opt_1 = [u_opt_1, ocp_sol.U_ocp(2,iter_planned)];
    % mpc = mpc.closedLoopSimulation(xinit,sim_para,1);
    % mpc.plot_closed_loop()

end % OCp for differen kappa, n=alpha=0
figure()
hold on
plot(kappa_list,u_opt_1,'kx-')
plot(kappa_list,atan(0.25.*kappa_list),'r-')
xlabel('kappa')
ylabel('steering')
title('uopt at iter planned')
legend('ocp','precompensation')

%% impact of inital velocities

p.modelparameter = 1;
p.plantparameter = p.modelparameter;
%x_plus_nom = nominal_dynamics(x,u,modelparameter);
x_plus_nom  = nominal_dynamics_no_beta(x,u,modelparameter);
f_nom = casadi.Function('f_nom',{x,u,modelparameter},{x_plus_nom},{'x','u','p'},{'x_next_nom'});
mpc = Standard_MPC(f_nom,f_nom,p.nx,p.nu,p.N,p.N_sim,p);

mpc.create_OCP(@StageCost,@TerminalCost);
% xinit = X_samples(:,k);

xinit = [0 0.0 0 0.3]';
initial_guess_X = repmat(xinit,[1,p.N+1]);
initial_guess_U = repmat([0.6;atan(0.25*kappa_list(k))],[1,p.N]);%zeros(p.nu,p.N);
[stats,ocp_sol]=mpc.solveOCP(xinit,initial_guess_X,initial_guess_U);
ocp_sol.U_ocp(:,1)

vinit_list = 0:0.05:1;
steer_vtest  = [];
v_vtest = [];
for k =1:length(vinit_list)
    % if k == 13
    %     brkpnt = 1;
    % end
%xinit = [0 0.5 0 vinit_list(k)]';
xinit = [0 0 0 vinit_list(k)]';
initial_guess_X = repmat(xinit,[1,p.N+1]);
initial_guess_U = repmat([0.6;atan(0.25*p.modelparameter)],[1,p.N]);%zeros(p.nu,p.N);
[stats,ocp_sol]=mpc.solveOCP(xinit,initial_guess_X,initial_guess_U);
v_vtest = [ v_vtest ocp_sol.U_ocp(1,1)];
steer_vtest = [ steer_vtest ocp_sol.U_ocp(2,1)];

end
figure()
hold on
plot(vinit_list,steer_vtest,'kx-')

xlabel('v')
ylabel('steering')
title('n=alpha=0,kappa=1,vref=0.6')

figure()
hold on
plot(vinit_list,v_vtest,'kx-')

xlabel('v')
ylabel('v_com')
title('n=alpha=0,kappa=1,vref=0.6')

function x_next = nominal_dynamics(x,u,p)
kappa = p;%modelparameter;
l_r = 0.125;
l_f = 0.125;
T = 0.00984;
T_s = 0.1;%1/60;%0.1;%1/60; % sampling time; simple euler integration
s = x(1);
n = x(2);
alpha = x(3);
v = x(4);
%x_vec = [s,n,alpha,v]';
v_com = u(1);
delta = u(2);
beta = 0.5*delta;%atan(0.5*tan(delta));%0.5*delta;

s_dot = v*cos(alpha+beta)/(1-n*kappa);
n_dot = v*sin(alpha+beta);
alpha_dot = (v*sin(beta)/l_r) - kappa*s_dot;
v_dot = (v_com-v)/T;%v_com;%(v_com-v)/T;%(v_com-v)/T;

x_dot = [s_dot,n_dot,alpha_dot,v_dot]';
x_next = x+T_s*x_dot;
%x_next = x_vec+T_s*x_dot;

end

function x_next = nominal_dynamics_no_beta(x,u,p)
kappa = p;%modelparameter;
l_r = 0.125;
l_f = 0.125;
T = 0.00984;
T_s = 0.1;%1/60;%0.1;%1/60; % sampling time; simple euler integration
s = x(1);
n = x(2);
alpha = x(3);
v = x(4);
%x_vec = [s,n,alpha,v]';
v_com = u(1);
delta = u(2);
%beta = 0.5*delta;%atan(0.5*tan(delta));%0.5*delta;

s_dot = v*cos(alpha)/(1-n*kappa);
n_dot = v*sin(alpha);
alpha_dot = v*(tan(delta)/(l_r+l_f)) - kappa*s_dot;
v_dot = (v_com-v)/T;%v_com;%(v_com-v)/T;%(v_com-v)/T;

x_dot = [s_dot,n_dot,alpha_dot,v_dot]';
x_next = x+T_s*x_dot;
%x_next = x_vec+T_s*x_dot;

end

function c = StageCost(x,u,p)
x_ref=p.x_ref;
u_ref = p.u_ref;
x_vec = [x(1) x(2) x(3) x(4)]';
u_vec = [u(1) u(2)]';
Q = p.Q;%diag([1,1]);
R = p.R;%1;
c = (x_vec-x_ref)'*Q*(x_vec-x_ref)+(u_vec-u_ref)'*R*(u_vec-u_ref);
end
function c = TerminalCost(x,p)
%P = diag([1,1]);
% P = 1.0e+03.*[2.0132    0.8561;
%     0.8561    0.6201];
x_ref=p.x_ref;
x_vec = [x(1) x(2) x(3) x(4)]';
P = p.P;
c = (x_vec-x_ref)'*P*(x_vec-x_ref);
end


function x_ol = open_loop_simulation(x,u,p)
x_ol = x;
for k = 1:p.N_sim
    x_ol(:,k+1) = nominal_dynamics_no_beta(x_ol(:,k),u,p.modelparameter);
end

end

