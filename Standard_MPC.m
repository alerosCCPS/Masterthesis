classdef Standard_MPC
    %% todo
    % nonlinear constraints + can change over horizon (eg p.xmin(i))
    % stage cost which can change along the horizon; maybe by defining
    % externally stagecost(x,u,i)
    % keep track of comp time    done:varying parameter in model and plant

    properties
        F_model
        F_plant
        Ts
        N
        N_sim
        nx
        nu
        np
        X
        U
        X_ocp
        U_ocp
        X_ocp_all
        U_ocp_all
        X_cl
        U_cl
        opti
        solverstats
        p
    end % properties
    methods
        function obj = Standard_MPC(F_model, F_plant,nx,nu, N, N_sim,p)
            obj.F_model = F_model;
            obj.F_plant = F_plant;
            %obj.Ts = Ts;
            obj.N = N;
            obj.N_sim = N_sim;
            obj.nx = nx; %size(F_model, 1);
            obj.nu = nu;%size(F_model, 2);
            %obj.np = size(F_model, 3);
            obj.solverstats = [];
            obj.opti = casadi.Opti();
            obj.X = obj.opti.variable(obj.nx,obj.N+1);
            obj.U = obj.opti.variable(obj.nu,obj.N);
            obj.p = p;
        end% fct StandardMPC Obj

        function obj = create_OCP(obj,StageCost,TerminalCost)
            J = 0;
            for k = 1:obj.N
                J = J + StageCost(obj.X(:,k),obj.U(:,k),obj.p);

            end% stage cost
            J = J + TerminalCost(obj.X(:,k+1),obj.p);
            obj.opti.minimize(J);
            % multiple shooting constr.
            for k = 1:obj.N
                x_next = obj.F_model(obj.X(:,k), obj.U(:,k),obj.p.modelparameter);
                obj.opti.subject_to(obj.X(:,k+1) == x_next)
            end % multiple shooting

            % for now simple box constraints..later nonlinear constraints
            % that can change over the horitzon, you can just make eg p.umax a
            % vector if you have no fancy nonlinear constraints
            for k = 1:obj.N
                obj.opti.subject_to(obj.p.umin <= obj.U(:,k) <= obj.p.umax);  
                obj.opti.subject_to(obj.p.xmin <= obj.X(:,k) <= obj.p.xmax);
            end % constraints
            obj.opti.subject_to(obj.p.xmin <= obj.X(:,obj.N+1) <= obj.p.xmax);
            % terminal Constraint
            if obj.p.useTerminal
                obj.opti.subject_to(0 <=  obj.X(:,obj.N+1)'*obj.p.P*obj.X(:,obj.N+1) <= obj.p.alpha )
            end
            %https://www.coin-or.org/Bonmin/option_pages/options_list_ipopt.html
            solver_opts = struct;
            solver_opts.ipopt.print_level = 0; % Set IPopt print level to suppress output
            %solver_opts.ipopt.print_timing_statistics = 'no';
            %solver_opts.ipopt.inf_pr_output = 'no';
            obj.opti.solver('ipopt',solver_opts)



        end % create ocp
        function [stats,obj] = solveOCP(obj,x_init,initial_guess_X,initial_guess_U)
            opti_now = obj.opti.copy(); % otherwise it leads to problems when repeateadly setting intial guesses
            % IC
            
            opti_now.subject_to(obj.X(:,1)==x_init)
            

            opti_now.set_initial(obj.X,initial_guess_X)
            opti_now.set_initial(obj.U,initial_guess_U)
            sol = opti_now.solve();
            obj.X_ocp = sol.value(obj.X);
            obj.U_ocp = sol.value(obj.U);
            stats = sol.stats();

        end % solve ocp

        function obj = closedLoopSimulation(obj,x_init,sim_para,update_guess)
            % get initial guess, can be relevant for more complex problems
            initial_guess_X = repmat(x_init,[1,obj.N+1]);
            initial_guess_U = zeros(obj.nu,obj.N);

            %[~,obj] = solveOCP(obj,x_init,zeros(obj.nx,obj.N+1),zeros(obj.nu,obj.N));
            
            [~,obj] = solveOCP(obj,x_init,initial_guess_X,initial_guess_U);
            
            x_init_now = x_init;
            
            
            % iterate and store results;
            obj.X_cl(:,1) = x_init; % for storing stuff
            for k = 1:obj.N_sim
                % if k == 49
                %     brkpnt = 1
                % end
                if update_guess
                    [stats,obj] = solveOCP(obj,x_init_now,obj.X_ocp,obj.U_ocp);
                else
                   [stats,obj] = solveOCP(obj,x_init_now,zeros(obj.nx,obj.N+1),zeros(obj.nu,obj.N));
                end
                % store results, so that we can make a detailed analysis
                % later
                obj.solverstats{k} = stats;
                obj.X_ocp_all{k} = obj.X_ocp;
                obj.U_ocp_all{k} = obj.U_ocp;
                obj.U_cl(:,k) = obj.U_ocp(:,1);
                
                %check feasibility
                if ~strcmp(obj.solverstats{k}.return_status, 'Solve_Succeeded') % mpc.solverstats{k}.return_status ~= 'Solve_Suceeded'
                    warning('Problem may be infeasible')
                end
                % simulate plant
                u_error = sim_para.u_error(k);
                obj.X_cl(:,k+1) = full(obj.F_plant(obj.X_cl(:,k), obj.U_cl(:,k) + u_error,obj.p.plantparameter));

                %set as new initial condition
                x_init_now = obj.X_cl(:,k+1);

            end % cl sim



        end % closed loop simulation
        
        function plot_closed_loop(obj)
            xsim = obj.X_cl';%';
            uopt = obj.U_cl; %maybe u is still wrong when plotting multi dim u

            for k = 1:obj.nx+obj.nu
                figure;
                hold on
                if k <= obj.nx
                    plot(xsim(:, k));
                    xlabel('timeIndex');
                    ylabel(['x_', num2str(k)]);
                else
                    plot(uopt(k-obj.nx, :));
                    xlabel('timeIndex');
                    ylabel(['u_', num2str(k-obj.nx)]);

                end % if
            end % for all x and u


        end % function plot_closed_loop

        function Elli = sample_ellipse(obj,P,alpha,sampling_method,num_points)
            P_inv = inv(P/alpha);
            n = size(P, 1);
            L = chol(P_inv, 'lower');
            ellipsoid_points = zeros(num_points, n);

            if strcmp(sampling_method, 'even')
                theta_values = linspace(0, 2*pi, num_points);

                for i = 1:num_points
                    unit_sphere_point = [cos(theta_values(i)), sin(theta_values(i)), zeros(1, n-2)]; %% this is probably not correct...only 2 dimsnsional!!
                    ellipsoid_points(i, :) = L * unit_sphere_point';
                end
            elseif strcmp(sampling_method, 'random')
                for i = 1:num_points
                    unit_sphere_point = randn(n, 1);
                    unit_sphere_point = unit_sphere_point / norm(unit_sphere_point);
                    ellipsoid_points(i, :) = L * unit_sphere_point;
                end
            else
                error('Invalid sampling method. Use ''even'' for evenly spaced or ''random'' for random sampling.');
            end
            warning('currently only tested for 2 dimensions')
            Elli = ellipsoid_points;

        end % sample ellipse


    end % methods

end % class