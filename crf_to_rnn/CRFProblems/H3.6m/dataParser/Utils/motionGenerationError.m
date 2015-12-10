function  mean_error = motionGenerationError( dirname,iter_use )

clrs='rgbkmc';

if nargin < 2
    iter_use = 4000;
end;

legend_to_add = {};
fnum = 1;
figure;
toplot = false;
mean_error = [];
for iteration =   iter_use
    R0 = eye(3);
    T0 = [0 0 0];
    errors = [];
    for N = 0:7
        
        if exist([dirname,'/ground_truth_forecast_N_', num2str(N) ,'.dat'],'file') ~= 2
            continue
        end;
        
        f=csvread([dirname,'/ground_truth_forecast_N_', num2str(N) ,'.dat']);
        fstd = std(f,1);
        idx_to_use = find(fstd>1e-4);

        expchannels = revertCoordinateSpace(f,R0,T0);
        eulerchannels = expchannels;
        for i = 1:size(expchannels,1)
            for j = 4:3:97
               eulerchannels(i,j:j+2) =  RotMat2Euler(expmap2rotmat(expchannels(i,j:j+2)));
            end;
        end;
        eulerchannels(:,1:6) = 0;
        fstd = std(eulerchannels,1);
        idx_to_use = find(fstd>1e-4);

        
        if exist([dirname,'/forecast_iteration_',num2str(iteration),'_N_',num2str(N),'.dat'],'file') ~= 2
            continue
        end;       
        f=csvread([dirname,'/forecast_iteration_',num2str(iteration),'_N_',num2str(N),'.dat']);
        expchannels = revertCoordinateSpace(f,R0,T0);
        eulerchannels_forecast = expchannels;
        for i = 1:size(expchannels,1)
            for j = 4:3:97
               eulerchannels_forecast(i,j:j+2) =  RotMat2Euler(expmap2rotmat(expchannels(i,j:j+2)));
            end;
        end;

        err = (eulerchannels(:,idx_to_use) - eulerchannels_forecast(:,idx_to_use)).^2;
        v=sum(err,2);
        errors(:,N+1) = sqrt(v);
    end;
    if size(errors,1) > 0
        toplot = true;
        mean_error = mean(errors,2);
        legend_to_add{fnum} = ['iteration = ',num2str(iteration)];
        if fnum <= size(clrs,2)
            clr = clrs(fnum);
        else
            clr = rand(1,3);
        end;
        p(fnum) = plot(mean_error,'color',clr,'linewidth',3);
        hold on;
        fnum = fnum + 1;
    end;
end;
if toplot
    l=legend(p,legend_to_add);
    set(l,'FontSize',20)
end
end

