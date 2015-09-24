clear all;
clc;
%close all;
names = dir;
fnum = 1;
clrs='rgbkmc';
fg = figure;
legend_to_add = {}  ;  
CM = jet(10);
folders = uigetdir2;
for i = 1:size(folders,2) %numel(names)
    %nm = names(i);
    nm = strsplit(folders{i},'/');
    nm.name = nm{end};
    %if nm.isdir && ~strcmp(nm.name , '.') && ~strcmp(nm.name , '..')
        
        disp(nm.name)
        substring = strsplit(nm.name,'_');
        
        x = strfind(substring,'clipnorm');
        for j = 1:size(x,2)
            if x{j} == 1
                clipnorm = substring{j+1};
                break;
            end;
        end;
        x = strfind(substring,'noise');
        for j = 1:size(x,2)
            if x{j} == 1
                noise = substring{j+2};
                break;
            end;
        end;

        x = strfind(substring,'nschd');
        for j = 1:size(x,2)
            if x{j} == 1
                noise = substring{j+1};
                break;
            end;
        end;
        
        decayrate=''
        x = strfind(substring,'decay');
        for j = 1:size(x,2)
            if x{j} == 1
                decayrate = substring{j+2};
                break;
            end;
        end;

        x = strfind(substring,'decschd');
        for j = 1:size(x,2)
            if x{j} == 1
                decayrate = substring{j+1};
                break;
            end;
        end;
        
        x = strfind(substring,'size');
        for j = 1:size(x,2)
            if x{j} == 1
                batch_size = substring{j+1};
                break;
            end;
        end;
        
        x = strfind(substring,'bs');
        for j = 1:size(x,2)
            if x{j} == 1
                batch_size = substring{j+1};
                break;
            end;
        end;
        
        T = str2num(substring{4});
        
        if T ~= 150 %||  str2num(clipnorm) ~= 5
            continue
        end;
        
        
       
        
        logfile = [nm.name, '/logfile.dat'];
        fid = fopen(logfile);
        tline = fgets(fid);
        iter = [];
        loss = [];
        validation = [];
        count = 1;
        while ischar(tline)
            tline = deblank(tline);
            vals = strsplit(tline,',');
            if numel(vals) == 1
                loss(count) = str2num(vals{1});
            elseif numel(vals) == 2
                loss(count) = str2num(vals{1});
                validation(count) = str2num(vals{2});
            end;
            count = count + 1;
            tline = fgets(fid);
        end  
        if fnum <= size(clrs,2)
            clr = clrs(fnum);
        else
            clr = rand(1,3);
        end;
        loss = loss*1.0/T;
        validation = validation*1.0/T;
        loss = interp1(1:numel(loss),loss,1:numel(loss));
        p(fnum) = plot(loss,'color',clr,'linewidth',3);
        legend_to_add{fnum} = ['T=',num2str(T),' bs=',batch_size, ' cnm=',clipnorm,' noi=',noise,' lr=',decayrate];
        hold on;
        plot(find(validation>0),(validation(find(validation>0))),'color',clr,'linewidth',3,'linestyle','--');
        fnum = fnum + 1;
        hold on;
    %end;
    
end;
l=legend(p,legend_to_add);
set(l,'FontSize',20)

disp('asda')
