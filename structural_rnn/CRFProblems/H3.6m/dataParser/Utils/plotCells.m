function plotCells( N )
    close all;
    model='checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final/';
    cellname= [model, 'videos/celllong_4000_N_',num2str(N),'.mat'];
    vals = load(cellname);
    
    showbw = 1;
    
    figure;
    %{
    subplot(5,1,1);
    
    m = im2bw(vals.torso);
    f=sum(m,1);
    disp('Torso');
    disp((find(f>140)));
    if showbw
        imshow(im2bw(vals.torso));
    else
        imshow((vals.torso));
    end;
    title('Torso');
    %}
    cell_size = size(vals.left_arm,2);
    T_s = size(vals.left_arm,1);  
    new_val = vals.left_leg - repmat(mean(vals.left_leg,1),T_s,1);
    
    subplot(1,2,1);
    imshow(repmat(new_val(:,436),1,50));
    title('Left leg');
    mval = min(new_val);
    maxval = max(new_val);
    colormap(b2r(mval(436),maxval(436)));      
    %colormap hot;
    
    subplot(1,2,2);
    new_val = vals.right_leg - repmat(mean(vals.right_leg,1),T_s,1);
    imshow(repmat(new_val(:,436),1,50));
    title('Right leg');
    mval = min(new_val);
    maxval = max(new_val);
    colormap(b2r(mval(436),maxval(436)));      
    print('-dpng',['leg_cells_',num2str(N),'.png']);
    
    figure;
    new_val = vals.left_arm - repmat(mean(vals.left_arm,1),T_s,1);
    subplot(1,2,1);
    imshow(repmat(new_val(:,19),1,50));
    title('Left arm');
    mval = min(new_val);
    maxval = max(new_val);
    colormap(b2r(mval(436),maxval(436)));      

    subplot(1,2,2);
    new_val = vals.right_arm - repmat(mean(vals.right_arm,1),T_s,1);
    imshow(repmat(new_val(:,19),1,50));
    title('Right arm');
    mval = min(new_val);
    maxval = max(new_val);
    colormap(b2r(mval(436),maxval(436)));      
    print('-dpng',['arm_cells_',num2str(N),'.png']);
    
    subplot(2,2,1);
    m = im2bw(vals.left_arm);
    f=sum(m,1);
    %disp('Left arm');
    %disp((find(f>140)));  
    if showbw
        imshow(im2bw(vals.left_arm));
    else
        imshow((vals.left_arm));
    end;
    title('Left arm');

    subplot(2,2,2);
    m = im2bw(vals.right_arm);
    f=sum(m,1);
    %disp('Right arm');
    %disp((find(f>140)));
    if showbw
        imshow(im2bw(vals.right_arm));
    else
        imshow((vals.right_arm));
    end;
    title('Right arm');

    subplot(2,2,3);
    m = im2bw(vals.left_leg);
    f=sum(m,1);
    %disp('Left leg');
    %disp((find(f>140)));
    if showbw
        imshow(im2bw(vals.left_leg));
    else
        imshow((vals.left_leg));
    end;
    title('Left leg');

    subplot(2,2,4);
    m = im2bw(vals.right_leg);
    f=sum(m,1);
    %disp('Right leg');
    %disp((find(f>140)));
    if showbw
        imshow(im2bw(vals.right_leg));
    else
        imshow((vals.right_leg));
    end;
    title('Right leg');

    figure;
    
    subplot(2,2,1);
    cell_size = size(vals.left_arm,2);
    T_s = size(vals.left_arm,1);
  
    new_val = vals.left_arm - repmat(mean(vals.left_arm,1),T_s,1);
    plot(mean(new_val,2));
    title('Left arm');
    
    subplot(2,2,2);
    new_val = vals.right_arm - repmat(mean(vals.right_arm,1),T_s,1);
    plot(mean(new_val,2));
    title('Right arm');

    subplot(2,2,3);
    new_val = vals.left_leg - repmat(mean(vals.left_leg,1),T_s,1);
    plot(mean(new_val,2));
    title('Left leg');

    subplot(2,2,4);
    new_val = vals.right_leg - repmat(mean(vals.right_leg,1),T_s,1);
    plot(mean(new_val,2));
    title('Right leg');
    %{
    figure;
    imshow((vals.left_leg(:,[5,98,436,39])));
    print('-dpng','left_leg_cells.png');

    figure;
    imshow(im2bw(vals.right_arm));
    print('-dpng','right_arm_cells.png');
    
    %close all;
    figure;
    pleg = plot(vals.left_leg(:,436),'linewidth',3,'color','r');
    %title('Left leg activation','fontsize',12);
    %xlabel('Time','fontsize',16);
    %ylabel('Cell activation','fontsize',16);
    set(gca,'fontsize',16);
    xlim([0 500]);
    print('-dpng','left_leg.png');

    %close all;
    figure;
    pleg = plot(vals.right_arm(:,19),'linewidth',3,'color','r');
    %title('Right arm activation','fontsize',12);
    %xlabel('Time','fontsize',16);
    %ylabel('Cell activation','fontsize',16);
    set(gca,'fontsize',16);
    xlim([0 500]);
    print('-dpng','right_arm.png');

    
    close all;
    for i =1 : 200 %size(vals.right_arm,2)
        ff = figure;
        gg= plot(vals.right_arm(:,i));
        legend(gg,[num2str(i)]);
        drawnow;
        F_right_arm(i) = getframe;
        close(ff);
    end;
    %}
end

