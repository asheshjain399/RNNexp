addpaths;
model='checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final/';
for i = 0:23
    T = 501;
    
    fname= [model, 'forecast_celllong_4000_N_',num2str(i),'.dat'];
    val=dlmread(fname,',');
    right_arm=val(1:T,:);
    torso=val(T+1:2*T,:);
    right_leg=val(2*T+1:3*T,:);
    left_leg=val(3*T+1:4*T,:);
    left_arm=val(4*T+1:5*T,:);
    fname= [model, 'videos/celllong_4000_N_',num2str(i),'.mat'];
    save(fname,'right_arm','left_arm','left_leg','torso','right_leg');
end;