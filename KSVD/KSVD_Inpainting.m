function [Dict Coeff]=KSVD_Inpainting(Dict,blkMatrixIm,blkMask,sigma,rc_min,max_coeff,max_iter)
% Author: Chongyang
% Date: 2016/06/05
% KSVD for Inpainting  
%
% sigma: residual error stopping criterion, normalized by signal norm  0.01
% rc_min: minimal residual correlation before stopping 0.01
% max_coeff: maximal number of non-zero coefficients for signal 10t


fprintf('KSVD begin .....\n')
tic
for KSVDiter=1:max_iter
    fprintf('iteration: %d/%d | time: %d \n',KSVDiter,max_iter,toc)
 	Coeff=OMP_Inpainting(Dict,blkMatrixIm.*blkMask,blkMask,sigma,rc_min,max_coeff); %256*146405
    for atom=1:1:size(Dict,2) % 256依次更新在字典中的所有列
        Omega=find(abs(Coeff(atom,:))>0); %atom行等于零的X项可以略去
        if isempty(Omega) 
            continue; 
        end;
        CoeffM=Coeff; 
        CoeffM(atom,:)=0; %等待更新的表达矩阵
        Err=blkMask(:,Omega).*(blkMatrixIm(:,Omega)-Dict*CoeffM(:,Omega));                 %计算误差 Err为误差矩阵
        
        dd=diag(1./(blkMask(:,Omega)*(Coeff(atom,Omega).^2)'))*(Err*Coeff(atom,Omega)');   %得到新的一行
        Dict(:,atom)=dd/norm(dd);    %  标准化使其2范数为1
        Coeff(atom,Omega)=(dd'*Err)./sum((blkMask(:,Omega).*(dd*ones(1,length(Omega)))).^2,1);  %复制列

    end;
%     save Dict Dict;
%     save Coeff Coeff;
end;