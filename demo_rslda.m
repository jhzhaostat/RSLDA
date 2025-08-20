%profile on;
clear all; clf; 
modelname={'rslda' };
Plot1=1; 
r=1; 
alg={'DL1' 'MM' 'DL10' }; % 'DL1', 'MM', 'DL10', 'DL50'
ini.md={'rand'}; 
subdim=50; % latent dimension q

% data generation
L=10; d=100; N=1000;
label_org=kron(1:L,ones(1,N/L)); 
Ni=N/L*ones(1,L); cNi=cumsum([0 Ni]);
x_org=zeros(d,N); c=0.4;
for i=1:L
    x_org(:,cNi(i)+1:cNi(i+1))=randn(d,Ni(i))+c;
end
Ni2=round(Ni/4);

% split data x_org into training x and test y
rng(r); cNi=cumsum([0 Ni]); index1=[]; index2=[];
for l=1:L
    ind=randperm(Ni(l)); ind1=ind(1:Ni2(l));
    ind2=1:Ni(l); ind2(ind1)=[]; index1=[index1 cNi(l)+ind1]; index2=[index2 cNi(l)+ind2];
end

x=x_org(:, index1); x_label=label_org(index1); nx=size(x, 2);
y=x_org(:, index2); y_label=label_org(index2); ny=size(y, 2);


disp_ini=1; disp_it=1;

if disp_ini
    fprintf('    data: %d points in %d dimensions\n', N, d);
    fprintf('    latent dimension: %d \n', subdim);
    fprintf('--> Training RSLDA by different algorithms:\n');
end

for i=1:length(alg)
    if ~disp_it fprintf(['\n--------->%6s:'], alg{i});
    else fprintf(['\n--------->%6s:\n'], alg{i}); end
    opts=[]; opts.alg=alg{i}; opts.maxit=100; opts.disp_it=disp_it;
    opts.tol = 1e-8; 
    opts.model=modelname{1};
    switch opts.alg
        case 'DL1'
            opts.subniters = 10; opts.alpha.p=1; opts.maxit=opts.maxit/10;
        case 'DL10'
            opts.subniters = 10; opts.alpha.p=1;
        case 'DL50'
            opts.subniters = 50; opts.alpha.p=2;
    end
    opts.clsf='NN';
    opts.ini.md=ini.md{1}; opts.ini.seed=1;
    [W,Sb,Sw,opts]=rsldaini(x,x_label, subdim, opts);
    [W,opts] = rslda(W,Sb,Sw, opts);
    T{i}=opts.time.ini+opts.time.preit+cumsum(opts.time.it);
    iternum(i)=opts.itnum;
    logL(i)=opts.logL;
    ll{i}=opts.errlog;

    if disp_ini fprintf('\n\t\t CPU time\t\t Iterations\t\t\t logL\n'); end
    fprintf('%15.3f', T{i}(end)); fprintf('%16d', iternum(i)); fprintf('%22.5f', logL(i));
end
fprintf('\n');
%% plot 

if Plot1
    figure(1);
    cr={'k--','b-','m:','g-.'};
    % loglike
    subplot(2,2,1)
    g=zeros(1,length(alg));
    for i=1:length(alg)
        g(i)=plot(ll{i},cr{i},'linewidth',1.5);hold on;
    end
    legend(g, alg,'location', 'best');
    %title('The evolvement of objective values')
    xlabel('Iteration number');
    ylabel('Objective value');

    % CPU time
    subplot(2,2,2)
    for i=1:length(alg)
        g(i)=plot(T{i}(1:iternum(i)),ll{i},cr{i},'linewidth',1.5);hold on
    end
    legend(g, alg,'location', 'best');
    %title('The evolvement of objective values')
    xlabel('CPU time');
    ylabel('Objective value');
    hold off;
end

% profile viewer;
% profile off;
