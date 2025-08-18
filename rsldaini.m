function [mo,data,opts]=rsldaini(data, subdim, opts)

if ~isfield(opts, 'TimeComp') opts.TimeComp=1; end
% 1: tic,toc; 0: CPUtime %tstart=cputime;
if opts.TimeComp tic; else tstart=cputime; end
x=data.X; x_label=data.label; [d, ndata1] = size(data.X);
mo.model=opts.model;
mo.q = min(subdim,d); q=mo.q;
if (~exist('opts', 'var')) opts = []; end
if ~isfield(opts, 'center') opts.center = 1; end
if ~isfield(opts, 'alg') opts.alg = 'MM-lambda'; end
if ~isfield(opts, 'ini') opts.ini.md = 'rand'; end
if ~isfield(opts, 'tol') opts.tol = 1e-8; end
if ~isfield(opts, 'regv') opts.regv=0.01; end
if strncmp(opts.alg,'DL',2) % DL1 by default
    if ~isfield(opts, 'subniters') opts.subniters = 10; end
    if ~isfield(opts.alpha, 'p') opts.alpha.p = 1; end
end
cls=unique(x_label); ncls=length(cls);
ni=sum(x_label==cls',2)';nis=ni.^.5;
%if opts.center w=mean(x,2); x=x-w; end
switch mo.model
    case 'rslda'
        mo.d = d;
        if opts.center w=mean(x,2); x=x-w; end
        proj='norm';
        if isfield(opts, 'proj') proj = opts.proj; end
        mi=zeros([d ncls]); x_mi=x;
        for i=1:ncls
            ind=(x_label==cls(i));
            xi=x(:, ind);
            mi(:, i)=mean(xi, 2);
            x_mi(:,ind)=xi-mi(:,i);
        end
        Hb=mi.*nis/ndata1^.5;
        data.B=Hb*Hb';
        data.T=x_mi*x_mi'/ndata1;
        data.T=data.T + opts.regv*trace(data.T)/d*eye(d);
        %data.T=x_mi*x_mi'/ndata1+1*eye(d);
    case 'krslda'
        mo.d = ndata1; xt=x';
        opts2=[];
        if ~isfield(opts, 'kernel')
            D=EuDist2(xt,[],0); Dm=mean(D(:));
            opts.kernel.t=sqrt(Dm/8); % tnnls use
            %  opts.kernel.t=sqrt(6); %
            % opts.kernel.t=sqrt(Dm/100);
        end
        opts2.t=opts.kernel.t;
        K = constructKernel(xt,[],opts2);

        % ====== Initialization
        % nSmp = size(K,1);
        % classLabel = unique(gnd);
        % nClass = length(classLabel);
        % Dim = nClass - 1;

        %K_orig = K;

        sumK = sum(K,2);
        H = repmat(sumK./ndata1,1,ndata1);
        K = K - H - H' + sum(sumK)/(ndata1^2);
        K = max(K,K');
        clear H;

        Hb = zeros(ncls,ndata1);
        for i = 1:ncls
            index = (x_label==cls(i));
            classMean = mean(K(index,:),1);
            %Hb (i,:) = sqrt(length(index))*classMean;
            Hb (i,:) = nis(i)*classMean;
        end
        data.B = Hb'*Hb/ndata1;
        data.T = K*K/ndata1;
        data.T=data.T + opts.regv*trace(data.T)/ndata1*eye(ndata1);
        data.B = max(data.B,data.B');
        data.T = max(data.T,data.T');
        d=ndata1;
        % for i=1:size(data.T,1)
        %     T(i,i) = T(i,i) + options.ReguAlpha;
        % end

        % B = double(B);
        % T = double(T);
end

switch opts.ini.md
    case 'rand'
        rng(opts.ini.seed)
        mo.W=randn(d,q); mo.W=orth(mo.W);
    case 'lda-orth'
        opts.q=q; opts.proj = 'orth';
        U=ldaU2(data.T, data.B, opts); mo.W=U{1};
    case 'trlda'
        opts1=opts; opts1.q=q; opts1.alg='MM-ml';
        rng(1); mo1=mo;
        mo1.W=randn(d,q); mo1.W=orth(mo1.W);
        [mo1, data, opts1] = trlda(mo1, data, opts1);
        mo.W=mo1.W;
    case 'lda-U'
        opts.q=q; opts.proj = 'unit';
        U=ldaU2(data.T, data.B, opts); mo.W=U{1};
    otherwise
        error(['Unknown algorithm initialization ', opts.ini]);
end
if opts.TimeComp opts.time.ini=toc; else opts.time.ini = cputime-tstart; end %opts.time.ini=cputime-tstart;

%S=x*x'/ndata1;

% Hb=comphb(x,x_label,nis);
% if d<=ndata1
%     S=x*x'/ndata1; [lmds,Us]=eigdec(S,d);    mlmd=sum(lmds)/d;
%     Us_tHb=Us'*Hb;
% else
%     T=x'*x/ndata1; [lmdt,Ut]=eigdec(T,ndata1);  mlmd=sum(lmdt)/d;
%     xUt=x*Ut; Ut_thb=comphb(Ut',x_label,nis);
% end