function [W, opts] = rslda(W,Sb,Sw,opts)
%   rslda performs Ratio Sum Linear Discriminant Analysis (RSLDA) using
%   Minorization-Maximization (MM) algorithm and Double-Loop (DL) algorithm
%
%   W:        d x q projection matrix
%   Sb:       d x d between-class sample covariance matrix
%   Sw:       d x d within-class sample covariance matrix
%  
%  opts:      A structure used for performing RSLDA
%             opts.alg: MM or DL algorihtms, including 'DL1', 'MM', 'DL10', 'DL50' 
%	          opts.maxit: maximum number of iterations for MM and the outer loop of DL; default 100
%             opts.subniters: the number of iterations s for the inner loop of DL          
%             opts.errlog:  a vector recording the objective value at each iteration
%	          opts.disp_it = 1 display error values; =0 not display 
%	          opts.tol: threshold (relative change in objective values); default 1e-8
%             opts.logL: final objective value.
%             opts.itnum: number of iterations
%             opts.time: time.preit: the time consumed before the loop; time.it:  a vector recording CPU time consumed at each iteration.

%   Reference: [1] Minorization-Maximization for Ratio Sum Linear Discriminant Analysis. Jianhua Zhao, Xuan Ma, and Guanliang Huang,
%   Technical Report, School of Statistics and Mathematics, Yunnan University of Finance and Economics
%	Copyright (c) Jianhua Zhao (2025)

%TimeComp=; % 1: tic,toc; 0: CPUtime %tstart=cputime;
if opts.TimeComp tic; else tstart=cputime; end
niters = 100; if isfield(opts, 'maxit') niters = opts.maxit; end
disp_it = 0; if isfield(opts, 'disp_it') disp_it = opts.disp_it; end
test = 0;if isfield(opts, 'tol') test = 1;end
store = 0;
if (nargout > 1)
    store = 1; opts.errlog = zeros(1, niters);   opts.time.it = zeros(1, niters);
end
d=size(W,1); Id=eye(d);
switch opts.alg
    case {'MM'}
        alpha=eigs(Sw,1);  % alpha=lambda for MM
    case {'DL1' 'DL10' 'DL50'}
        alpha=trace(Sw)*10^opts.alpha.p; S=alpha*Id-Sw; % set alpha as in RSLDA paper
    otherwise
        error(['Unknown algorithm ', opts.alg]);
end

if opts.TimeComp opts.time.preit=toc; else opts.time.preit=cputime-tstart;end %opts.time.preit = cputime-tstart;

for n = 1:niters
    if opts.TimeComp tic; else tstart=cputime;end %tstart=cputime;
    opts.itnum = n;
    BW=Sb*W; TW=Sw*W;  SW=alpha*W-TW;%dxq
    dWBW=sum(W.*BW,1); dWTW=sum(W.*TW,1); %1xq
    if (disp_it || store || test)
        ek=dWBW./dWTW; 
        e=sum(ek);
        if opts.TimeComp opts.time.it(n)=toc; else opts.time.it(n)=cputime-tstart;end
        if store
            opts.errlog(n) = e;
        end
        if (disp_it > 0)
            if n>1
                fprintf(1, 'Cycle %4d  logL %11.6f, relative increment %e\n', n, e, (e-eold)/abs(eold));
            else
                fprintf(1, 'Cycle %4d  logL %11.6f\n', n, e);
            end
            if (n > 1 && e<eold)
                fprintf('----> Obj value decreased in iteration %4d\n', n);
            end
        end
        if test
            if (n > 1 && abs(e - eold)/abs(e) < opts.tol)
                opts.logL = e; opts.logLk = ek;
                opts.errlog = opts.errlog(1:opts.itnum);
                opts.time.it = opts.time.it(1:opts.itnum);
                return;
            else
                eold = e;
            end
        end
    end
    if opts.TimeComp tic; else tstart=cputime;end 
    D=dWBW.*dWTW.^-2; %1xq
    switch opts.alg
        case {'MM'}
            F=BW.*(dWTW).^-1+SW.*D; % compute F' rather than F
            [U,~,V]=svd(F,"econ");
            W=U*V';
        case {'DL1' 'DL10' 'DL50'}
            for i=1:opts.subniters
                if i>1 BW=Sb*W; dWBW=sum(W.*BW,1); SW=S*W; end
                H=SW.*D+BW.*(D./dWBW).^.5;
                [U,~,V]=svd(H,"econ");
                W=U*V';
            end
        otherwise
            error(['Unknown algorithm ', opts.alg]);
    end
    if opts.TimeComp opts.time.it(n)=opts.time.it(n)+toc;
    else opts.time.it(n)=opts.time.it(n)+cputime-tstart;end
end
opts.logL = e; opts.logLk = ek;
opts.errlog = opts.errlog(1:opts.itnum);
opts.time.it = opts.time.it(1:opts.itnum);
end

