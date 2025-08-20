function [W,Sb,Sw,opts]=rsldaini(x,x_label,subdim, opts)
%   rsldaini perform initialization of RSLDA parameters
%
%   x:        d x N data matrix
%   x_label:  1 x N label vector of data x
%   subdim:   required latent dimension q

%   W:        d x q projection matrix
%   Sb:       d x d between-class sample covariance matrix
%   Sw:       d x d within-class sample covariance matrix
%  
%  opts:      A structure used for performing RSLDA
%             opts.center: =1 center data x; =0 not center
%             opts.alg: MM or DL algorihtms, including 'DL1', 'MM', 'DL10', 'DL50'; default MM 
%             opts.time: time.ini: time consumed by initialization
%             opts.ini: ini.md: initialization method, default rand
%	          opts.tol: threshold (relative change in objective values); default 1e-8
%             opts.subniters: number of iterations s for inner loop of DL          
%             opts.alpha: auxiliary parameter required by DL

if ~isfield(opts, 'TimeComp') opts.TimeComp=1; end
% 1: tic,toc; 0: CPUtime %tstart=cputime;
if opts.TimeComp tic; else tstart=cputime; end
[d, ndata1] = size(x);
q = min(subdim,d);
if (~exist('opts', 'var')) opts = []; end
if ~isfield(opts, 'center') opts.center = 1; end
if ~isfield(opts, 'alg') opts.alg = 'MM'; end
if ~isfield(opts, 'ini') opts.ini.md = 'rand'; end
if ~isfield(opts, 'tol') opts.tol = 1e-8; end
if strncmp(opts.alg,'DL',2) % DL1 by default
    if ~isfield(opts, 'subniters') opts.subniters = 10; end
    if ~isfield(opts.alpha, 'p') opts.alpha.p = 1; end
end
cls=unique(x_label); ncls=length(cls);
ni=sum(x_label==cls',2)';nis=ni.^.5;

if opts.center w=mean(x,2); x=x-w; end
mi=zeros([d ncls]); x_mi=x;
for i=1:ncls
    ind=(x_label==cls(i));
    xi=x(:, ind);
    mi(:, i)=mean(xi, 2);
    x_mi(:,ind)=xi-mi(:,i);
end
Hb=mi.*nis/ndata1^.5;
Sb=Hb*Hb';
Sw=x_mi*x_mi'/ndata1;

% random initialization of W
switch opts.ini.md
    case 'rand'
        rng(opts.ini.seed)
        W=randn(d,q); W=orth(W);
end

if opts.TimeComp opts.time.ini=toc; else opts.time.ini = cputime-tstart; end %opts.time.ini=cputime-tstart;

