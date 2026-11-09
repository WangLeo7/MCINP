"""Unified discrete operator: all dimensions use u_t = -A u and interaction K, with no dimension-dependent sign."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def symbols(n, alpha, equation):
    if n % 2 or not 1 <= alpha <= 2: raise ValueError('even N and alpha in [1,2] required')
    if equation != 'signed': raise ValueError('the public release supports the signed directional equation only')
    k=np.fft.fftfreq(n)*n
    active=(k!=0)&(abs(k)!=n/2)
    # At alpha = 1, the signed right multiplier equals sign(k); avoid evaluating 0**0 at the zero mode.
    left=np.zeros(n); right=np.zeros(n)
    sign=np.sign(k[active])
    left[active]=sign*abs(k[active]); right[active]=sign*abs(k[active])**(alpha-1)
    return left,right

def action(u, ds, alphas, equation, adjoint=False):
    n=u.shape[0]; dim=u.ndim; axes=tuple(range(dim)); h=np.fft.fftn(u); out=np.zeros_like(h)
    for j,(d,a) in enumerate(zip(ds,alphas)):
        l,r=symbols(n,float(a),equation); shape=[1]*dim; shape[j]=n
        l=l.reshape(shape); r=r.reshape(shape)
        if adjoint: l,r=r,l
        out+=l*np.fft.fftn(d*np.fft.ifftn(r*h))
    value=np.fft.ifftn(out)
    if not np.iscomplexobj(u):
        if np.max(abs(value.imag))>1e-10*max(1,np.max(abs(value.real))): raise ValueError('real-output gate')
        return value.real
    return value

def operator(ds, alphas, equation):
    shape=ds[0].shape; dim=len(shape); n=shape[0]; size=n**dim
    # Expose only matvec and rmatvec; do not form a size-by-size dense matrix for N64 or N128.
    A=LinearOperator((size,size),matvec=lambda v:action(np.asarray(v).reshape(shape),ds,alphas,equation).ravel(),
        rmatvec=lambda v:action(np.asarray(v).reshape(shape),ds,alphas,equation,True).ravel(),dtype=np.float64)
    trace=sum(float(d.mean())*np.sum(np.prod(symbols(n,float(a),equation),axis=0))*n**(dim-1) for d,a in zip(ds,alphas))
    return A,float(trace)

def flow(u,ds,alphas,t,equation,method='exponential',tol=2e-11):
    if t==0:return u.copy()
    A,tr=operator(ds,alphas,equation)
    if method=='exponential': return expm_multiply(-t*A,u.ravel(),traceA=-t*tr).reshape(u.shape)
    if method=='dop853':
        # Independent time discretization with a matrix-free right-hand side. Stiff cases may be expensive; self-comparison is not an independent check.
        z=solve_ivp(lambda _,v:-(A@v),(0,float(t)),u.ravel(),method='DOP853',rtol=tol,atol=tol*1e-2)
        if not z.success:raise RuntimeError(z.message)
        return z.y[:,-1].reshape(u.shape)
    if method=='dense':
        if u.ndim!=1:raise ValueError('dense cross-check restricted to 1D')
        # Build an independent Fourier-convolution matrix rather than sampling the production matvec column by column.
        n=len(u); l,r=symbols(n,float(alphas[0]),equation); dh=np.fft.fft(ds[0])/n
        M=l[:,None]*dh[(np.arange(n)[:,None]-np.arange(n)[None,:])%n]*r[None,:]
        return np.fft.ifft(expm(-t*M)@np.fft.fft(u)).real
    raise ValueError(method)

def trajectory(u,ds,alphas,times,equation,switch=None,alphas_after=None,method='exponential',tol=2e-11):
    # For every post-switch query, restart from the state at the switch time rather than from u0.
    switched=None
    if switch is not None: switched=flow(u,ds,alphas,switch,equation,method,tol)
    return np.stack([flow(u,ds,alphas,t,equation,method,tol) if switch is None or t<=switch
        else flow(switched,ds,alphas_after,t-switch,equation,method,tol) for t in times])

def relative(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))

def qualify(u,ds,alphas,times,equation,cap=1e-8,switch=None,alphas_after=None):
    if equation!='signed':raise ValueError('author-selected production family is directional signed')
    if not np.isfinite(u).all() or not np.isfinite(ds).all() or np.min(ds)<=0:raise ValueError('finite/positive coefficient gate')
    kw=dict(switch=switch,alphas_after=alphas_after)
    truth=trajectory(u,ds,alphas,times,equation,**kw)
    independent=trajectory(u,ds,alphas,times,equation,method='dense' if u.ndim==1 else 'dop853',**kw)
    refined=trajectory(u,ds,alphas,times,equation,method='dop853',tol=2e-12,**kw)
    pos=np.asarray(times)>0
    errors=[relative(x,y) for x,y in zip(truth[pos],independent[pos])]
    refinement=[relative(x,y) for x,y in zip(independent[pos],refined[pos])]
    rms=[float(np.sqrt(np.mean((x-y)**2))) for x,y in zip(truth[pos],independent[pos])]
    norms=[float(np.sqrt(np.mean(x*x))) for x in truth[pos]]
    mass=float(max(abs(x.mean()-u.mean()) for x in truth))
    report={'per_time_independent_error':errors,'per_time_refinement_error':refinement,
        'absolute_rms_errors':rms,'truth_rms':norms,'cap':cap,
        'qualified':all((e<=cap if norm>1e-10 else ab<=1e-12) for e,ab,norm in zip(errors,rms,norms)) and max(refinement)<=cap and mass<=1e-10,
        'relative_uncertainty_estimate':max(errors+refinement)+100*np.finfo(float).eps,
        'absolute_uncertainty_estimate':max(rms)+100*np.finfo(float).eps*max(1,float(np.max(abs(u)))),
        'dimension':u.ndim,'N':u.shape[0],
        'independent_method':'dense Fourier expm' if u.ndim==1 else 'DOP853',
        'production_method':'matrix-free exponential action','mass_error':mass}
    if not report['qualified']:raise RuntimeError(str(report))
    return truth,report
