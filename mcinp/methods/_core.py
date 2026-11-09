"""Shared propagation kernel for the interaction-correction methods.

The finite-grid signed directional propagator is implemented once, here. MCINP, Global-Q and
MCP-only differ only in how the interaction correction is parameterised, so they all run this
same time stepper. Sharing it is deliberate: it makes the comparison between the three methods a
comparison of correction models, not of two independently written numerical schemes.

Nothing in this module is dimension specific. The Fourier grid, the signed multipliers, the zero-mode
and even-grid Nyquist masking, the interaction band and the Hermitian projection are all derived from
the rank of the input, so the same code serves the one- and two-dimensional problems.
"""
import torch
__all__ = ['hermitian', 'multipliers', 'prepare', 'propagate']
def hermitian(v, axes):
    """Project ``v`` onto the Hermitian subspace along ``axes``, the symmetry of a real input."""
    partner = v
    for axis in axes:
        partner = torch.roll(torch.flip(partner, (axis,)), 1, axis)
    return (v + partner.conj()) / 2
def multipliers(ks, alpha, n, equation, shp):
    """Return the signed directional multiplier pair ``(m, m**alpha)`` per coordinate.

    ``m_j = sgn(k_j) |k_j|`` and the fractional partner is ``sgn(k_j) |k_j|**(alpha_j - 1)``. The zero
    mode is always frozen; for even ``n`` the coordinate's Nyquist plane is frozen as well, because a
    signed multiplier is not defined there.
    """
    pairs = []
    for j, k in enumerate(ks):
        active = (k != 0) & ((abs(k) != n / 2) if equation == 'signed' else True)
        magnitude = abs(k).clamp_min(1)
        sign = torch.sign(k) if equation == 'signed' else torch.ones_like(k)
        pairs.append((torch.where(active, sign * magnitude, 0),
                      torch.where(active, sign * magnitude ** (alpha[:, j].reshape(shp) - 1), 0)))
    return pairs
def prepare(u, ds, a, t, bandwidth, steps, equation):
    """Assemble every grid quantity the correction variants need, from the input rank alone."""
    dim = u.dim() - 1
    n = u.shape[-1]
    axes = tuple(range(1, dim + 1))
    shp = (u.shape[0],) + (1,) * dim
    k = torch.fft.fftfreq(n, d=1 / n, device=u.device, dtype=u.dtype)
    ks = torch.meshgrid(*([k] * dim), indexing='ij')
    pairs = multipliers(ks, a, n, equation, shp)
    coefficients = torch.fft.fftn(ds, dim=tuple(range(2, dim + 2)), norm='forward')
    means = ds.mean(tuple(range(2, dim + 2)))
    lam = sum(means[:, j].reshape(shp) * left * right for j, (left, right) in enumerate(pairs))
    band = torch.stack([abs(component) for component in ks]).amax(0) <= bandwidth
    band = band & ~torch.stack([component == 0 for component in ks]).all(0)
    drifted = torch.fft.ifftn(coefficients * band, dim=tuple(range(2, dim + 2)), norm='forward').real
    return dict(dim=dim, n=n, axes=axes, shp=shp, ks=ks, pairs=pairs, means=means, alpha=a,
                lam=lam, tau=t / steps, drifted=drifted)
def propagate(module, u, ds, a, t):
    """Integrate the mean-coefficient part and apply the module's interaction correction.

    ``module`` supplies ``dim``, ``bandwidth``, ``steps``, ``equation`` and ``correction_kind``
    (``'modewise'``, ``'scalar'`` or ``'none'``); a modewise module additionally supplies
    ``q_value(w, J, parts, ctx, expand)``. The correction has the same negative sign in both
    dimensions, and ``q`` is a real scalar per sample, never a full state.
    """
    ctx = prepare(u, ds, a, t, module.bandwidth, module.steps, module.equation)
    axes, shp = ctx['axes'], ctx['shp']
    lam, tau, drifted, pairs = ctx['lam'], ctx['tau'], ctx['drifted'], ctx['pairs']
    state = torch.fft.fftn(u, dim=axes, norm='forward')
    expand = lambda field: field.reshape(shp).expand_as(state.real)
    for _ in range(module.steps if module.correction_kind != 'none' else 1):
        step = t if module.correction_kind == 'none' else tau
        carrier = state * torch.exp(-step.reshape(shp) * lam)
        if module.correction_kind == 'none':
            state = carrier
            break
        parts = [left * torch.fft.fftn(drifted[:, j] * torch.fft.ifftn(right * carrier, dim=axes, norm='forward'), dim=axes, norm='forward')
                 for j, (left, right) in enumerate(pairs)]
        interaction = sum(parts)
        q = module.q if module.correction_kind == 'scalar' else module.q_value(carrier, interaction, parts, ctx, expand)
        state = hermitian(carrier - step.reshape(shp) * interaction * q, axes)
    out = torch.fft.ifftn(state, dim=axes, norm='forward')
    if out.imag.detach().abs().max() > 2e-5:
        raise ValueError('Hermitian gate')
    return torch.where((t == 0).reshape(shp), u, out.real)
