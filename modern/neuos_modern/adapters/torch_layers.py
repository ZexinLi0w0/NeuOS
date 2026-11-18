"""Per-layer NeuOS for any torch.nn.Module (closest to the Caffe original).

Every leaf module call is one stage, in execution order. ``twins`` maps a leaf's
qualified name to its approximate replacement (e.g. a low-rank factorisation of a
conv); only those leaves are approximable. Twins must not be submodules of the
model itself (they would be wrapped as stages too).

    neu = attach(model, ctl, twins={"features.0": lowrank_conv0})
    with ctl_job(model):
        y = model(x)
    print(neu.last_log)

``lowrank_twins(model, rank)`` builds the twins the original NeuOS used
(models/lowrank, Jaderberg et al. 2014 scheme 2): each KxK Conv2d becomes a
vertical Kx1 conv to ``rank`` channels followed by a horizontal 1xK conv.
"""
import contextlib


def _leaves(model):
    return [(n, m) for n, m in model.named_modules() if not any(True for _ in m.children())]


class LayerNeuOS:
    def __init__(self, model, ctl, twins=None, n_stages=None):
        import torch
        self.torch, self.model, self.ctl = torch, model, ctl
        self.twins = dict(twins or {})
        leaves = _leaves(model)
        self.n_stages = n_stages or len(leaves)  # updated to the observed count after each job
        self.cuda = any(p.is_cuda for p in model.parameters())
        self.active, self.last_log, self._k = None, None, 0
        self._saved = []
        for name, m in leaves:
            self._saved.append((m, m.__dict__.get("forward")))
            m.forward = self._wrap(name, m.forward)

    def _sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def _wrap(self, name, fwd):
        def forward(*args, **kwargs):
            job = self.active
            if job is None:
                return fwd(*args, **kwargs)
            k, self._k = self._k, self._k + 1
            self._sync()
            approx = job.before_stage(k, name in self.twins)
            out = (self.twins[name] if approx else fwd)(*args, **kwargs)
            self._sync()
            job.after_stage(k)
            return out
        return forward

    @contextlib.contextmanager
    def job(self):
        """One inference == one NeuOS job."""
        self.active, self._k = self.ctl.begin_job(self.n_stages), 0
        try:
            yield self
        except BaseException:
            self.active = None      # interrupted mid-stage (e.g. SIGTERM): no end() on a partial job
            raise
        job, self.active = self.active, None
        self.last_log = job.end()
        self.n_stages = self._k or self.n_stages

    def detach(self):
        for m, fwd in self._saved:
            if fwd is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = fwd
        self._saved = []


def lowrank_conv(conv, rank):
    """SVD factorisation of a KxK ``nn.Conv2d`` (groups=1, dilation=1) into
    Sequential(Conv2d(Cin, r, (K,1)), Conv2d(r, Cout, (1,K))).

    With W[o,c,y,x], M[(c,y),(o,x)] = W[o,c,y,x] = U S V^T; the vertical filter k is
    U[:,k]*sqrt(S_k) reshaped to (c,y), the horizontal one V[:,k]*sqrt(S_k) as (o,x).
    rank >= min(Cin*K, Cout*K) is exact."""
    import torch
    from torch import nn
    w = conv.weight.detach()
    cout, cin, kh, kw = w.shape
    r = max(1, min(int(rank), cin * kh, cout * kw))
    # SVD on the CPU: the Jetson torch wheels' CUDA linalg can fail to load (cusolver mismatch)
    m = w.permute(1, 2, 0, 3).reshape(cin * kh, cout * kw).double().cpu()
    u, sv, vh = torch.linalg.svd(m, full_matrices=False)
    root = sv[:r].sqrt()
    vert = (u[:, :r] * root).T.reshape(r, cin, kh, 1)
    horiz = (vh[:r].T * root).reshape(cout, kw, r).permute(0, 2, 1).reshape(cout, r, 1, kw)
    (sh, sw), (ph, pw) = conv.stride, conv.padding
    v = nn.Conv2d(cin, r, (kh, 1), stride=(sh, 1), padding=(ph, 0), bias=False,
                  device=w.device, dtype=w.dtype)
    h = nn.Conv2d(r, cout, (1, kw), stride=(1, sw), padding=(0, pw), bias=conv.bias is not None,
                  device=w.device, dtype=w.dtype)
    with torch.no_grad():
        v.weight.copy_(vert)
        h.weight.copy_(horiz)
        if conv.bias is not None:
            h.bias.copy_(conv.bias)
    return nn.Sequential(v, h).eval()


def lowrank_twins(model, rank=0.25):
    """``{name: low-rank twin}`` for every factorisable conv (KxK with K > 1, groups=1,
    dilation=1, numeric padding). ``rank``: an int, a float ratio of the full rank
    min(Cin*K, Cout*K), or a dict ``{name: int}`` like models/lowrank/config.json
    (convs missing from the dict are skipped)."""
    import math
    from torch import nn
    twins = {}
    for name, m in model.named_modules():
        if not (isinstance(m, nn.Conv2d) and m.groups == 1 and m.dilation == (1, 1)
                and m.kernel_size[0] > 1 and m.kernel_size[0] == m.kernel_size[1]
                and isinstance(m.padding, tuple)):
            continue
        k, full = m.kernel_size[0], min(m.in_channels, m.out_channels) * m.kernel_size[0]
        if isinstance(rank, dict):
            if name not in rank:
                continue
            r = rank[name]
        else:
            r = rank if isinstance(rank, int) else math.ceil(rank * full)
        twins[name] = lowrank_conv(m, r)
    return twins


def attach(model, ctl, twins=None, n_stages=None):
    model._neuos = LayerNeuOS(model, ctl, twins, n_stages)
    return model._neuos


def ctl_job(model):
    return model._neuos.job()
