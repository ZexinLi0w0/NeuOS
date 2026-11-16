"""Per-layer NeuOS for any torch.nn.Module (closest to the Caffe original).

Every leaf module call is one stage, in execution order. ``twins`` maps a leaf's
qualified name to its approximate replacement (e.g. a low-rank factorisation of a
conv); only those leaves are approximable. Twins must not be submodules of the
model itself (they would be wrapped as stages too).

    neu = attach(model, ctl, twins={"features.0": lowrank_conv0})
    with ctl_job(model):
        y = model(x)
    print(neu.last_log)
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
        finally:
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


def attach(model, ctl, twins=None, n_stages=None):
    model._neuos = LayerNeuOS(model, ctl, twins, n_stages)
    return model._neuos


def ctl_job(model):
    return model._neuos.job()
