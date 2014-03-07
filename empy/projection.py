#!/usr/bin/env python

from .util import *

class Projection:
    def get_ax(self, d):
        try:
            return d.pop('ax')
        except KeyError:
            return plt.gca()

    def plot(self, v, *args, **kwargs):
        ax = self.get_ax(kwargs)
        
        v, sel = self(v)
        v[~sel] = np.nan
        ax.plot(v[...,0], v[...,1], *args, **kwargs)

    def old_circles(self, vecs, thetas=None, *args, **kwargs):
        if thetas is None:
            thetas = np.zeros(vecs.shape[:-1])

        if kwargs.get("lw") == "auto":
            lens = vlen(vecs)
            minl = np.amin(lens[lens>0])
            maxl = np.amax(lens)
            if abs(minl-maxl) < 1e-5:
                minl *= 0.9
                maxl *= 1.1
            kwargs["lw"] = np.interp(lens, [minl,maxl], [2, 0.5])

        keys = []
        values = []
        for k,v in kwargs.items():
            if isinstance(v, np.ndarray):
                keys.append(k)
                values.append(v)
                del kwargs[k]

        for values in zip(vecs, thetas, *values):
            v,t = values[:2]
            if keys:
                kwargs2 = dict(zip(keys, values[2:]))
                kwargs2.update(kwargs)
            else:
                kwargs2 = kwargs
            self.plot( circle(v,t), **kwargs2)

    def circles(self, vecs, thetas=None, *args, **kwargs):
        if thetas is None:
            thetas = np.zeros(vecs.shape[:-1])
       
        try:
            kwargs['colors'] = kwargs.pop("c")
        except KeyError:
            pass

        try:
            kwargs["linewidths"] = kwargs.pop("lw")
        except KeyError:
            pass

        if kwargs.get("linewidths", None) == "auto":
            lens = vlen(vecs)
            minl = np.amin(lens[lens>0])
            maxl = np.amax(lens)
            if abs(minl-maxl) < 1e-5:
                minl *= 0.9
                maxl *= 1.1
            kwargs["linewidths"] = np.interp(lens, [minl,maxl], [2, 0.5])
        
        lines = []
        for v,t in zip(vecs, thetas):
            v,sel = self(circle(v,t))
            v[~sel] = np.nan
            lines.append(v)

        ax = self.get_ax(kwargs)
        from matplotlib.collections import LineCollection
        ax.add_collection( LineCollection(lines, **kwargs) )

    def points(self, vecs, *args, **kwargs):
        kwargs.setdefault("edgecolors", "none")
        kwargs.setdefault("zorder", 3)

        if kwargs.get("s") == "auto":
            lens = vlen(vecs)
            minl = np.amin(lens[lens>0])
            maxl = np.amax(lens)
            kwargs["s"] = np.interp(lens, [minl,maxl], [50, 5])

        vecs, sel = self(vecs)

        # apply sel to all parameters (color for example)
        for k,v in kwargs.items():
            if isinstance(v,np.ndarray):
                kwargs[k] = v[sel]  

        ax = self.get_ax(kwargs)
        return ax.scatter(vecs[sel,0], vecs[sel,1], **kwargs)

    def labels(self, vecs, labels=None, *args, **kwargs):
        vecs = np.asarray(vecs)
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "bottom")
        kwargs.setdefault("clip_on", True)

        import matplotlib.patheffects as PathEffects
        kwargs.setdefault("path_effects", [PathEffects.withStroke(linewidth=2, foreground="w")])

        if labels is None:
            labels = [str(v) for v in vecs]
        elif callable(labels):
            labels = [labels(v) for v in vecs]

        order = np.argsort(vlen(vecs))
        labels = np.asarray(labels)

        keys = []
        values = []
        for k,v in kwargs.items():
            if isinstance(v, np.ndarray):
                keys.append(k)
                values.append(v[order])
                del kwargs[k]
        
        done = set()  
        for values in zip(vecs[order], labels[order], *values):
            v, l = values[:2]
            if keys:
                kwargs2 = dict(zip(keys, values[2:]))
                kwargs2.update(kwargs)
            else:
                kwargs2 = kwargs

            self.text(v, l, done=done, **kwargs2)

    def text(self, v, s, *args, **kwargs):
        v, sel = self(v)
        if sel:
            try:
                done = kwargs.pop('done')
            except KeyError:
                pass
            else:
                key = tuple(np.around(v[...,:2], 4))
                if key in done:
                    return
                done.add(key)

            ax = self.get_ax(kwargs)
            return ax.text(v[...,0], v[...,1], s, *args, **kwargs)
        
class Stereo(Projection):
    def __call__(self, v):
        return v[...,:2]/(vlen(v) + v[...,2])[...,np.newaxis], v[...,2] > -1e-5

class ThreeD(Projection):
    def get_ax(self, d):
        try:
            return d.pop('ax')
        except KeyError:
            from mpl_toolkits.mplot3d import Axes3D
            return plt.gca(projection='3d')

    def plot(self, v, *args, **kwargs):
        ax = self.get_ax(kwargs)
        return ax.plot(v[...,0], v[...,1], v[...,2], *args, **kwargs)

    def text(self, v, l, *args, **kwargs):
        ax = self.get_ax(kwargs)
        return ax.text(v[...,0], v[...,1], v[...,2], l, *args, **kwargs)

    def points(self, vecs, **kwargs):
        kwargs.setdefault("edgecolors", "none")
        if kwargs.get("s") == "auto":
            lens = vlen(vecs)
            minl = np.amin(lens[lens>0])
            maxl = np.amax(lens)
            kwargs["s"] = np.interp(lens, [minl,maxl], [50, 5])

        ax = self.get_ax(kwargs)
        return ax.scatter(vecs[...,0], vecs[...,1], vecs[...,2], **kwargs)

class Flat(Projection):
    def __call__(self, v):
        return v[...,:2], np.abs(v[...,2]) < 1e-5

class EDiff(Projection):
    pass

class Laue(Projection):
    pass
