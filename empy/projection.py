#!/usr/bin/env python

from .util import *

class Projection:
    
    def __init__(self, **kwargs):
        try:
            self.ax = kwargs.pop('ax')
        except KeyError:
            self.ax = plt.gca()

        self.equal_aspect() 

    def equal_aspect(self, size=None):
        self.ax.set_aspect("equal")
        if size is not None:
            self.ax.set_xlim(-size, size)
            self.ax.set_ylim(-size, size)
    
    def plot(self, vecs, *args, **kwargs):
        v, sel = self(vecs)
        v[~sel] = np.nan
        return self.ax.plot(v[...,0], v[...,1], *args, **kwargs)
        
    def kikuchi(self, vecs, *args, **kwargs):
        kwargs.setdefault("lw","auto")
        kwargs.setdefault("c","c")
        Uacc = kwargs.pop("Uacc", getattr(self, "Uacc", 200e3))
        from numpy import arcsin
        return self.circles(vecs, arcsin(vlen(vecs)/2/klen(Uacc)), *args, **kwargs)
    
    def circles(self, vecs, thetas=None, *args, **kwargs):
        return self._circles_line2d(vecs, thetas, *args, **kwargs)
        # this is *much* slower using pdf and ps backend
        #return self._circles_linecollection(vecs, thetas=None, *args, **kwargs)

    def _circles_line2d(self, vecs, thetas=None, *args, **kwargs):
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
        ret = []
        for values in zip(vecs, thetas, *values):
            v,t = values[:2]
            if keys:
                kwargs2 = dict(zip(keys, values[2:]))
                kwargs2.update(kwargs)
            else:
                kwargs2 = kwargs
            ret.extend( self.plot( circle(v,t), **kwargs2) )
        return ret

    def _circles_linecollection(self, vecs, thetas=None, *args, **kwargs):
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

        from matplotlib.collections import LineCollection
        self.ax.add_collection( LineCollection(lines, **kwargs) )

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

        return self.ax.scatter(vecs[sel,0], vecs[sel,1], **kwargs)

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

            return self.ax.text(v[...,0], v[...,1], s, *args, **kwargs)

 
class ThreeD(Projection):
    def __init__(self, **kwargs):
        try:
            self.ax = kwargs.pop('ax')
        except KeyError:
            from mpl_toolkits.mplot3d import Axes3D
            self.ax = plt.gca(projection='3d')
    
    def plot(self, v, *args, **kwargs):
        return self.ax.plot(v[...,0], v[...,1], v[...,2], *args, **kwargs)

    def text(self, v, l, *args, **kwargs):
        return self.ax.text(v[...,0], v[...,1], v[...,2], l, *args, **kwargs)

    def points(self, vecs, **kwargs):
        kwargs.setdefault("edgecolors", "none")
        if kwargs.get("s") == "auto":
            lens = vlen(vecs)
            minl = np.amin(lens[lens>0])
            maxl = np.amax(lens)
            kwargs["s"] = np.interp(lens, [minl,maxl], [50, 5])

        return self.ax.scatter(vecs[...,0], vecs[...,1], vecs[...,2], **kwargs)

class Stereo(Projection):

    def __init__(self, **kwargs):
        Projection.__init__(self, **kwargs)
        self.ax.axis('off')
        self.equal_aspect(1.05) 

    def __call__(self, v):
        return v[...,:2]/(vlen(v) + v[...,2])[...,np.newaxis], v[...,2] > -1e-5


class Flat(Projection):

    def __call__(self, v):
        return v[...,:2], np.abs(v[...,2]) < 1e-5

class Laue(Projection):

    def __init__(self, **kwargs):
        Projection.__init__(self, **kwargs)
        self.direction = kwargs.pop("direction", "back")

    def __call__(self, v):
        # calculate wave vector (wavelength) that diffracts at each recip. space vector
        k = -vdot(v,v)/2./v[...,2]
        return v[...,:2]/(k + v[...,2])[...,np.newaxis], k<0 if self.direction == "back" else k>0

class EDiff(Projection):
    pass


