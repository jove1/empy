
def simple(v):
    return "{0:g}{1:g}{2:g}".format(*v)

def latex(v):
    pos = "{:g}".format
    neg = r"\bar{{{:g}}}".format
    #neg = r"\overline{{{:g}}}".format
    return "$\mathregular{{{0}{1}{2}}}$".format(*[neg(-x) if x<0 else pos(x) for x in v])

def html(v):
    pos = "{:g}".format
    neg = "<span style='text-decoration: overline;'>{:g}</span>".format
    return "{0}{1}{2}".format(*[neg(-x) if x<0 else pos(x) for x in v])

def unicode(v):
    pos = "{:g}".format
    def neg(x):
        overline = u"\u0305"
        macron =  u"\u0304"
        x = pos(x)
        if len(x) == 1:
            return x+macron
        else:
            return u"".join(sum(zip(x,overline*len(x)),()))
    return u"{0}{1}{2}".format(*[neg(-x) if x<0 else pos(x) for x in v])
