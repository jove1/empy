
def fancy(v):
    v = ("\\bar{{{:g}}}".format(-x) if x<0 else "{:g}".format(x) for x in v)
    return "${0}{1}{2}$".format(*v)

def simple(v):
    return "{0:g}{1:g}{2:g}".format(*v)

def html(v):
    v = ("<span style='text-decoration: overline;'>{:g}</span>".format(-x) if x<0 else "{:g}".format(x) for x in v)
    return "{0}{1}{2}".format(*v)
