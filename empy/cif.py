
import shlex, warnings, re

def read(fname):

    def try_number(s):
        m = re.match("^((\d+\.?\d*)|(\.\d+))(\(\d+\))?$", s)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return float(m.group(1))
        return s

    data = {}
    LOOP, VALUE, LOOP_DATA, TOP = range(4)
    state = TOP

    lex = shlex.shlex(open(fname),fname)
    lex.commenters = "#"
    lex.quotes = "';"
    lex.whitespace_split = True
    stripchars = lex.quotes + lex.whitespace

    for tok in lex:
        if state == TOP:
            if tok == "loop_":
                names = []
                state = LOOP
            elif tok.startswith("_"):
                name = tok.lstrip("_")
                state = VALUE
            elif tok.startswith("data_"):
                pass
            else:
                warnings.warn("unknown token '%s' file '%s' line %s\n" % (tok, lex.infile, lex.lineno))

        elif state == VALUE:
            data[name] = try_number(tok.strip(stripchars))
            state = TOP

        elif state == LOOP:
            if tok.startswith("_"):
                name = tok.lstrip("_")
                data[name] = []
                names.append(name)
            else:
                lex.push_token(tok)
                state = LOOP_DATA
                pos = 0

        elif state == LOOP_DATA:
            if tok.startswith("_") or tok == "loop_" or tok == "data_":
                lex.push_token(tok)
                state = TOP
            else:
                data[names[pos%len(names)]].append(try_number(tok.strip(stripchars)))
                pos += 1

    return data
