from pylearn2.utils.string_utils import find_number
from pylearn2.utils.string_utils import tokenize_by_number
from pylearn2.utils.string_utils import number_aware_alphabetical_key

def test_find_number_0():
    r = find_number('sss')
    assert r is None

def test_find_number_1():
    s = 'jashlhl123sfs'
    r = find_number(s)
    assert s[r[0]:r[1]] == '123'

def test_find_number_2():
    s = 'aghwirougiuhfajlsopka"-987?'
    r = find_number(s)
    assert s[r[0]:r[1]] == '-987'


def test_find_number_3():
    s = 'jq% misdirect/ 82ghn 931'
    r = find_number(s)
    assert s[r[0]:r[1]] == '82'

def test_find_number_4():
    s = 'the quick brown fox 54.6 jumped'
    r = find_number(s)
    assert s[r[0]:r[1]] == '54.6'


def test_find_number_5():
    s = 'over the laz-91.2y dog'
    r = find_number(s)
    assert s[r[0]:r[1]] == '-91.2'

def test_find_number_6():
    s = 'query1e5 not found'
    r = find_number(s)
    assert s[r[0]:r[1]] == '1e5'

def test_find_number_7():
    s = 'sdglk421.e6'
    r = find_number(s)
    assert s[r[0]:r[1]] == '421.'

def test_find_number_8():
    s = 'ryleh -14e7$$!$'
    r = find_number(s)
    assert s[r[0]:r[1]] == '-14e7'

def token_lists_nearly_match(l,r):
    if len(l) != len(r):
        print "lengths don't match"
        print len(l)
        print len(r)
        return False

    for l_elem, r_elem in zip(l,r):
        if l_elem != r_elem:
            print l_elem," doesn't match ",r_elem
            return False

    return True

def test_tokenize_0():
    s = ' 123 klsdgh 56.7?98.2---\%-1e3'
    true_tokens = [' ',123,' klsdgh ',56.7,'?',98.2,'---\%',-1e3]
    tokens = tokenize_by_number(s)
    assert token_lists_nearly_match(tokens, true_tokens)

def test_number_aware_alphabetical_key():

    l = [ 'mystr_1', 'mystr_10', 'mystr_2', 'mystr_1_a', 'mystr' ]

    l.sort(key = number_aware_alphabetical_key )

    print l

    assert l == [ 'mystr', 'mystr_1', 'mystr_1_a', 'mystr_2', 'mystr_10' ]
