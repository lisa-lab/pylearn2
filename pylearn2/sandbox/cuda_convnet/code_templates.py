
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"


def output_same_shape(out_arg_name, in_arg_name):
    return """
    const int *%(out_arg_name)s_dims = %(in_arg_name)s_dims;

    if (CudaNdarray_prep_output(& %%(%(out_arg_name)s)s, 4,
                                %(out_arg_name)s_dims))
    {
        %%(fail)s;
    }

    { // setup_nv_%(out_arg_name)s brace #1
    NVMatrix nv_%(out_arg_name)s(%%(%(out_arg_name)s)s,
     %(out_arg_name)s_dims[0] * %(out_arg_name)s_dims[1] *
     %(out_arg_name)s_dims[2],
     %(out_arg_name)s_dims[3], "%%(class_name)s:nv_%(out_arg_name)s");
    """ % locals()


def nv_matrix_create(arg_name):
    return """
    {
    NVMatrix nv_%(arg_name)s(%%(%(arg_name)s)s,
                  %(arg_name)s_dims[0] * %(arg_name)s_dims[1] *
                  %(arg_name)s_dims[2], %(arg_name)s_dims[3],
                  "%%(class_name)s:nv_%(arg_name)s");
    """ % locals()


def ensure_same_shape(new_arg, old_arg):
    return """
    if (%(new_arg)s_dims[0] != %(old_arg)s_dims[0] ||
        %(new_arg)s_dims[1] != %(old_arg)s_dims[1] ||
        %(new_arg)s_dims[2] != %(old_arg)s_dims[2] ||
        %(new_arg)s_dims[3] != %(old_arg)s_dims[3]) {
        PyErr_SetString(PyExc_ValueError, "%%(class_name)s: %(new_arg)s "
                                          "must have same shape as "
                                          "%(old_arg)s");
        %%(fail)s;
    }
    """ % locals()


def contiguity_check(arg_name):
    return """
    if (!CudaNdarray_is_c_contiguous(%%(%(arg_name)s)s))
    {
        PyErr_SetString(PyExc_ValueError,
            "%%(class_name)s: %(arg_name)s must be C contiguous");
        %%(fail)s;
    }
    """ % locals()


def dimension_check(arg_name, ndim):
    return """
    if (%%(%(arg_name)s)s->nd != %(ndim)d)
    {
        PyErr_Format(PyExc_ValueError,
            "%(arg_name)s must have ndim=%(ndim)d, got nd=%%%%i",
            %%(%(arg_name)s)s->nd);
        %%(fail)s;
    }
    """ % locals()

