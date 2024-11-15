#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "cards.h"

static PyMethodDef method_def[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "ext",
    NULL, // Docs
    -1,   // Size of per-interpreter state
    method_def};

PyMODINIT_FUNC
PyInit_ext()
{
    return PyModule_Create(&module_def);
}
