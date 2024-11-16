#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "cards.h"
#include "hand.h"
#include "game.h"

static PyMethodDef METHODS[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT,
    "core",
    NULL, // Docs
    -1,   // Size of per-interpreter state
    METHODS};

PyMODINIT_FUNC
PyInit_core()
{
    return PyModule_Create(&MODULE);
}
