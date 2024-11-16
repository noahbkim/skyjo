#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "cards.h"
#include "hand.h"
#include "game.h"

// MARK: Game

typedef struct
{
    PyObject_HEAD;
    players_t players;
    cards_t cards;
    round_t round;
    turn_t turn;
} PyGameObject;

static PyTypeObject PyGameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "skyjo.Game",
    .tp_doc = PyDoc_STR("State and simulation for a single Skyjo game."),
    .tp_basicsize = sizeof(PyGameObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

// MARK: Module

static PyMethodDef METHODS[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT,
    "core",
    PyDoc_STR("Bindings for the C Skyjo simulation."),
    -1, // Size of per-interpreter state
    METHODS};

PyMODINIT_FUNC
PyInit_core()
{
    PyObject *module = NULL;

    // Cancel initialization if types aren't constructed yet.
    if (PyType_Ready(&PyGameType) < 0)
    {
        return NULL;
    }

    // Construct the containing module object.
    module = PyModule_Create(&MODULE);
    if (module == NULL)
    {
        goto error;
    }

    // Add a reference to our `Game` type.
    if (PyModule_AddObjectRef(module, "Game", (PyObject *)&PyGameType) < 0)
    {
        goto error;
    }

    return module;

error:
    Py_XDECREF(module);
    return NULL;
}
