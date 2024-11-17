#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "cards.h"
#include "hand.h"
#include "game.h"

// MARK: Game

typedef struct
{
    PyObject_HEAD;
    PyObject *agents[PLAYERS_MAX]; // Pointers to each agent instance.
    player_t players[PLAYERS_MAX]; // The hand and score of each player.
    size_t size;                   // The number of agents and players.
    round_t round;                 // State for the current round.
    turn_t turn;                   // State for the current turn.
    cards_t cards;                 // The draw and discard piles.
} PyGameObject;

static int PyGameObject_Clear(PyObject *self)
{
    PyGameObject *this = (PyGameObject *)self;

    // Set size to zero to ensure we don't access `NULL` agents.
    this->size = 0;
    for (playercount_t i = 0; i < PLAYERS_MAX; ++i)
    {
        Py_CLEAR(this->agents[i]);
    }

    return 0;
}

/** The `__init__` of the game object.
 *
 * Accepts a single argument `agents` that must yield simulation agent
 * instances when iterated through.
 */
static int PyGameObject_Init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyGameObject *this = (PyGameObject *)self;
    PyObject *agents = NULL;
    PyObject *agents_iter = NULL;
    Py_ssize_t agent_index = 0;
    PyObject *agent = NULL;

    // Parse `*args` and `**kwargs` to accept an argument `agents`.
    static char *kwlist[] = {"agents", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &agents))
    {
        goto error;
    }

    // Raise an exception if the agent object is not iterable.
    agents_iter = PyObject_GetIter(agents);
    if (!agents_iter)
    {
        goto error;
    }

    // Place each agent in our array. `PyIter_Next` increases the
    // refcount of each yielded agent
    while ((agent = PyIter_Next(agents_iter)) != NULL)
    {
        if (agent_index >= PLAYERS_MAX)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Expected between %d and %d players, got at least %d",
                PLAYERS_MIN, PLAYERS_MAX, index + 1);
            goto error;
        }

        this->agents[agent_index] = agent;
        agent = NULL; // Avoid double dereferencing.
        agent_index += 1;
    }

    if (agent_index < 3)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Expected between %d and %d players, got %d",
            PLAYERS_MIN, PLAYERS_MAX, agent_index);
        goto error;
    }

    // Note that `agent_index` is incremented once extra.
    this->size = agent_index;
    return 0;

error:
    Py_XDECREF(agents);
    Py_XDECREF(agents_iter);
    Py_XDECREF(agent);
    PyGameObject_Clear(self);
    return -1;
}

static int PyGameObject_Traverse(PyObject *self, visitproc visit, void *arg)
{
    PyGameObject *this = (PyGameObject *)self;

    // The only `PyObject` references we hold are agents.
    for (size_t i = 0; i < PLAYERS_MAX; ++i)
    {
        Py_VISIT(this->agents[i]);
    }

    return 0;
}

static void PyGameObject_Dealloc(PyObject *self)
{
    PyGameObject *this = (PyGameObject *)self;
    PyObject_GC_UnTrack(this);
    Py_TRASHCAN_BEGIN(this, PyGameObject_Dealloc);
    PyGameObject_Clear(self);
    Py_TRASHCAN_END;
}

static PyObject *PyGameObject_Play(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyGameObject *this = (PyGameObject *)self;
    unsigned int seed;

    // Parse `*args` and `**kwargs` to accept an argument `seed`.
    static char *kwlist[] = {"seed", NULL};
    if (PyTuple_GET_SIZE(args) == 0 && kwargs == NULL)
    {
        // If the user didn't pass anything, don't parse arguments.
        // This saves us from a tricky default situation with `seed`.
    }
    else if (PyArg_ParseTupleAndKeywords(args, kwargs, "I", kwlist, &seed))
    {
        srand(seed);
    }
    else
    {
        goto error;
    }

    // Completely wipe previous state.
    round_restore(&this->round);
    turn_restore(&this->turn);
    cards_restore(&this->cards);
    for (playercount_t i = 0; i < PLAYERS_MAX; ++i)
    {
        player_restore(this->players + i);
    }

    // Shuffle the deck.
    cards_shuffle(&this->cards);

    // Deal cards.
    for (handsize_t f = 0; f < HAND_SIZE; ++f)
    {
        for (playercount_t p = 0; p < this->size; ++p)
        {
            finger_t *finger = this->players[p].hand.fingers + f;
            finger_deal(finger, cards_deal(&this->cards));
        }
    }

    Py_RETURN_NONE;

error:
    return NULL;
}

static PyObject *PyGameObject_FlipCard(PyObject *self, PyObject *args)
{
    Py_RETURN_NONE;
}

static PyObject *PyGameObject_DrawCard(PyObject *self, PyObject *args)
{
    Py_RETURN_NONE;
}

static PyObject *PyGameObject_DiscardAndFlipCard(PyObject *self, PyObject *args)
{
    Py_RETURN_NONE;
}

static PyObject *PyGameObject_PlaceDrawnCard(PyObject *self, PyObject *args)
{
    Py_RETURN_NONE;
}

static PyMethodDef PyGameType_Methods[] = {
    {"play", _PyCFunction_CAST(PyGameObject_Play), METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Run a Skyjo simulation.")},
    {"flip_card", PyGameObject_FlipCard, METH_O, PyDoc_STR("Flip a card at the start of the round.")},
    {"draw_card", PyGameObject_DrawCard, METH_O, PyDoc_STR("Take a card from the draw pile.")},
    {"discard_and_flip", PyGameObject_DiscardAndFlipCard, METH_O, PyDoc_STR("Discard the drawn card and flip one in hand.")},
    {"place_drawn", PyGameObject_PlaceDrawnCard, METH_O, PyDoc_STR("Run a Skyjo simulation")},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject PyGameType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
        .tp_name = "skyjo.Game",
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = PyDoc_STR("State and simulation for a single Skyjo game."),
    .tp_traverse = PyGameObject_Traverse,
    .tp_clear = PyGameObject_Clear,
    .tp_dealloc = PyGameObject_Dealloc,
    .tp_basicsize = sizeof(PyGameObject),
    .tp_itemsize = 0,
    .tp_init = PyGameObject_Init,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = PyType_GenericNew,
    .tp_free = PyObject_GC_Del,
    .tp_methods = PyGameType_Methods,
};

// MARK: Module

static PyMethodDef PySkyjoModule_Methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef PySkyjoModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "core",
    .m_doc = PyDoc_STR("Bindings for the C Skyjo simulation."),
    .m_size = -1, // Size of per-interpreter state
    .m_methods = PySkyjoModule_Methods};

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
    module = PyModule_Create(&PySkyjoModule);
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
