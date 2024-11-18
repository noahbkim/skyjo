#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "cards.h"
#include "hand.h"
#include "game.h"

// MARK: Agent

_Py_IDENTIFIER(join);
_Py_IDENTIFIER(flip);
_Py_IDENTIFIER(turn);

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
                "expected between %d and %d players, got at least %d",
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
            "expected between %d and %d players, got %d",
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
    PyObject *buffer = NULL;
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
            this->players[p].hand.columns = (f + 1) / HAND_ROWS;
        }
    }

    // Ask each agent to flip cards for their player.
    for (playercount_t p = 0; p < this->size; ++p)
    {
        turn_flip(&this->turn);
        if (!PyObject_CallMethodOneArg(this->agents[p], _PyUnicode_FromId(&PyId_flip), self))
        {
            goto error;
        }

        if (this->turn.state != FLIP_DONE)
        {
            PyErr_Format(
                PyExc_RuntimeError,
                "player %R at index %d did not flip two cards",
                this->agents[p], p);
            goto error;
        }
    }

    Py_RETURN_NONE;

error:
    Py_XDECREF(buffer);
    return NULL;
}

static bool _PyHand_ParseIndex(PyObject *arg, const hand_t *hand, handsize_t *index)
{
    if (PyLong_Check(arg))
    {
        Py_ssize_t value = PyLong_AsSsize_t(arg);
        if (PyErr_Occurred())
        {
            return false;
        }

        Py_ssize_t hand_size = hand->columns * HAND_ROWS;
        if (value < -hand_size || value >= hand_size)
        {
            PyErr_Format(
                PyExc_IndexError,
                "hand index %R out of range for hand size %d rows, %d columns",
                arg, HAND_ROWS, hand->columns);
            return false;
        }

        *index = (handsize_t)((value + hand_size) % hand_size);
        return true;
    }
    else if (PyTuple_Check(arg))
    {
        Py_ssize_t arg_size = PyTuple_GET_SIZE(arg);
        if (arg_size != 2)
        {
            PyErr_Format(
                PyExc_IndexError,
                "hand coordinates %R must be of size 2, not %zd",
                arg, arg_size);
            return false;
        }

        Py_ssize_t row = PyLong_AsSsize_t(PyTuple_GET_ITEM(arg, 0));
        Py_ssize_t column = PyLong_AsSsize_t(PyTuple_GET_ITEM(arg, 1));
        if (PyErr_Occurred())
        {
            return false;
        }

        if (row < -HAND_ROWS || row >= HAND_ROWS || column < -hand->columns || column >= hand->columns)
        {
            PyErr_Format(
                PyExc_IndexError,
                "hand index %R out of range for hand size %d rows, %d columns",
                arg, HAND_ROWS, hand->columns);
            return false;
        }

        row = (row + HAND_ROWS) % HAND_ROWS;
        column = (column + hand->columns) % hand->columns;
        *index = (handsize_t)(row * hand->columns + column);
        return true;
    }
    else
    {
        PyErr_Format(
            PyExc_TypeError,
            "hand indices may be integers or (row, column) tuples, not %T",
            arg);
        return false;
    }
}

static PyObject *PyGameObject_FlipCard(PyObject *self, PyObject *arg)
{
    PyGameObject *this = (PyGameObject *)self;
    turn_t *turn = &this->turn;
    switch (turn->state)
    {
    case FLIP_FIRST:
        if (!_PyHand_ParseIndex(arg, &this->players[turn->index].hand, &turn->finger_index))
        {
            return NULL;
        }
        turn->state = FLIP_SECOND;
        Py_RETURN_NONE;
    case FLIP_SECOND:
        if (!_PyHand_ParseIndex(arg, &this->players[turn->index].hand, &turn->second_finger_index))
        {
            return NULL;
        }
        if (turn->second_finger_index == turn->finger_index)
        {
            PyErr_Format(
                PyExc_RuntimeError,
                "the card at %R has already been flipped",
                arg);
            return NULL;
        }
        turn->state = FLIP_DONE;
        Py_RETURN_NONE;
    case FLIP_DONE:
        PyErr_SetString(
            PyExc_RuntimeError,
            "two cards have already been flipped");
        return NULL;
    default:
        PyErr_SetString(
            PyExc_RuntimeError,
            "this method may not be called in the current game state");
        return NULL;
    }
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
