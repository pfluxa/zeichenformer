





#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Include your C tokenizer implementations.
// These headers must contain the function prototypes we've developed.
#include "numerical.h"
#include "category.h"
#include "timestamp.h"

// ============================================================================
// NumericalTokenizer Python Class
// ============================================================================
typedef struct {
    PyObject_HEAD
    NumericalTokenizer tokenizer;
} PyNumericalTokenizer;

// --- Dealloc, New, Init ---

static void PyNumericalTokenizer_dealloc(PyNumericalTokenizer* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyNumericalTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyNumericalTokenizer* self = (PyNumericalTokenizer*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyNumericalTokenizer_init(PyNumericalTokenizer* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"num_bits", "offset", NULL};
    int num_bits = 8;
    int offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &num_bits, &offset)) {
        return -1;
    }
    numerical_init(&self->tokenizer, num_bits, offset);
    return 0;
}

// --- Methods: fit, encode, decode ---

static PyObject* PyNumericalTokenizer_fit(PyNumericalTokenizer* self, PyObject* args) {
    PyArrayObject* in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL;
    }

    // Ensure the input array is a C-style contiguous array of doubles.
    PyArrayObject* contiguous_array = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)in_array, NPY_DOUBLE, 1, 1);
    if (!contiguous_array) {
        return NULL;
    }

    double* data = (double*)PyArray_DATA(contiguous_array);
    npy_intp size = PyArray_SIZE(contiguous_array);

    numerical_fit(&self->tokenizer, data, size);

    Py_DECREF(contiguous_array);
    Py_RETURN_NONE;
}

static PyObject* PyNumericalTokenizer_encode(PyNumericalTokenizer* self, PyObject* args) {
    PyArrayObject* in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL;
    }

    PyArrayObject* contiguous_array = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)in_array, NPY_DOUBLE, 1, 1);
    if (!contiguous_array) {
        return NULL;
    }

    npy_intp n_samples = PyArray_SIZE(contiguous_array);
    double* values = (double*)PyArray_DATA(contiguous_array);

    PyObject* result_list = PyList_New(n_samples);
    if (!result_list) {
        Py_DECREF(contiguous_array);
        return NULL;
    }

    // Allocate a buffer for the C function to write indices into.
    int* indices_buffer = malloc(self->tokenizer.num_bits * sizeof(int));
    if (!indices_buffer) {
        Py_DECREF(contiguous_array);
        Py_DECREF(result_list);
        return PyErr_NoMemory();
    }

    for (npy_intp i = 0; i < n_samples; ++i) {
        int count = 0;
        numerical_encode(&self->tokenizer, values[i], indices_buffer, &count);

        if (count < 0) {
            PyErr_SetString(PyExc_ValueError, "Encoding failed. Tokenizer might not be fitted.");
            Py_DECREF(result_list); // The list contains partially filled data, so discard it.
            goto cleanup;
        }

        npy_intp dims[1] = {count};
        PyObject* token_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!token_array) {
            Py_DECREF(result_list);
            goto cleanup;
        }

        memcpy(PyArray_DATA((PyArrayObject*)token_array), indices_buffer, count * sizeof(int));
        PyList_SET_ITEM(result_list, i, token_array); // SET_ITEM steals the reference
    }

cleanup:
    free(indices_buffer);
    Py_DECREF(contiguous_array);

    if (PyErr_Occurred()) {
        return NULL;
    }
    return result_list;
}

static PyObject* PyNumericalTokenizer_decode(PyNumericalTokenizer* self, PyObject* args) {
    PyObject* in_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &in_list)) {
        return NULL;
    }

    Py_ssize_t n_samples = PyList_GET_SIZE(in_list);
    npy_intp dims[1] = {n_samples};
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!out_array) {
        return NULL;
    }
    float* out_data = (float*)PyArray_DATA(out_array);

    for (Py_ssize_t i = 0; i < n_samples; ++i) {
        PyObject* token_obj = PyList_GET_ITEM(in_list, i); // Borrows reference
        PyArrayObject* token_array = (PyArrayObject*)PyArray_ContiguousFromAny(token_obj, NPY_INT32, 1, 1);
        if (!token_array) {
            Py_DECREF(out_array);
            return NULL;
        }

        int* indices = (int*)PyArray_DATA(token_array);
        int count = PyArray_SIZE(token_array);

        out_data[i] = (float)numerical_decode(&self->tokenizer, indices, count);
        Py_DECREF(token_array);
    }

    return (PyObject*)out_array;
}

// --- Getters ---
static PyObject* PyNumericalTokenizer_get_num_tokens(PyNumericalTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.num_bits);
}

static PyObject* PyNumericalTokenizer_get_vocab_size(PyNumericalTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.num_bits);
}

// --- Method Table & Type ---
static PyMethodDef PyNumericalTokenizer_methods[] = {
    {"fit", (PyCFunction)PyNumericalTokenizer_fit, METH_VARARGS, "Fit tokenizer to numerical data."},
    {"encode", (PyCFunction)PyNumericalTokenizer_encode, METH_VARARGS, "Encode numerical value(s) into sparse tokens."},
    {"decode", (PyCFunction)PyNumericalTokenizer_decode, METH_VARARGS, "Decode sparse tokens back to numerical values."},
    {NULL}
};

static PyGetSetDef PyNumericalTokenizer_getset[] = {
    {"num_tokens", (getter)PyNumericalTokenizer_get_num_tokens, NULL, "Number of tokens used for encoding.", NULL},
    {"vocab_size", (getter)PyNumericalTokenizer_get_vocab_size, NULL, "The largest possible token value.", NULL},
    {NULL}
};

static PyTypeObject PyNumericalTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.NumericalTokenizer",
    .tp_basicsize = sizeof(PyNumericalTokenizer),
    .tp_dealloc = (destructor)PyNumericalTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Tokenizes numerical data using recursive interval bisection.",
    .tp_methods = PyNumericalTokenizer_methods,
    .tp_getset = PyNumericalTokenizer_getset,
    .tp_init = (initproc)PyNumericalTokenizer_init,
    .tp_new = PyNumericalTokenizer_new
};


// ============================================================================
// CategoryTokenizer Python Class
// ============================================================================
typedef struct {
    PyObject_HEAD
    CategoryTokenizer tokenizer;
} PyCategoryTokenizer;

// --- Dealloc, New, Init ---
static void PyCategoryTokenizer_dealloc(PyCategoryTokenizer* self) {
    category_free(&self->tokenizer);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyCategoryTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    return type->tp_alloc(type, 0);
}

static int PyCategoryTokenizer_init(PyCategoryTokenizer* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"offset", NULL};
    int offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &offset)) {
        return -1;
    }
    category_init(&self->tokenizer, offset);
    return 0;
}

// --- Methods: fit, encode, decode ---
static PyObject* PyCategoryTokenizer_fit(PyCategoryTokenizer* self, PyObject* args) {
    PyArrayObject* in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL;
    }

    if (PyArray_TYPE(in_array) != NPY_OBJECT) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of object type (strings).");
        return NULL;
    }

    npy_intp n_samples = PyArray_SIZE(in_array);
    const char** c_values = malloc(n_samples * sizeof(char*));
    if (!c_values) return PyErr_NoMemory();

    for (npy_intp i = 0; i < n_samples; ++i) {
        PyObject* item = *(PyObject**)PyArray_GETPTR1(in_array, i);
        if (!PyUnicode_Check(item)) {
            free(c_values);
            PyErr_SetString(PyExc_TypeError, "All items in the array must be strings.");
            return NULL;
        }
        c_values[i] = PyUnicode_AsUTF8(item);
        if (!c_values[i]) {
            free(c_values);
            return NULL; // PyUnicode_AsUTF8 sets an error
        }
    }

    category_fit(&self->tokenizer, c_values, n_samples);
    free(c_values);
    Py_RETURN_NONE;
}

static PyObject* PyCategoryTokenizer_encode(PyCategoryTokenizer* self, PyObject* args) {
    PyArrayObject* in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL;
    }

    if (PyArray_TYPE(in_array) != NPY_OBJECT) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of object type (strings).");
        return NULL;
    }

    npy_intp n_samples = PyArray_SIZE(in_array);
    PyObject* result_list = PyList_New(n_samples);
    if (!result_list) return NULL;

    for (npy_intp i = 0; i < n_samples; ++i) {
        PyObject* item = *(PyObject**)PyArray_GETPTR1(in_array, i);
        const char* value = PyUnicode_AsUTF8(item);
        if (!value) {
            Py_DECREF(result_list);
            return NULL;
        }
        
        int token = category_encode(&self->tokenizer, value);
        
        npy_intp dims[1] = {1};
        PyObject* token_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!token_array) {
            Py_DECREF(result_list);
            return NULL;
        }
        *(int*)PyArray_DATA((PyArrayObject*)token_array) = token;
        PyList_SET_ITEM(result_list, i, token_array);
    }
    return result_list;
}

static PyObject* PyCategoryTokenizer_decode(PyCategoryTokenizer* self, PyObject* args) {
    PyObject* in_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &in_list)) {
        return NULL;
    }

    Py_ssize_t n_samples = PyList_GET_SIZE(in_list);
    npy_intp dims[1] = {n_samples};
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_OBJECT);
    if (!out_array) return NULL;

    for (Py_ssize_t i = 0; i < n_samples; ++i) {
        PyObject* token_obj = PyList_GET_ITEM(in_list, i);
        PyArrayObject* token_array = (PyArrayObject*)PyArray_ContiguousFromAny(token_obj, NPY_INT32, 1, 1);
        if (!token_array) {
            Py_DECREF(out_array);
            return NULL;
        }
        
        int token = *(int*)PyArray_DATA(token_array);
        const char* value_str = category_decode(&self->tokenizer, token);
        PyObject* py_str = PyUnicode_FromString(value_str);
        
        // PyArray_SETITEM steals the reference to py_str, so no DECREF is needed.
        if (PyArray_SETITEM(out_array, PyArray_GETPTR1(out_array, i), py_str) < 0) {
            Py_DECREF(token_array);
            Py_DECREF(out_array);
            return NULL;
        }
        Py_DECREF(token_array);
    }
    return (PyObject*)out_array;
}


// --- Getters ---
static PyObject* PyCategoryTokenizer_get_num_tokens(PyCategoryTokenizer* self, void* closure) {
    return PyLong_FromLong(1);
}

static PyObject* PyCategoryTokenizer_get_vocab_size(PyCategoryTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.num_tokens);
}

// --- Method Table & Type ---
static PyMethodDef PyCategoryTokenizer_methods[] = {
    {"fit", (PyCFunction)PyCategoryTokenizer_fit, METH_VARARGS, "Fit tokenizer to a NumPy array of strings."},
    {"encode", (PyCFunction)PyCategoryTokenizer_encode, METH_VARARGS, "Encode a NumPy array of strings into tokens."},
    {"decode", (PyCFunction)PyCategoryTokenizer_decode, METH_VARARGS, "Decode a list of token arrays back to strings."},
    {NULL}
};

static PyGetSetDef PyCategoryTokenizer_getset[] = {
    {"num_tokens", (getter)PyCategoryTokenizer_get_num_tokens, NULL, "Number of tokens used for encoding.", NULL},
    {"vocab_size", (getter)PyCategoryTokenizer_get_vocab_size, NULL, "Total vocabulary size including special tokens.", NULL},
    {NULL}
};

static PyTypeObject PyCategoryTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.CategoryTokenizer",
    .tp_basicsize = sizeof(PyCategoryTokenizer),
    .tp_dealloc = (destructor)PyCategoryTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Tokenizes categorical string data.",
    .tp_methods = PyCategoryTokenizer_methods,
    .tp_getset = PyCategoryTokenizer_getset,
    .tp_init = (initproc)PyCategoryTokenizer_init,
    .tp_new = PyCategoryTokenizer_new
};


// ============================================================================
// TimestampTokenizer Python Class
// ============================================================================
typedef struct {
    PyObject_HEAD
    TimestampTokenizer tokenizer;
} PyTimestampTokenizer;

// --- Dealloc, New, Init ---
static void PyTimestampTokenizer_dealloc(PyTimestampTokenizer* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTimestampTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    return type->tp_alloc(type, 0);
}

static int PyTimestampTokenizer_init(PyTimestampTokenizer* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"min_year", "max_year", "offset", NULL};
    int min_year = 1970, max_year = 2070, offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist, &min_year, &max_year, &offset))
        return -1;
    timestamp_init(&self->tokenizer, min_year, max_year, offset);
    return 0;
}

// --- Methods: encode, decode ---
static PyObject* PyTimestampTokenizer_encode(PyTimestampTokenizer* self, PyObject* args) {
    PyArrayObject* in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        return NULL;
    }

    if (PyArray_TYPE(in_array) != NPY_OBJECT) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of object type (strings).");
        return NULL;
    }

    npy_intp n_samples = PyArray_SIZE(in_array);
    PyObject* result_list = PyList_New(n_samples);
    if (!result_list) return NULL;

    int tokens_buffer[6];
    for (npy_intp i = 0; i < n_samples; ++i) {
        PyObject* item = *(PyObject**)PyArray_GETPTR1(in_array, i);
        const char* iso_str = PyUnicode_AsUTF8(item);
        if (!iso_str) {
            Py_DECREF(result_list);
            return NULL;
        }

        int count = 0;
        timestamp_encode(&self->tokenizer, iso_str, tokens_buffer, &count);
        if (count < 0) {
            PyErr_SetString(PyExc_ValueError, "Failed to encode timestamp. It may be invalid or out of range.");
            Py_DECREF(result_list);
            return NULL;
        }

        npy_intp dims[1] = {6};
        PyObject* token_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!token_array) {
            Py_DECREF(result_list);
            return NULL;
        }
        memcpy(PyArray_DATA((PyArrayObject*)token_array), tokens_buffer, 6 * sizeof(int));
        PyList_SET_ITEM(result_list, i, token_array);
    }
    return result_list;
}

static PyObject* PyTimestampTokenizer_decode(PyTimestampTokenizer* self, PyObject* args) {
    PyObject* in_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &in_list)) {
        return NULL;
    }

    Py_ssize_t n_samples = PyList_GET_SIZE(in_list);
    npy_intp dims[1] = {n_samples};
    PyArrayObject* out_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_OBJECT);
    if (!out_array) return NULL;

    char iso_buffer[64];
    for (Py_ssize_t i = 0; i < n_samples; ++i) {
        PyObject* token_obj = PyList_GET_ITEM(in_list, i);
        PyArrayObject* token_array = (PyArrayObject*)PyArray_ContiguousFromAny(token_obj, NPY_INT32, 1, 1);
        if (!token_array || PyArray_SIZE(token_array) != 6) {
            Py_XDECREF(token_array);
            Py_DECREF(out_array);
            PyErr_SetString(PyExc_ValueError, "All items in list must be arrays of 6 integer tokens.");
            return NULL;
        }
        
        int* tokens = (int*)PyArray_DATA(token_array);
        timestamp_decode(&self->tokenizer, tokens, 6, iso_buffer, sizeof(iso_buffer));
        PyObject* py_str = PyUnicode_FromString(iso_buffer);
        
        if (PyArray_SETITEM(out_array, PyArray_GETPTR1(out_array, i), py_str) < 0) {
            Py_DECREF(token_array);
            Py_DECREF(out_array);
            return NULL;
        }
        Py_DECREF(token_array);
    }
    return (PyObject*)out_array;
}

// --- Getter ---
static PyObject* PyTimestampTokenizer_get_num_tokens(PyTimestampTokenizer* self, void* closure) {
    return PyLong_FromLong(6);
}

static PyObject* PyTimestampTokenizer_get_vocab_size(PyTimestampTokenizer* self, void* closure) {
    return PyLong_FromLong((long)(fmax(self->tokenizer.max_year - self->tokenizer.min_year, 60) + 1));
}


// --- Method Table & Type ---
static PyMethodDef PyTimestampTokenizer_methods[] = {
    {"encode", (PyCFunction)PyTimestampTokenizer_encode, METH_VARARGS, "Encode a NumPy array of ISO strings into tokens."},
    {"decode", (PyCFunction)PyTimestampTokenizer_decode, METH_VARARGS, "Decode a list of token arrays back to ISO strings."},
    {NULL}
};


static PyGetSetDef PyTimestampTokenizer_getset[] = {
    {"num_tokens", (getter)PyTimestampTokenizer_get_num_tokens, NULL, "Number of tokens used for encoding.", NULL},
    {"vocab_size", (getter)PyTimestampTokenizer_get_vocab_size, NULL, "Total vocabulary size including special tokens.", NULL},
    {NULL}
};

static PyTypeObject PyTimestampTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.TimestampTokenizer",
    .tp_basicsize = sizeof(PyTimestampTokenizer),
    .tp_dealloc = (destructor)PyTimestampTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Tokenizes ISO 8601 timestamp strings into 6 components.",
    .tp_methods = PyTimestampTokenizer_methods,
    .tp_init = (initproc)PyTimestampTokenizer_init,
    .tp_getset = PyTimestampTokenizer_getset,
    .tp_new = PyTimestampTokenizer_new
};


// ============================================================================
// Module Definition
// ============================================================================
static PyModuleDef _tokenizers_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "zeichenformer._tokenizers",
    .m_doc = "High-performance tokenizers written in C.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__tokenizers(void) {
    PyObject* m;
    if (PyType_Ready(&PyNumericalTokenizerType) < 0 ||
        PyType_Ready(&PyCategoryTokenizerType) < 0 ||
        PyType_Ready(&PyTimestampTokenizerType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&_tokenizers_module);
    if (!m) return NULL;

    Py_INCREF(&PyNumericalTokenizerType);
    if (PyModule_AddObject(m, "NumericalTokenizer", (PyObject*)&PyNumericalTokenizerType) < 0) {
        Py_DECREF(&PyNumericalTokenizerType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyCategoryTokenizerType);
    if (PyModule_AddObject(m, "CategoryTokenizer", (PyObject*)&PyCategoryTokenizerType) < 0) {
        Py_DECREF(&PyCategoryTokenizerType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyTimestampTokenizerType);
    if (PyModule_AddObject(m, "TimestampTokenizer", (PyObject*)&PyTimestampTokenizerType) < 0) {
        Py_DECREF(&PyTimestampTokenizerType);
        Py_DECREF(m);
        return NULL;
    }

    import_array();
    if (PyErr_Occurred()) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
