#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "binary.h"
#include "category.h"
#include "timestamp.h"

// =====================
// BinaryTokenizer Class
// =====================
typedef struct __attribute__((aligned(8))) {
    PyObject_HEAD
    BinaryTokenizer tokenizer;
} PyBinaryTokenizer;

// --- Dealloc, New, Init ---
static void PyBinaryTokenizer_dealloc(PyBinaryTokenizer* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBinaryTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyBinaryTokenizer* self = (PyBinaryTokenizer*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    int num_bits = 8;
    int offset = 0;
    binary_init(&self->tokenizer, num_bits, offset);
    return (PyObject*)self;
}

static int PyBinaryTokenizer_init(PyBinaryTokenizer* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"num_bits", "offset", NULL};
    int num_bits = 8;
    int offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &num_bits, &offset))
        return -1;
    binary_init(&self->tokenizer, num_bits, offset);
    return 0;
}

// --- Methods: fit, encode, decode ---
static PyObject* PyBinaryTokenizer_fit(PyBinaryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;
    PyObject* array = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!array) {
        PyErr_SetString(PyExc_TypeError, "Could not convert input to float array");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)array);
    npy_intp size = PyArray_SIZE((PyArrayObject*)array);
    binary_fit(&self->tokenizer, data, size);
    Py_DECREF(array);
    Py_RETURN_NONE;
}

static PyObject* PyBinaryTokenizer_encode(PyBinaryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;

    // Initialize NumPy API (only once)
    import_array();
    static int numpy_initialized = 0;
    if (!numpy_initialized) {
        numpy_initialized = 1;
    }

    if (PyFloat_Check(input)) {
        // Single float case - return 1D array of indices
        double value = PyFloat_AsDouble(input);
        int indices[self->tokenizer.num_bits + 2];
        int count;
        binary_encode(&self->tokenizer, value, indices, &count);

        npy_intp dims[1] = {count};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!np_array) return NULL;

        int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
        memcpy(data, indices, count * sizeof(int));
        return np_array;

    } else if (PySequence_Check(input)) {
        // Sequence case - return array of arrays
        PyObject* seq = PySequence_Fast(input, "Expected a sequence");
        if (!seq) return NULL;

        Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
        PyObject* output = PyList_New(len);
        if (!output) {
            Py_DECREF(seq);
            return NULL;
        }

        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            double value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(seq);
                Py_DECREF(output);
                return NULL;
            }

            int indices[self->tokenizer.num_bits + 2];
            int count;
            binary_encode(&self->tokenizer, value, indices, &count);

            npy_intp dims[1] = {count};
            PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
            if (!np_array) {
                Py_DECREF(seq);
                Py_DECREF(output);
                return NULL;
            }

            int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
            memcpy(data, indices, count * sizeof(int));
            PyList_SET_ITEM(output, i, np_array);
        }

        Py_DECREF(seq);
        return output;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected float or sequence of floats");
        return NULL;
    }
}

static PyObject* __PyBinaryTokenizer_encode(PyBinaryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;

    if (PyFloat_Check(input)) {
        double value = PyFloat_AsDouble(input);
        int indices[self->tokenizer.num_bits + 2];
        int count;
        binary_encode(&self->tokenizer, value, indices, &count);
        PyObject* result = PyList_New(count);
        for (int i = 0; i < count; i++)
            PyList_SET_ITEM(result, i, PyLong_FromLong(indices[i]));
        return result;
    } else {
        PyObject* seq = PySequence_Fast(input, "Expected a sequence");
        if (!seq) return NULL;
        Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
        PyObject* output = PyList_New(len);
        if (!output) {
            Py_DECREF(seq);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            double value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(seq);
                Py_DECREF(output);
                return NULL;
            }
            int indices[self->tokenizer.num_bits];
            int count;
            binary_encode(&self->tokenizer, value, indices, &count);
            PyObject* tokens = PyList_New(count);
            for (int j = 0; j < count; j++)
                PyList_SET_ITEM(tokens, j, PyLong_FromLong(indices[j]));
            PyList_SET_ITEM(output, i, tokens);
        }
        Py_DECREF(seq);
        return output;
    }
}

static PyObject* PyBinaryTokenizer_decode(PyBinaryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;

    if (PySequence_Check(input)) {
        Py_ssize_t len_input = PySequence_Size(input);
        if (len_input <= 0) return NULL;
        PyObject* output = PyList_New(len_input);
        if (!output) return NULL;
        for (Py_ssize_t i = 0; i < len_input; i++) {
            double value;
            PyObject* tokens = PySequence_GetItem(input, i);
            if (!tokens) return NULL;
            Py_ssize_t len = PySequence_Size(tokens);
            if (len < 0) return NULL;
            if (len == 0) value = NAN;
            if (len > 0) {
                int indices[len];
                for (Py_ssize_t j = 0; j < len; j++) {
                    PyObject* item = PySequence_GetItem(tokens, j);
                    indices[j] = PyLong_AsLong(item);
                    Py_DECREF(item);
                    if (PyErr_Occurred()) return NULL;
                }
                value = binary_decode(&self->tokenizer, indices, len);
            }
            PyList_SET_ITEM(output, i, PyFloat_FromDouble(value));
        }
        return output;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of sequences.");
        return NULL;
    }
}

// --- Getters ---
static PyObject* PyBinaryTokenizer_get_num_bits(PyBinaryTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.fitted ? self->tokenizer.num_bits : -1);
}

static PyObject* PyBinaryTokenizer_get_max_active_features(PyBinaryTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.fitted ? self->tokenizer.num_bits : -1);
}

// --- Method Table & Type ---
static PyMethodDef PyBinaryTokenizer_methods[] = {
    {"fit", (PyCFunction)PyBinaryTokenizer_fit, METH_VARARGS, "Fit to data"},
    {"encode", (PyCFunction)PyBinaryTokenizer_encode, METH_VARARGS, "Encode values"},
    {"decode", (PyCFunction)PyBinaryTokenizer_decode, METH_VARARGS, "Decode tokens"},
    {NULL}
};

static PyGetSetDef PyBinaryTokenizer_getset[] = {
    {"num_bits", (getter)PyBinaryTokenizer_get_num_bits, NULL, "Number of bits (+2 sentinels)", NULL},
    {"max_active_features", (getter)PyBinaryTokenizer_get_max_active_features, NULL, "Maximum active features", NULL},
    {NULL}
};

static PyTypeObject PyBinaryTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.BinaryTokenizer",
    .tp_basicsize = sizeof(PyBinaryTokenizer),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyBinaryTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Tokenizes numerical data using recursive interval bi-section.",
    .tp_methods = PyBinaryTokenizer_methods,
    .tp_getset = PyBinaryTokenizer_getset,
    .tp_init = (initproc)PyBinaryTokenizer_init,
    .tp_new = PyBinaryTokenizer_new
};

// =====================
// CategoryTokenizer Class
// =====================
typedef struct __attribute__((aligned(8))) {
    PyObject_HEAD
    CategoryTokenizer tokenizer;
} PyCategoryTokenizer;

// --- Dealloc, New, Init ---
static void PyCategoryTokenizer_dealloc(PyCategoryTokenizer* self) {
    category_free(&self->tokenizer);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyCategoryTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyCategoryTokenizer* self = (PyCategoryTokenizer*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    
    int offset = 0;
    category_init(&self->tokenizer, offset);
    
    return (PyObject*)self;
}

static int PyCategoryTokenizer_init(PyCategoryTokenizer* self, PyObject* args, PyObject* kwds) {
    int offset = 0;
    PyObject* categories = NULL;
    static char* kwlist[] = {"categories", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist, &categories, &offset))
        return -1;
    
    if(offset) {
        if(offset > 0) category_init(&self->tokenizer, offset);
    }

    if (categories) 
    {
        PyObject* seq = PySequence_Fast(categories, "Expected a sequence");
        if (!seq) return -1;
        Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
        const char* values[len];
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            values[i] = PyUnicode_AsUTF8(item);
            if (!values[i]) {
                Py_DECREF(seq);
                return -1;
            }
        }
        category_fit(&self->tokenizer, values, len);
        Py_DECREF(seq);
    }
    return 0;
}

// --- Methods: fit, encode, decode ---
static PyObject* PyCategoryTokenizer_fit(PyCategoryTokenizer* self, PyObject* args) {
    PyObject* values;
    if (!PyArg_ParseTuple(args, "O", &values)) return NULL;
    PyObject* seq = PySequence_Fast(values, "Expected a sequence");
    if (!seq) return NULL;
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    const char* c_values[len];
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        c_values[i] = PyUnicode_AsUTF8(item);
        if (!c_values[i]) {
            Py_DECREF(seq);
            return NULL;
        }
    }
    category_fit(&self->tokenizer, c_values, len);
    Py_DECREF(seq);
    Py_RETURN_NONE;
}

static PyObject* PyCategoryTokenizer_encode(PyCategoryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;

    // Initialize NumPy API (only once)
    import_array();
    static int numpy_initialized = 0;
    if (!numpy_initialized) {
        numpy_initialized = 1;
    }

    if (PyUnicode_Check(input)) {
        // Single string case - return 1D numpy array with single element
        const char* value = PyUnicode_AsUTF8(input);
        int token = category_encode(&self->tokenizer, value);
        
        npy_intp dims[1] = {1};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!np_array) return NULL;
        
        int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
        data[0] = token;
        return np_array;
        
    } else if (PySequence_Check(input)) {
        // Sequence case - return 1D numpy array of tokens
        Py_ssize_t len = PySequence_Size(input);
        if (len == -1) return NULL;
        
        npy_intp dims[1] = {len};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!np_array) return NULL;
        
        int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
        
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(input, i);
            if (!item) {
                Py_DECREF(np_array);
                return NULL;
            }
            
            if (!PyUnicode_Check(item)) {
                Py_DECREF(item);
                Py_DECREF(np_array);
                PyErr_SetString(PyExc_TypeError, "Expected string in sequence");
                return NULL;
            }
            
            const char* value = PyUnicode_AsUTF8(item);
            data[i] = category_encode(&self->tokenizer, value);
            Py_DECREF(item);
        }
        return np_array;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected string or sequence of strings");
        return NULL;
    }
}

static PyObject* PyCategoryTokenizer_decode(PyCategoryTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;
    if (PyLong_Check(input)) {
        int token = PyLong_AsLong(input);
        const char* value = category_decode(&self->tokenizer, token);
        return PyUnicode_FromString(value);
    } else if (PySequence_Check(input)) {
        Py_ssize_t len = PySequence_Size(input);
        if (len == -1) return NULL;
        PyObject* result = PyList_New(len);
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(input, i);
            int token = PyLong_AsLong(item);
            const char* value = category_decode(&self->tokenizer, token);
            PyList_SET_ITEM(result, i, PyUnicode_FromString(value));
            Py_DECREF(item);
        }
        return result;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected int or sequence");
        return NULL;
    }
}

// --- Getters ---
static PyObject* PyCategoryTokenizer_get_num_bits(PyCategoryTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.fitted ? self->tokenizer.num_categories + 2 : -1);
}

static PyObject* PyCategoryTokenizer_get_num_categories(PyCategoryTokenizer* self, void* closure) {
    return PyLong_FromLong(self->tokenizer.fitted ? self->tokenizer.num_categories : -1);
}

static PyObject* PyCategoryTokenizer_get_max_active_features(PyCategoryTokenizer* self, void* closure) {
    return PyLong_FromLong(3);  // 2 sentinels + 1 active category
}

// --- Method Table & Type ---
static PyMethodDef PyCategoryTokenizer_methods[] = {
    {"fit", (PyCFunction)PyCategoryTokenizer_fit, METH_VARARGS, "Fit to categories"},
    {"encode", (PyCFunction)PyCategoryTokenizer_encode, METH_VARARGS, "Encode values"},
    {"decode", (PyCFunction)PyCategoryTokenizer_decode, METH_VARARGS, "Decode tokens"},
    {NULL}
};

static PyGetSetDef PyCategoryTokenizer_getset[] = {
    {"num_bits", (getter)PyCategoryTokenizer_get_num_bits, NULL, "Number of bits (+2 sentinels)", NULL},
    {"num_categories", (getter)PyCategoryTokenizer_get_num_categories, NULL, "Number of categories", NULL},
    {"max_active_features", (getter)PyCategoryTokenizer_get_max_active_features, NULL, "Maximum active features", NULL},
    {NULL}
};

static PyTypeObject PyCategoryTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.CategoryTokenizer",
    .tp_basicsize = sizeof(PyCategoryTokenizer),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyCategoryTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Category tokenizer with sentinel tokens",
    .tp_methods = PyCategoryTokenizer_methods,
    .tp_getset = PyCategoryTokenizer_getset,
    .tp_init = (initproc)PyCategoryTokenizer_init,
    .tp_new = PyCategoryTokenizer_new
};

// =====================
// TimestampTokenizer Class
// =====================
typedef struct __attribute__((aligned(8))) {
    PyObject_HEAD
    TimestampTokenizer tokenizer;
} PyTimestampTokenizer;

// --- Dealloc, New, Init ---
static void PyTimestampTokenizer_dealloc(PyTimestampTokenizer* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTimestampTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyTimestampTokenizer* self = (PyTimestampTokenizer*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    int offset = 0;
    timestamp_init(&self->tokenizer, 2000, 2100, offset);  // Default range
    return (PyObject*)self;
}

static int PyTimestampTokenizer_init(PyTimestampTokenizer* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"min_year", "max_year", "offset", NULL};
    int min_year = 2000, max_year = 2100, offset = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist, &min_year, &max_year, &offset))
        return -1;
    timestamp_init(&self->tokenizer, min_year, max_year, offset);
    return 0;
}

// --- Methods: encode, decode ---
static PyObject* PyTimestampTokenizer_encode(PyTimestampTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;
    
    // Import numpy array type (only done once)
    static PyObject* numpy_module = NULL;
    static PyObject* array_type = NULL;
    if (numpy_module == NULL) {
        numpy_module = PyImport_ImportModule("numpy");
        if (!numpy_module) return NULL;
        array_type = PyObject_GetAttrString(numpy_module, "array");
        if (!array_type) return NULL;
    }

    if (PyUnicode_Check(input)) {
        // Single string case
        const char* iso = PyUnicode_AsUTF8(input);
        int tokens[6], count;
        timestamp_encode(&self->tokenizer, iso, tokens, &count);
        
        // Create numpy array from tokens
        npy_intp dims[1] = {count};
        PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
        if (!np_array) return NULL;
        
        // Copy data
        int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
        memcpy(data, tokens, count * sizeof(int));
        return np_array;
        
    } else if (PySequence_Check(input)) {
        // Sequence case
        Py_ssize_t len = PySequence_Size(input);
        if (len <= 0) return NULL;
        
        PyObject* result = PyList_New(len);
        if (!result) return NULL;
        
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(input, i);
            if (!item) {
                Py_DECREF(result);
                return NULL;
            }
            
            if (!PyUnicode_Check(item)) {
                Py_DECREF(item);
                Py_DECREF(result);
                PyErr_SetString(PyExc_TypeError, "Expected string in sequence");
                return NULL;
            }
            
            const char* iso = PyUnicode_AsUTF8(item);
            int tokens[6], count;
            timestamp_encode(&self->tokenizer, iso, tokens, &count);
            
            // Create numpy array for each item
            npy_intp dims[1] = {count};
            PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_INT32);
            if (!np_array) {
                Py_DECREF(item);
                Py_DECREF(result);
                return NULL;
            }
            
            // Copy data
            int* data = (int*)PyArray_DATA((PyArrayObject*)np_array);
            memcpy(data, tokens, count * sizeof(int));
            
            PyList_SET_ITEM(result, i, np_array);
            Py_DECREF(item);
        }
        return result;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected string or sequence of strings");
        return NULL;
    }
}

static PyObject* PyTimestampTokenizer_decode(PyTimestampTokenizer* self, PyObject* args) {
    PyObject* input;
    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;
    if (PySequence_Check(input)) {
        Py_ssize_t len = PySequence_Size(input);
        if (len <= 0) return NULL;
        PyObject* result = PyList_New(len);
        for (Py_ssize_t i = 0; i < len; i++)
        {
            PyObject* item = PySequence_GetItem(input, i);
            // assume list of numpy arrays
            if(PySequence_Check(item))
            {
                Py_ssize_t len2 = PySequence_Size(item);
                int tokens[len2];
                for (Py_ssize_t j = 0; j < len2; j++) {
                    PyObject* token = PySequence_GetItem(item, j);
                    tokens[j] = PyLong_AsLong(token);
                    Py_DECREF(token);
                }
                char output[64];
                timestamp_decode(&self->tokenizer, tokens, len2, output);
                PyList_SET_ITEM(result, i, PyUnicode_FromString(output));
                Py_DECREF(item);
            }
        }
        return result;
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected sequence");
        return NULL;
    }
}

// --- Getter ---
static PyObject* PyTimestampTokenizer_get_num_bits(PyTimestampTokenizer* self, void* closure) {
    // int total = (self->tokenizer.max_year - self->tokenizer.min_year + 1) +
    //             12 + 31 + 24 + 60 + 60;
    return PyLong_FromLong(self->tokenizer.num_tokens);
}

static PyObject* PyTimestampTokenizer_get_max_active_features(PyTimestampTokenizer* self, void* closure) {
    return PyLong_FromLong(6);  // 6 active features
}

// --- Method Table & Type ---
static PyMethodDef PyTimestampTokenizer_methods[] = {
    {"encode", (PyCFunction)PyTimestampTokenizer_encode, METH_VARARGS, "Encode timestamp"},
    {"decode", (PyCFunction)PyTimestampTokenizer_decode, METH_VARARGS, "Decode tokens"},
    {NULL}
};

static PyGetSetDef PyTimestampTokenizer_getset[] = {
    {"num_bits", (getter)PyTimestampTokenizer_get_num_bits, NULL, "Total number of bits", NULL},
    {"max_active_features", (getter)PyTimestampTokenizer_get_max_active_features, NULL, "Total number features active (worst case)", NULL},
    {NULL}
};

static PyTypeObject PyTimestampTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "zeichenformer.TimestampTokenizer",
    .tp_basicsize = sizeof(PyTimestampTokenizer),
    .tp_dealloc = (destructor)PyTimestampTokenizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "ISO 8601 timestamp tokenizer",
    .tp_methods = PyTimestampTokenizer_methods,
    .tp_getset = PyTimestampTokenizer_getset,
    .tp_init = (initproc)PyTimestampTokenizer_init,
    .tp_new = PyTimestampTokenizer_new
};

// =====================
// Module Definition
// =====================
static PyModuleDef _tokenizers_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "zeichenformer._tokenizers",
    .m_doc = "High-performance tokenizers",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__tokenizers(void) {
    PyObject* m;
    if (PyType_Ready(&PyBinaryTokenizerType) < 0 ||
        PyType_Ready(&PyCategoryTokenizerType) < 0 ||
        PyType_Ready(&PyTimestampTokenizerType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&_tokenizers_module);
    if (!m) return NULL;
    Py_INCREF(&PyBinaryTokenizerType);
    Py_INCREF(&PyCategoryTokenizerType);
    Py_INCREF(&PyTimestampTokenizerType);
    PyModule_AddObject(m, "BinaryTokenizer", (PyObject*)&PyBinaryTokenizerType);
    PyModule_AddObject(m, "CategoryTokenizer", (PyObject*)&PyCategoryTokenizerType);
    PyModule_AddObject(m, "TimestampTokenizer", (PyObject*)&PyTimestampTokenizerType);
    import_array();
    return m;
}
