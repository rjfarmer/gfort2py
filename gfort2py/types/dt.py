# SPDX-License-Identifier: GPL-2.0+

import ctypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..compilation import Compile, CompileArgs, Modulise
from ..utils import copy_array, is_64bit
from .base import AllocStrategy, f_type
from .module import get_module

__all__ = [
    "ftype_dt",
    "ftype_class",
    "ftype_dt_explicit",
    "ftype_dt_assumed_shape",
]


_all_dts: dict[tuple[str, int], type[ctypes.Structure]] = {}
_building_dts: set[tuple[str, int]] = set()
_pointer_owners: dict[tuple[int, str], Any] = {}


def _dt_cache_module_key(module_name: str, module_obj=None) -> str:
    if module_obj is not None:
        filename = getattr(module_obj, "filename", None)
        if filename:
            return str(Path(filename).resolve())
        return f"module_obj:{id(module_obj)}"

    return f"module_name:{module_name}"


def _array_descriptor_ctype(ndims: int) -> type[ctypes.Structure]:
    index_t: Any
    size_t: Any
    if is_64bit():
        index_t = ctypes.c_int64
        size_t = ctypes.c_int64
    else:
        index_t = ctypes.c_int32
        size_t = ctypes.c_int32

    class _bounds14(ctypes.Structure):
        _fields_ = [
            ("stride", index_t),
            ("lbound", index_t),
            ("ubound", index_t),
        ]

    class _dtype_type(ctypes.Structure):
        _fields_ = [
            ("elem_len", size_t),
            ("version", ctypes.c_int32),
            ("rank", ctypes.c_byte),
            ("type", ctypes.c_byte),
            ("attribute", ctypes.c_ushort),
        ]

    class _fAllocArray(ctypes.Structure):
        _fields_ = [
            ("base_addr", ctypes.c_void_p),
            ("offset", size_t),
            ("dtype", _dtype_type),
            ("span", index_t),
            ("dims", _bounds14 * ndims),
        ]

    return _fAllocArray


def _find_ftype(ftype: str, kind: int):
    # Import lazily to avoid circular import during types package initialization.
    from . import find_ftype

    return find_ftype(ftype, kind)


def _array_index(index: Any, shape: tuple[int, ...]) -> int:
    if len(shape) == 0:
        raise ValueError("Array is not allocated")

    if isinstance(index, tuple):
        ind = int(np.ravel_multi_index(index, shape, order="F"))
    else:
        ind = int(index)

    if ind < 0 or ind >= int(np.prod(shape)):
        raise IndexError("Out of bounds")

    return ind


def _dt_index_debug_enabled() -> bool:
    value = os.environ.get("GFORT2PY_DT_INDEX_DEBUG", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _shape_from_descriptor(desc: ctypes.Structure, ndims: int) -> tuple[int, ...]:
    shape = []
    for i in range(ndims):
        shape.append(desc.dims[i].ubound - desc.dims[i].lbound + 1)
    return tuple(int(i) for i in shape)


@dataclass(frozen=True)
class _TypeBoundProcedureInfo:
    name: str
    proc_ref: int
    nopass: bool
    pass_arg: str
    pass_arg_num: int


class _BoundTypeProcedure:
    def __init__(self, owner: "ftype_dt", info: _TypeBoundProcedureInfo) -> None:
        self._owner = owner
        self._info = info

    def _procedure(self):
        from ..procedures import factory as proc_factory

        lib = getattr(self._owner, "_lib", None)
        if lib is None:
            raise RuntimeError("No shared library handle bound for type-bound call")

        module_obj = getattr(self._owner, "_module_obj", None)
        if module_obj is None:
            module_obj = get_module(self._owner._resolved_use_module_name())

        symbol = module_obj[self._info.proc_ref]
        return proc_factory(symbol)(lib, symbol, module_obj)

    def __call__(self, *args, **kwargs):
        proc = self._procedure()
        if self._info.nopass:
            return proc(*args, **kwargs)

        pass_pos = max(self._info.pass_arg_num - 1, 0)
        try:
            pass_ref = proc.definition.properties.formal_argument[pass_pos]
            pass_symbol = proc._module[pass_ref]
            if pass_symbol.type.lower() == "class":
                raise NotImplementedError(
                    "Type-bound PASS with polymorphic CLASS dummy is not supported yet"
                )
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

        if self._info.pass_arg and self._info.pass_arg in kwargs:
            raise ValueError(
                f"Type-bound PASS argument '{self._info.pass_arg}' is implicit"
            )

        if pass_pos < len(args):
            candidate = args[pass_pos]
            explicit_self = (
                hasattr(candidate, "_ctype")
                and callable(getattr(candidate, "pointer", None))
                and callable(getattr(candidate, "pointer2", None))
            )
            if explicit_self:
                raise ValueError(
                    "PASS object is implicit and must not be passed explicitly"
                )

        if pass_pos > len(args):
            raise ValueError(
                f"Invalid PASS position {pass_pos + 1} for {self._info.name}"
            )

        new_args = list(args)
        new_args.insert(pass_pos, self._owner)
        return proc(*new_args, **kwargs)


class _DTModuleResolutionMixin:
    def _resolved_symbol_module(self) -> str:
        # Access bound symbol directly from instance state; touching the
        # `_sym` property can raise when no symbol is bound.
        symbol = self.__dict__.get("_symbol", None)
        return getattr(symbol, "module", "")

    def _resolved_use_module_name(self) -> str:
        sym_module = self._resolved_symbol_module()
        if sym_module not in {"", "."}:
            return sym_module

        module_obj = getattr(self, "_module_obj", None)
        if module_obj is not None:
            try:
                first_key = next(iter(module_obj.keys()))
                name = module_obj[first_key].module
                if name not in {"", "."}:
                    return name
            except (AttributeError, KeyError, StopIteration, TypeError):
                pass

        module_name = getattr(self, "_module_name", "")
        if module_name not in {"", "."}:
            return module_name

        return sym_module

    def _resolved_module_file(self) -> Path | None:
        module_obj = getattr(self, "_module_obj", None)
        if module_obj is not None:
            filename = getattr(module_obj, "filename", None)
            if filename:
                return Path(filename)

        module_name = getattr(self, "_module_name", "")
        if module_name in {"", "."}:
            module_name = self._resolved_use_module_name()

        if module_name in {"", "."}:
            return None

        return Path(get_module(module_name).filename)


class ftype_dt(_DTModuleResolutionMixin, f_type):
    kind = -1
    dtype = np.dtype(object)

    def __init__(self, value=None):
        if "_module_name" not in self.__dict__ or "_type_id" not in self.__dict__:
            self._module_name, self._type_id = self._type_info_from_symbol(self._sym)
        super().__init__()
        self.value = value

    @staticmethod
    def _type_info_from_symbol(symbol) -> tuple[str, int]:
        module_name = symbol.module
        type_id = int(symbol.properties.typespec.class_ref)
        return module_name, type_id

    @classmethod
    def _dt_definition(cls, module_name: str, type_id: int, module_obj=None):
        if (module_name == "" or module_name == ".") and module_obj is not None:
            return module_obj[type_id]
        return get_module(module_name)[type_id]

    @classmethod
    def _component_is_pointer(cls, comp) -> bool:
        attribute = getattr(comp, "attribute", None)
        if attribute is not None:
            return bool(getattr(attribute, "pointer", False))

        attributes = getattr(comp, "attributes", None)
        if attributes is not None:
            return bool(getattr(attributes, "pointer", False))

        properties = getattr(comp, "properties", None)
        if properties is not None:
            return bool(getattr(properties.attributes, "pointer", False))

        return False

    @classmethod
    def _component_ctype(cls, module_name: str, comp, module_obj=None) -> Any:
        base_ctype: Any
        if comp.typespec.is_dt:
            if cls._component_is_pointer(comp):
                base_ctype = cls._build_ctype(
                    module_name,
                    int(comp.typespec.class_ref),
                    module_obj=module_obj,
                )
                return ctypes.POINTER(base_ctype)

            base_ctype = cls._build_ctype(
                module_name,
                int(comp.typespec.class_ref),
                module_obj=module_obj,
            )
        else:
            ctype_name = comp.typespec.type.lower()
            kind = int(comp.typespec.kind)
            if ctype_name == "character":
                strlen = int(comp.typespec.charlen.value)
                if strlen <= 0:
                    strlen = 1
                base_ctype = ctypes.c_char * strlen
            else:
                base_ctype = _find_ftype(ctype_name, kind)().ctype

        if comp.array.is_array:
            if comp.array.is_explicit:
                return base_ctype * int(np.prod(comp.array.pyshape))
            return _array_descriptor_ctype(comp.array.rank)

        return base_ctype

    @classmethod
    def _build_ctype(
        cls, module_name: str, type_id: int, module_obj=None
    ) -> type[ctypes.Structure]:
        key = (_dt_cache_module_key(module_name, module_obj=module_obj), type_id)
        if key in _all_dts:
            return _all_dts[key]

        class dt(ctypes.Structure):
            pass

        _all_dts[key] = dt
        _building_dts.add(key)

        try:
            definition = cls._dt_definition(module_name, type_id, module_obj=module_obj)
            fields: list[tuple[str, Any]] = []
            for name in definition.properties.components.keys():
                comp = definition.properties.components[name]
                fields.append(
                    (
                        comp.name,
                        cls._component_ctype(module_name, comp, module_obj=module_obj),
                    )
                )

            dt._fields_ = fields
            return dt
        except Exception:
            _all_dts.pop(key, None)
            raise
        finally:
            _building_dts.discard(key)

    def _components(self):
        definition = self._dt_definition(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )
        return definition.properties.components

    @staticmethod
    def _parse_type_bound_from_raw(raw: str) -> dict[str, _TypeBoundProcedureInfo]:
        res: dict[str, _TypeBoundProcedureInfo] = {}
        for match in re.finditer(r"\(\s*'([^']+)'\s+\(([^()]*)\)\)", raw):
            name = match.group(1)
            meta = match.group(2)
            parsed = re.search(
                r"\b(NOPASS|PASS)\b.*?'([^']*)'\s+(\d+)\s+(\d+)\s*$",
                meta,
            )
            if parsed is None:
                continue

            mode = parsed.group(1)
            pass_arg = parsed.group(2)
            pass_arg_num = int(parsed.group(3))
            proc_ref = int(parsed.group(4))
            if proc_ref <= 0:
                continue

            res[name] = _TypeBoundProcedureInfo(
                name=name,
                proc_ref=proc_ref,
                nopass=mode == "NOPASS",
                pass_arg=pass_arg,
                pass_arg_num=pass_arg_num,
            )

        return res

    @classmethod
    def _type_bound_procedures_for(
        cls,
        module_name: str,
        type_id: int,
        module_obj,
        visited: set[int],
    ) -> dict[str, _TypeBoundProcedureInfo]:
        if type_id in visited:
            return {}
        visited.add(type_id)

        definition = cls._dt_definition(module_name, type_id, module_obj=module_obj)
        components = definition.properties.components
        local: dict[str, _TypeBoundProcedureInfo] = {}

        for key in components.keys():
            comp = components[key]
            proc_meta = getattr(comp, "proc_pointer", None)
            if proc_meta is None:
                continue

            proc_ref = int(getattr(proc_meta, "proc_ref", 0))
            if proc_ref <= 0:
                continue

            info = _TypeBoundProcedureInfo(
                name=comp.name,
                proc_ref=proc_ref,
                nopass=bool(getattr(proc_meta, "nopass", False)),
                pass_arg=str(getattr(proc_meta, "pass_arg", "")),
                pass_arg_num=int(getattr(proc_meta, "pass_arg_num", 0)),
            )
            local[info.name] = info

        if len(local) == 0:
            raw = str(getattr(definition.properties, "_raw", ""))
            local = cls._parse_type_bound_from_raw(raw)

        inherited: dict[str, _TypeBoundProcedureInfo] = {}
        for key in components.keys():
            comp = components[key]
            if comp.array.is_array:
                continue
            if not getattr(comp.typespec, "is_dt", False):
                continue

            parent_id = int(comp.typespec.class_ref)
            if parent_id <= 0 or parent_id == type_id:
                continue

            parent_definition = cls._dt_definition(
                module_name, parent_id, module_obj=module_obj
            )
            if comp.name.lower() != parent_definition.name.lower():
                continue

            inherited.update(
                cls._type_bound_procedures_for(
                    module_name, parent_id, module_obj, visited
                )
            )

        inherited.update(local)
        return inherited

    def _type_bound_procedures(self) -> dict[str, _TypeBoundProcedureInfo]:
        return self._type_bound_procedures_for(
            self._module_name,
            self._type_id,
            getattr(self, "_module_obj", None),
            set(),
        )

    @property
    def ftype(self):
        definition = self._dt_definition(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )
        return definition.name

    @property
    def ctype(self) -> type[ctypes.Structure]:
        return self._build_ctype(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )

    @classmethod
    def from_existing_ctype(
        cls, module_name: str, type_id: int, ctype_obj, module_obj=None, lib=None
    ) -> "ftype_dt":
        c = cls.__new__(cls)
        c._module_name = module_name
        c._type_id = type_id
        c._module_obj = module_obj
        c._lib = lib
        c._symbol = None
        c.__init__()  # type: ignore[misc]
        c._ctype = ctype_obj
        return c

    def _component_shape(self, comp) -> tuple[int, ...]:
        if comp.array.is_explicit:
            return tuple(int(i) for i in comp.array.pyshape)
        return tuple()

    def _alloc_component_source(self, comp, shape: tuple[int, ...]) -> Modulise:
        dims = ",".join([f"1:{i}" for i in shape])
        use_module_name = self._resolved_use_module_name()
        dt_name = self.ftype

        string = f"""
        subroutine alloc_component(x)
        use {use_module_name}, only: {dt_name}
        type({dt_name}), intent(inout) :: x
        if (allocated(x%{comp.name})) deallocate(x%{comp.name})
        allocate(x%{comp.name}({dims}))
        end subroutine alloc_component
        """
        return Modulise(string)

    def _allocate_component_array(self, comp, value, shape: tuple[int, ...]) -> None:
        if (
            value.base_addr is not None
            and _shape_from_descriptor(value, comp.array.rank) == shape
        ):
            return

        code = self._alloc_component_source(comp, shape)
        compiled = Compile(code.as_module(), name=code.strhash())
        comp_args = CompileArgs()
        module_file = self._resolved_module_file()
        if module_file is not None and module_file.parent:
            comp_args.INCLUDE_FLAGS = f"-I{module_file.parent}"
        if not compiled.compile(args=comp_args):
            raise ValueError(f"Failed to allocate derived-type component {comp.name}")

        lib = compiled.platform.load_library(compiled.library_filename)
        sub = getattr(lib, f"__{compiled.name}_MOD_alloc_component")
        sub(ctypes.byref(self._ctype))

        refreshed = getattr(self._ctype, comp.name)
        if refreshed.base_addr is None:
            raise ValueError(
                f"Allocation failed for derived-type component {comp.name}"
            )

    def _ensure_pointer_component(self, comp):
        value = getattr(self._ctype, comp.name)
        if bool(value):
            return value

        pointee_ctype = self._build_ctype(
            self._module_name,
            int(comp.typespec.class_ref),
            module_obj=getattr(self, "_module_obj", None),
        )
        pointee = pointee_ctype()
        pointer = ctypes.pointer(pointee)

        # Keep the pointee alive independently of temporary wrappers.
        _pointer_owners[(ctypes.addressof(self._ctype), comp.name)] = pointee

        setattr(self._ctype, comp.name, pointer)
        refreshed = getattr(self._ctype, comp.name)
        if not bool(refreshed):
            raise ValueError(f"Allocation failed for pointer component {comp.name}")

        return refreshed

    def _get_array_value(self, comp, value):
        ctype_name = comp.typespec.type.lower()
        kind = int(comp.typespec.kind)
        elem_size: int
        if ctype_name == "character":
            strlen = int(comp.typespec.charlen.value)
            if strlen <= 0 and hasattr(value, "dtype"):
                strlen = int(value.dtype.elem_len)
            if strlen <= 0:
                strlen = 1
            dtype = np.dtype(f"S{strlen}")
            elem_size = strlen
        else:
            base = _find_ftype(ctype_name, kind)()
            dtype = base.dtype
            elem_size = ctypes.sizeof(base.ctype)
        if comp.array.is_explicit:
            shape = self._component_shape(comp)
            arr = np.ctypeslib.as_array(value).reshape(shape, order="F")
            return arr.astype(dtype)

        if value.base_addr is None:
            return None

        shape = _shape_from_descriptor(value, comp.array.rank)
        array = np.zeros(shape, dtype=dtype, order="F")
        copy_array(
            value.base_addr,
            array.ctypes.data,
            elem_size,
            int(np.prod(shape)),
        )
        return array

    def _set_array_value(self, comp, value, input_value):
        if input_value is None:
            if comp.array.is_explicit:
                raise ValueError("Explicit-shape array component must not be None")
            return

        ctype_name = comp.typespec.type.lower()
        kind = int(comp.typespec.kind)
        elem_size: int
        if ctype_name == "character":
            if not np.issubdtype(np.asarray(input_value).dtype, np.bytes_):
                raise TypeError("Character strings must be bytes (S dtype)")
            strlen = int(comp.typespec.charlen.value)
            if strlen <= 0:
                strlen = int(np.asarray(input_value).dtype.itemsize)
            if strlen <= 0:
                strlen = 1
            dtype = np.dtype(f"S{strlen}")
            elem_size = strlen
        else:
            base = _find_ftype(ctype_name, kind)()
            dtype = base.dtype
            elem_size = ctypes.sizeof(base.ctype)

        array = np.asfortranarray(input_value).astype(dtype, copy=False)

        if comp.array.is_explicit:
            shape = self._component_shape(comp)
            if tuple(array.shape) != shape:
                raise ValueError(f"Wrong shape, got {array.shape} expected {shape}")

            flat = array.ravel("F")
            copy_array(
                flat.ctypes.data,
                ctypes.addressof(value),
                elem_size,
                int(np.prod(shape)),
            )
            return

        shape = tuple(int(i) for i in array.shape)
        if len(shape) != comp.array.rank:
            raise ValueError(
                f"Wrong number of dimensions, got {len(shape)} expected {comp.array.rank}"
            )

        self._allocate_component_array(comp, value, shape)

        flat = array.ravel("F")
        copy_array(
            flat.ctypes.data,
            value.base_addr,
            elem_size,
            int(np.prod(shape)),
        )

    def __getitem__(self, key):
        comps = self._components()
        if key not in comps:
            raise KeyError(f"{key} not present in {self.ftype}")

        comp = comps[key]
        value = getattr(self._ctype, key)

        if comp.typespec.is_dt:
            if comp.array.is_array:
                raise NotImplementedError(
                    "Derived-type components that are arrays are unsupported"
                )

            if self._component_is_pointer(comp):
                value = self._ensure_pointer_component(comp)
                return ftype_dt.from_existing_ctype(
                    self._module_name,
                    int(comp.typespec.class_ref),
                    value.contents,
                    module_obj=getattr(self, "_module_obj", None),
                    lib=getattr(self, "_lib", None),
                )

            return ftype_dt.from_existing_ctype(
                self._module_name,
                int(comp.typespec.class_ref),
                value,
                module_obj=getattr(self, "_module_obj", None),
                lib=getattr(self, "_lib", None),
            )

        if comp.array.is_array:
            return self._get_array_value(comp, value)

        if comp.typespec.type.lower() == "character":
            return bytes(value).decode("ascii")

        return value

    def __setitem__(self, key, value):
        comps = self._components()
        if key not in comps:
            raise KeyError(f"{key} not present in {self.ftype}")

        comp = comps[key]
        cur = getattr(self._ctype, key)

        if comp.typespec.is_dt:
            if comp.array.is_array:
                raise NotImplementedError(
                    "Derived-type components that are arrays are unsupported"
                )

            if self._component_is_pointer(comp):
                nested = self[key]
            else:
                nested = ftype_dt.from_existing_ctype(
                    self._module_name,
                    int(comp.typespec.class_ref),
                    cur,
                    module_obj=getattr(self, "_module_obj", None),
                    lib=getattr(self, "_lib", None),
                )
            nested.value = value
            return

        if comp.array.is_array:
            self._set_array_value(comp, cur, value)
            return

        if comp.typespec.type.lower() == "character":
            if hasattr(value, "encode"):
                value = value.encode("ascii")
            strlen = int(comp.typespec.charlen.value)
            if len(value) > strlen:
                value = value[:strlen]
            else:
                value = value + b" " * (strlen - len(value))
            setattr(self._ctype, key, value)
            return

        setattr(self._ctype, key, value)

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        if value is None:
            return
        for key, val in value.items():
            self[key] = val

    def keys(self) -> list[str]:
        return list(self._components().keys())

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    def __getattr__(self, key: str):
        procs = self._type_bound_procedures()
        if key in procs:
            return _BoundTypeProcedure(self, procs[key])

        raise AttributeError(f"{key} not present in {self.ftype}")

    def __dir__(self):
        return sorted(set([*self.keys(), *self._type_bound_procedures().keys()]))

    def __repr__(self):
        return f"type({self.ftype})"


class ftype_class(_DTModuleResolutionMixin, f_type):
    kind = -1
    dtype = np.dtype(object)

    def __init__(self, value=None):
        if "_module_name" not in self.__dict__ or "_class_id" not in self.__dict__:
            self._module_name = self._sym.module
            self._class_id = int(self._sym.properties.typespec.class_ref)
        super().__init__()
        self.value = value
        self._vptr_owner = None

    @property
    def ftype(self):
        return "class"

    @property
    def ctype(self):
        class _ClassDescriptor(ctypes.Structure):
            _fields_ = [("_data", ctypes.c_void_p), ("_vptr", ctypes.c_void_p)]

        return _ClassDescriptor

    def _set_data_pointer(self, value) -> None:
        self._ctype._data = ctypes.c_void_p(ctypes.addressof(value._ctype))

    def _set_vptr(self, value) -> None:
        if not hasattr(self._ctype, "_vptr"):
            return

        lib = getattr(value, "_lib", None)
        if lib is None:
            return

        module_obj = getattr(self, "_module_obj", None)
        if module_obj is None:
            return

        module_name = value._resolved_use_module_name().lower()
        type_name = value.ftype
        candidates = [
            f"__vtab_{module_name}_{type_name}",
            f"__vtab_{module_name}_{type_name.lower()}",
        ]

        vtab_symbol = None
        for key in module_obj.keys():
            sym = module_obj[key]
            if sym.name in candidates:
                vtab_symbol = sym
                break

        if vtab_symbol is None:
            return

        vtab_ctype = ftype_dt._build_ctype(
            self._module_name,
            int(vtab_symbol.properties.typespec.class_ref),
            module_obj=module_obj,
        )
        vtab_obj = vtab_ctype.in_dll(lib, vtab_symbol.mangled_name)
        self._vptr_owner = vtab_obj
        self._ctype._vptr = ctypes.c_void_p(ctypes.addressof(vtab_obj))

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        if value is None:
            return

        is_wrapper = (
            hasattr(value, "_ctype")
            and callable(getattr(value, "pointer", None))
            and callable(getattr(value, "pointer2", None))
        )
        if not is_wrapper:
            raise TypeError(
                "Polymorphic CLASS arguments require a derived-type wrapper"
            )

        self._set_data_pointer(value)
        self._set_vptr(value)


class ftype_dt_array(f_type):
    kind = -1
    dtype = np.dtype(object)

    def __init__(self, value=None):
        self._module_name, self._type_id = ftype_dt._type_info_from_symbol(self._sym)
        self._dt_ctype = ftype_dt._build_ctype(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )
        super().__init__()
        self.value = value

    @property
    def ftype(self):
        definition = ftype_dt._dt_definition(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )
        return definition.name

    def _shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _base_address(self) -> int:
        raise NotImplementedError

    def _index(self, index: Any) -> int:
        shape = self._shape()
        if _dt_index_debug_enabled():
            print(
                "[gfort2py.dt-index-debug] "
                f"index={index!r} "
                f"index_type={type(index).__name__} "
                f"shape={shape!r} "
                f"shape_type={type(shape).__name__}",
                file=sys.stderr,
                flush=True,
            )
        return _array_index(index, shape)

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        if value is None:
            return

        # Allow placeholders for intent(out) style calls where callers pass
        # an empty dict/list to request allocation/output-only behavior.
        if isinstance(value, dict) and len(value) == 0:
            return
        if isinstance(value, list) and len(value) == 0:
            return

        arr = np.asarray(value, dtype=object)
        shape = tuple(int(i) for i in arr.shape)
        self._ensure_shape(shape)
        for idx in np.ndindex(shape):
            self[idx].value = arr[idx]

    def _ensure_shape(self, shape: tuple[int, ...]):
        raise NotImplementedError

    def __getitem__(self, key):
        ind = self._index(key)
        address = self._base_address() + ind * ctypes.sizeof(self._dt_ctype)
        ctype_obj = self._dt_ctype.from_address(address)
        return ftype_dt.from_existing_ctype(
            self._module_name,
            self._type_id,
            ctype_obj,
            module_obj=getattr(self, "_module_obj", None),
            lib=getattr(self, "_lib", None),
        )

    def __setitem__(self, key, value):
        self[key].value = value

    def keys(self) -> list[str]:
        definition = ftype_dt._dt_definition(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        )
        return list(definition.properties.components.keys())

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    def __dir__(self):
        return list(self.keys())


class ftype_dt_explicit(ftype_dt_array):
    @property
    def ctype(self):
        return self._dt_ctype * int(np.prod(self._shape()))

    def _shape(self) -> tuple[int, ...]:
        return tuple(int(i) for i in self._sym.properties.array_spec.pyshape)

    def _base_address(self) -> int:
        return ctypes.addressof(self._ctype)

    def _ensure_shape(self, shape: tuple[int, ...]):
        if shape != self._shape():
            raise ValueError(f"Wrong shape, got {shape} expected {self._shape()}")

    def __repr__(self):
        return f"type({self.ftype})({self._shape()})"


class ftype_dt_assumed_shape(_DTModuleResolutionMixin, ftype_dt_array):
    alloc_strategy: AllocStrategy = AllocStrategy.FORTRAN

    @property
    def ctype(self):
        return _array_descriptor_ctype(self._sym.properties.array_spec.rank)

    def _shape(self) -> tuple[int, ...]:
        if self._ctype.base_addr is None:
            return tuple()
        return _shape_from_descriptor(self._ctype, self._sym.properties.array_spec.rank)

    def _base_address(self) -> int:
        if self._ctype.base_addr is None:
            raise ValueError("Array is not allocated")
        return int(self._ctype.base_addr)

    def _alloc_source(self, shape: tuple[int, ...]) -> Modulise:
        dims = ",".join([":"] * len(shape))
        shape_s = ",".join([str(i) for i in shape])
        use_module_name = self._resolved_use_module_name()
        dt_name = ftype_dt._dt_definition(
            self._module_name,
            self._type_id,
            module_obj=getattr(self, "_module_obj", None),
        ).name

        string = f"""
        subroutine alloc(x)
        use {use_module_name}, only: {dt_name}
        type({dt_name}), allocatable, dimension({dims}), intent(out) :: x
        if(allocated(x)) deallocate(x)
        allocate(x({shape_s}))
        end subroutine alloc
        """
        return Modulise(string)

    def _allocate(self, shape: tuple[int, ...]):
        code = self._alloc_source(shape)
        comp = Compile(code.as_module(), name=code.strhash())
        args = CompileArgs()
        module_file = self._resolved_module_file()
        if module_file is not None and module_file.parent:
            args.INCLUDE_FLAGS = f"-I{module_file.parent}"

        if not comp.compile(args=args):
            raise ValueError("Failed to allocate derived-type array")

        lib = comp.platform.load_library(comp.library_filename)
        sub = getattr(lib, f"__{comp.name}_MOD_alloc")
        sub(ctypes.byref(self._ctype))

        if self._ctype.base_addr is None:
            raise ValueError("Allocation failed")

    def _ensure_shape(self, shape: tuple[int, ...]):
        if len(shape) != self._sym.properties.array_spec.rank:
            raise ValueError(
                f"Wrong number of dimensions, got {len(shape)} expected {self._sym.properties.array_spec.rank}"
            )

        if self._ctype.base_addr is None or self._shape() != shape:
            self._allocate(shape)

    def __repr__(self):
        s = ",".join([str(i) for i in self._shape()])
        return f"type({self.ftype})({s})"
