"""
Python IO for nemo snapshot
Author: Zhaozhou Li (lizz.astro@gmail.com)
Example:
    file = '../data/N1e6_d0.22_g0.16_0.00_f0.00.snap'
    snap = open_nemo(file, file_meta=False)
    print(snap.History.data, end='\n\n')
    print(snap['SnapShot/0/Particles'], end='\n\n')
    print(snap['SnapShot/0/Parameters/Time'].data)
    print(snap.SnapShot[0].Parameters.Time.data)
    print(snap.SnapShot[-1].Parameters.Time.data)
    t = snap['SnapShot/0/Parameters/Time'].data
    x = snap['SnapShot/0/Particles/Position'].data
    v = snap['SnapShot/0/Particles/Velocity'].data
    m = snap['SnapShot/0/Particles/Mass'].data
    # ------------
    import unsio.input as uns_in
    fp_uns = uns_in.CUNS_IN(file, float32=True)
    fp_uns.nextFrame("mxv")
    t2 = fp_uns.getData('time')[1]
    x2 = fp_uns.getData("all", "pos")[1].reshape(-1, 3)
    v2 = fp_uns.getData("all", "vel")[1].reshape(-1, 3)
    m2 = fp_uns.getData('all', 'mass')[1]
    assert (x==x2).all()
    assert (v==v2).all()
History:
    02 Feb 2022:
        Improve the speed and IO
        Add meta file
    25 Aug 2021: 
        Initilize
"""

# reference
# https://github.com/teuben/nemo/blob/master/inc/filestruct.h
# https://github.com/teuben/nemo/blob/master/src/kernel/io/filesecret.h
# https://github.com/teuben/nemo/blob/master/src/kernel/io/filesecret.c


"""
data model example
{
    "_type": "i8",
    "_shape": [2, 5],
    "_data": null,
    "_data_offset": 0
    "_item_offset": 0
}
"""

__all__ = ['open_nemo']

import numpy as np
import struct
import json
from pathlib import Path


SingMagic = (0o11 << 8) + 0o222
PlurMagic = (0o13 << 8) + 0o222
xstrNULL = b'\x00'

TypeDict = {
    'a': 'i1',      # AnyType    - anything at all
    'c': 'u1',      # CharType   - printable chars  [use u1 as proxy]
    'b': 'S1',      # ByteType   - unprintable chars
    's': 'i2',      # ShortType  - short integers
    'i': 'i4',      # IntType    - standard integers
    'l': 'i8',      # LongType   - long integers
    'h': 'f2',      # HalfpType  - half precision floating
    'f': 'f4',      # FloatType  - short floating
    'd': 'f8',      # DoubleType - long floating
    '(': 'set',     # SetType    - begin compound item
    ')': 'tes',     # TesType    - end of compound item
}

MaxTagLen = 65      # max tag length, limited for simplicity
MaxVecDim = 9       # max num of vec dim, limited for safety
MaxSetLen = 65      # max num of components in compound item


def getxstr(file, dtype=None):
    """
    xstr is a sequence ends with \x00
    dtype:
        None for string type
    """
    if dtype is None:
        nbyte = 1
    else:
        dtype = np.dtype(dtype)
        nbyte = dtype.itemsize

    end = xstrNULL * nbyte
    buf = []
    while True:
        s = file.read(nbyte)
        if s == end:
            break
        buf.append(s)
    buf = b''.join(buf)

    if dtype is None:
        return buf.decode()
    else:
        return np.frombuffer(buf, dtype).tolist()


def gettype(file):
    "equiv to getxstr(file)"
    return file.read(2)[:1].decode()


def get_item(file, load=False):
    item_offset = file.tell()

    buff = file.read(4)
    if len(buff) == 0:
        raise EOFError('File end')
    elif len(buff) != 4:
        raise ValueError('Unexpected file end')
    elif buff[3:4] != xstrNULL:
        raise ValueError('Unexpected file structure')

    magic = struct.unpack('h', buff[0:2])[0]  # int16
    if magic != SingMagic and magic != PlurMagic:
        raise ValueError(f"Bad magic number: {magic}")

    type = buff[2:3].decode()
    type = TypeDict[type]

    # magic = np.fromfile(file, 'i2', 1)
    # type = gettype(file)

    if type == 'set':
        tag = getxstr(file)
        return tag, dict(_type=type, _item_offset=item_offset)

    elif type == 'tes':
        return None, dict(_type=type)  # end of group

    else:
        tag = getxstr(file)

        if magic == SingMagic:
            shape = []
            size = 1
        else:
            shape = getxstr(file, 'i4')
            size = np.prod(shape)

        data_offset = file.tell()

        dtype = np.dtype(type)
        if load or len(shape) == 0:
            data = np.fromfile(file, dtype, size).reshape(shape)
            if type == 'u1':
                data = data.tobytes().strip(b'\x00').decode()
            elif len(shape) == 0:
                data = data.item()  # scalar
        else:
            file.seek(size * dtype.itemsize, 1)
            data = None

        return tag, dict(_type=type, _shape=shape,
                         _item_offset=item_offset,
                         _data_offset=data_offset,
                         _data=data)


def add_item(item, subtag, subitem):
    "put repetition into a list"
    if subtag in item:
        if item[subtag]['_type'] != 'list':
            item[subtag] = {'_type': 'list',
                            '_shape': [1],
                            '_item_offset': item[subtag]['_item_offset'],
                            '0': item[subtag]}  # first repetition
        n = item[subtag]['_shape'][0]
        item[subtag][str(n)] = subitem
        item[subtag]['_shape'][0] = n + 1
    else:
        item[subtag] = subitem


def read_item(file, load=False):
    tag, item = get_item(file, load=load)

    if item['_type'] == 'set':
        while True:
            subtag, subitem = read_item(file, load=load)
            if subitem['_type'] == 'tes':
                break
            add_item(item, subtag, subitem)

    return tag, item


def read_file(file, load=False):
    if file is not None and not hasattr(file, 'tell'):
        file = open(file, 'rb')

    item = dict(_type='set', _item_offset=0)
    while True:
        try:
            subtag, subitem = read_item(file, load=load)
            add_item(item, subtag, subitem)
        except EOFError:
            break

    return item


class ItemView:
    """
    item = read_file(file)
    snap = itemview(item)
    """

    def __init__(self, item, file=None, cache=False):
        if file is not None and not hasattr(file, 'tell'):
            file = open(file, 'rb')

        for key, val in item.items():
            if isinstance(val, dict):
                setattr(self, key, ItemView(val, file=file, cache=cache))
            else:
                setattr(self, key, val)
            self._file = file
            self._cache = cache
            # self._item = item

    def __repr__(self):
        if self._type in ['set', 'list']:
            desc_list = []
            for key, val in self.__dict__.items():
                if key.startswith('_'):
                    continue
                elif hasattr(val, '_shape'):
                    if len(val._shape) == 0:
                        desc = f"{val._type} {key}: {val._data}"
                    else:
                        desc = f"{val._type} {key}{val._shape}"
                else:
                    desc = f"{val._type} {key}"
                desc_list.append(desc)
            return "\n".join(desc_list)
        else:
            if len(self._shape) == 0:
                return f"{self._type}: {self._data}"
            else:
                return f"{self._type}{self._shape}"

    def __getitem__(self, key):
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            elif '/' in key:
                keys = key.strip('/').split('/')
                val = self
                for key in keys:
                    val = val[key]
                return val
        else:
            if self._type == 'list':
                if isinstance(key, (int, np.integer)):
                    if key >= 0:
                        key_str = str(key)
                    else:
                        key_str = str(key + self._shape[0])
                    if hasattr(self, key_str):
                        return getattr(self, key_str)
        raise KeyError(key)

    def __getattr__(self, key):
        if key == 'data' and self._type not in ['set', 'list']:
            return self._load_data()
        else:
            raise AttributeError(f"'itemview' object has no attribute '{key}'")

    def _load_data(self):
        if self._data is None:
            file = self._file
            dtype = self._type
            shape = self._shape
            offset = self._data_offset

            if shape:
                size = np.prod(shape)
            else:
                size = 1

            file.seek(offset)
            data = np.fromfile(file, dtype, size).reshape(shape)
            if dtype == 'u1':
                data = data.tobytes().strip(b'\x00').decode()

            if self._cache:
                self._data = data

            return data
        else:
            return self._data


def open_nemo(file_snap, file_meta=False, rebuild_meta=False, cache=False):
    """
    file_snap: str
        Path of snapshot file.
    file_meta: bool, str
        False for disable using meta.
        True for using default path, a string for path of meta file.
    rebuild_meta: bool
        Check if the meta file is consistent with the snap file,
        update the meta file when necessary.
    cache: bool
        If store the data in the itemview object
    """
    file = Path(file_snap).expanduser().resolve()
    assert file.is_file()

    if file_meta:
        try:
            file_meta = Path(file_meta).expanduser().resolve()
        except Exception:
            file_meta = file.parent / 'meta' / (file.name + '.meta')
            file_meta = Path(file_meta).expanduser().resolve()

        assert file != file_meta

        if file_meta.is_file():
            item = json.load(open(file_meta, 'r'))

            if rebuild_meta:
                item_true = read_file(file, load=False)
                if item != json.loads(json.dumps(item_true)):
                    item = item_true
                    json.dump(item, open(file_meta, 'w'), indent=2)
        else:
            item = read_file(file, load=False)

            file_meta.parent.mkdir(exist_ok=True)
            json.dump(item, open(file_meta, 'w'), indent=2)
    else:
        item = read_file(file, load=False)

    return ItemView(item, file=file, cache=cache)
