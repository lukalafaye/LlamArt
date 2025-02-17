# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FFTAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FFTAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFFTAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def FFTAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # FFTAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FFTAttribute
    def Inverse(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # FFTAttribute
    def LocalBound(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def FFTAttributeStart(builder):
    builder.StartObject(2)

def Start(builder):
    FFTAttributeStart(builder)

def FFTAttributeAddInverse(builder, inverse):
    builder.PrependBoolSlot(0, inverse, 0)

def AddInverse(builder, inverse):
    FFTAttributeAddInverse(builder, inverse)

def FFTAttributeAddLocalBound(builder, localBound):
    builder.PrependBoolSlot(1, localBound, 0)

def AddLocalBound(builder, localBound):
    FFTAttributeAddLocalBound(builder, localBound)

def FFTAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return FFTAttributeEnd(builder)
