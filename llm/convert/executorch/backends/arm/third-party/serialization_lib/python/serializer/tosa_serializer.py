# Copyright (c) 2020-2023, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import struct
import serializer.tosa_serializer as ts
import json
import flatbuffers
import numpy as np
from enum import IntEnum, unique
from tosa import (
    TosaGraph,
    TosaRegion,
    TosaBasicBlock,
    TosaTensor,
    TosaOperator,
    Version,
)
import tosa.DType as TosaDType
import tosa.Op as TosaOp

# Keep version number in sync with the version default value with schema/tosa.fbs
TOSA_VERSION_MAJOR = 0
TOSA_VERSION_MINOR = 80
TOSA_VERSION_PATCH = 1
TOSA_VERSION_DRAFT = False
TOSA_VERSION = [
    TOSA_VERSION_MAJOR,
    TOSA_VERSION_MINOR,
    TOSA_VERSION_PATCH,
    TOSA_VERSION_DRAFT,
]

# File identifier needs to be kept in sync with schema/tosa.fbs
TOSA_GRAPH_IDENTIFIER = b"\x54\x4F\x53\x41"

# With the way flatc generates its python types, there is no programatic way
# to get string names for the integer types.  Manually maintain a string table
# here.
DType = TosaDType.DType()
DTypeNames = [
    "UNKNOWN",
    "BOOL",
    "UINT8",
    "INT4",
    "INT8",
    "INT16",
    "INT32",
    "INT48",
    "FP32",
    "UINT16",
    "FP16",
    "BF16",
    "SHAPE",
]

ByteMask = np.uint64(0xFF)


def dtype_str_to_val(name):

    for i in range(len(DTypeNames)):
        if name.casefold() == DTypeNames[i].casefold():
            return i
    raise Exception("Unable to parse DType name {}".format(name))


class TosaSerializerUnion:
    """This class handles encapsulating and serializing union types into flatbuffers"""

    def __init__(self):

        # A tuple of the start and end functions.
        # Set by the options constructors below
        self.optFcns = None

        # The type from the tosa.Options enumeration.
        # Set by the options constructors below.
        self.utype = None

        # Each of these lists is a tuple of the add function and the
        # value being added.  Set by the options constructors below.
        self.ints = []
        self.bools = []
        self.floats = []
        self.strings = []
        self.int16vecs = []
        self.intvecs = []
        self.fpvecs = []

    def serialize(self, builder):

        # We have to build strings and vectors first
        strList = []
        intVecList = []
        fpVecList = []

        for fcn, val in self.strings:
            strList.append((fcn, builder.CreateString(val)))

        for fcn, val in self.intvecs:
            intVecList.append((fcn, TosaSerializer.serializeInt32Vec(builder, val)))

        for fcn, val in self.int16vecs:
            intVecList.append((fcn, TosaSerializer.serializeInt16Vec(builder, val)))

        for fcn, val in self.fpvecs:
            fpVecList.append((fcn, TosaSerializer.serializeFpVec(builder, val)))

        startFcn, endFcn = self.optFcns

        # Then serialize the options object from the list of primitives and
        # other serialized values
        startFcn(builder)
        for fcn, val in self.ints:
            fcn(builder, val)

        for fcn, val in self.bools:
            fcn(builder, val)

        for fcn, val in self.floats:
            fcn(builder, val)

        for fcn, val in strList:
            fcn(builder, val)

        for fcn, val in intVecList:
            fcn(builder, val)

        for fcn, val in fpVecList:
            fcn(builder, val)

        return endFcn(builder)


class TosaSerializerAttribute(TosaSerializerUnion):
    """This class handles encapsulating all of the enumerated types for attributes"""

    def __init__(self):
        super().__init__()

    def PoolAttribute(
        self,
        kernel,
        stride,
        pad,
        input_zp,
        output_zp,
        accum_dtype,
    ):
        from tosa import PoolAttribute as a, Attribute

        self.utype = Attribute.Attribute().PoolAttribute

        self.optFcns = (a.Start, a.End)
        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddKernel, kernel))
        self.intvecs.append((a.AddStride, stride))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.ints.append((a.AddAccumDtype, accum_dtype))

    def ConvAttribute(self, pad, stride, dilation, input_zp, weight_zp, local_bound):
        from tosa import ConvAttribute as a, Attribute

        self.utype = Attribute.Attribute().ConvAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))

    def TransposeConvAttribute(
        self, outpad, stride, output_shape, input_zp, weight_zp, local_bound
    ):
        from tosa import TransposeConvAttribute as a, Attribute

        self.utype = Attribute.Attribute().TransposeConvAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddOutPad, outpad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddOutputShape, output_shape))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))

    def PadAttribute(self, serializer_builder, padding, pad_const_int, pad_const_fp):
        from tosa import PadAttribute as a, Attribute

        self.utype = Attribute.Attribute().PadAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPadding, padding))
        self.ints.append((a.AddPadConstInt, pad_const_int))

        # pad_const_fp attribute serialized as uint8 vector
        pad_const_float_as_bytes = struct.pack("<f", pad_const_fp)
        serialized_pad_const_fp = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, pad_const_float_as_bytes
        )

        self.floats.append((a.AddPadConstFp, serialized_pad_const_fp))

    def AxisAttribute(self, axis):
        from tosa import AxisAttribute as a, Attribute

        self.utype = Attribute.Attribute().AxisAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReshapeAttribute(self, new_shape):
        from tosa import ReshapeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReshapeAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddNewShape, new_shape))

    def SliceAttribute(self, start, size):
        from tosa import SliceAttribute as a, Attribute

        self.utype = Attribute.Attribute().SliceAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddStart, start))
        self.intvecs.append((a.AddSize, size))

    def TileAttribute(self, multiples):
        from tosa import TileAttribute as a, Attribute

        self.utype = Attribute.Attribute().TileAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddMultiples, multiples))

    def ResizeAttribute(self, scale, offset, border, mode):
        from tosa import ResizeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ResizeAttribute
        self.optFcns = (a.Start, a.End)

        self.int16vecs.append((a.AddScale, scale))
        self.int16vecs.append((a.AddOffset, offset))
        self.int16vecs.append((a.AddBorder, border))
        self.ints.append((a.AddMode, mode))

    def ClampAttribute(self, serializer_builder, minint, maxint, minfp, maxfp):
        from tosa import ClampAttribute as a, Attribute

        self.utype = Attribute.Attribute().ClampAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddMinInt, minint))
        self.ints.append((a.AddMaxInt, maxint))

        # min/max float attributes serialized as uint8 vectors
        minfp_bytes = struct.pack("<f", minfp)
        maxfp_bytes = struct.pack("<f", maxfp)
        serialized_minfp_bytes = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, minfp_bytes
        )
        serialized_maxfp_bytes = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, maxfp_bytes
        )

        self.floats.append((a.AddMinFp, serialized_minfp_bytes))
        self.floats.append((a.AddMaxFp, serialized_maxfp_bytes))

    def RescaleAttribute(
        self,
        input_zp,
        output_zp,
        multiplier,
        shift,
        scale32,
        double_round,
        per_channel,
        input_unsigned,
        output_unsigned,
    ):
        from tosa import RescaleAttribute as a, Attribute

        self.utype = Attribute.Attribute().RescaleAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.intvecs.append((a.AddMultiplier, multiplier))
        self.intvecs.append((a.AddShift, shift))
        self.bools.append((a.AddScale32, scale32))
        self.bools.append((a.AddDoubleRound, double_round))
        self.bools.append((a.AddPerChannel, per_channel))
        self.bools.append((a.AddInputUnsigned, input_unsigned))
        self.bools.append((a.AddOutputUnsigned, output_unsigned))

    def MulAttribute(self, shift):
        from tosa import MulAttribute as a, Attribute

        self.utype = Attribute.Attribute().MulAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddShift, shift))

    def ArithmeticRightShiftAttribute(self, round):
        from tosa import ArithmeticRightShiftAttribute as a, Attribute

        self.utype = Attribute.Attribute().ArithmeticRightShiftAttribute
        self.optFcns = (
            a.Start,
            a.End,
        )

        self.bools.append((a.AddRound, round))

    def CondIfAttribute(self, then_branch, else_branch):
        from tosa import CondIfAttribute as a, Attribute

        self.utype = Attribute.Attribute().CondIfAttribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddThenBranch, then_branch))
        self.strings.append((a.AddElseBranch, else_branch))

    def WhileLoopAttribute(self, cond_branch, body_branch):
        from tosa import WhileLoopAttribute as a, Attribute

        self.utype = Attribute.Attribute().WhileLoopAttribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddCondBranch, cond_branch))
        self.strings.append((a.AddBodyBranch, body_branch))

    def TransposeAttribute(self, perms):
        from tosa import TransposeAttribute as a, Attribute

        self.utype = Attribute.Attribute().TransposeAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPerms, perms))

    def TableAttribute(self, table):
        from tosa import TableAttribute as a, Attribute

        self.utype = Attribute.Attribute().TableAttribute
        self.optFcns = (a.Start, a.End)

        self.int16vecs.append((a.AddTable, table))

    def MatMulAttribute(self, A_zp, B_zp):
        from tosa import MatMulAttribute as a, Attribute

        self.utype = Attribute.Attribute().MatMulAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAZp, A_zp))
        self.ints.append((a.AddBZp, B_zp))

    def FullyConnectedAttribute(self, input_zp, weight_zp):
        from tosa import FullyConnectedAttribute as a, Attribute

        self.utype = Attribute.Attribute().FullyConnectedAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))

    def NegateAttribute(self, input1_zp, output_zp):
        from tosa import NegateAttribute as a, Attribute

        self.utype = Attribute.Attribute().NegateAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInput1Zp, input1_zp))
        self.ints.append((a.AddOutputZp, output_zp))

    def FFTAttribute(self, inverse, local_bound):
        from tosa import FFTAttribute as a, Attribute

        self.utype = Attribute.Attribute().FFTAttribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddInverse, inverse))
        self.bools.append((a.AddLocalBound, local_bound))

    def RFFTAttribute(self, local_bound):
        from tosa import RFFTAttribute as a, Attribute

        self.utype = Attribute.Attribute().RFFTAttribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddLocalBound, local_bound))


class TosaSerializerTensor:
    def __init__(
        self,
        name,
        shape,
        dtype,
        data=None,
        placeholderFilename=None,
    ):
        self.name = name

        if isinstance(shape, np.ndarray):
            shape = shape.astype(int).tolist()
        shape = list(map(int, shape))

        self.shape = shape
        self.dtype = dtype

        if dtype == DType.FP32 or dtype == DType.BF16:
            fntype = np.float32
        elif dtype == DType.FP16:
            fntype = np.float16
        else:
            fntype = int

        if isinstance(data, np.ndarray):
            data = data.flatten().astype(fntype).tolist()
            data = list(map(fntype, data))
            self.data = data
        elif isinstance(data, list):
            data = list(map(fntype, data))
            self.data = data
        else:
            self.data = None

        # Filename for placeholder tensors.  These get generated by the test generation
        # process and are written to disk, but are considered input tensors by the
        # network so they do not appear in the TOSA serialiazation.  However, if we
        # want to form a unit test around these input tensors, we can get the filename
        # from here.
        self.placeholderFilename = placeholderFilename

    def __str__(self):
        concatString = "TosaSerializerTensor name: {} shape: {} dtype: {}".format(
            self.name,
            self.shape,
            DTypeNames[self.dtype],
        )
        return concatString

    def setDtype(self, dtype):
        self.dtype = dtype

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fb_shapes = TosaSerializer.serializeInt32Vec(builder, self.shape)
        if self.data:
            u8_data = list()
            # little endianess
            if self.dtype == DType.BOOL:
                for val in self.data:
                    val_u8 = np.uint8(val)
                    u8_data.append(val_u8)
            elif self.dtype == DType.INT4:
                in_size = len(self.data)
                out_size = (in_size + 1) // 2
                for i in range(out_size):
                    val_0 = self.data[2 * i]
                    if (2 * i + 1) < in_size:
                        val_1 = self.data[2 * i + 1]
                    else:
                        val_1 = 0
                    val_i8 = (val_0 & 0xF) | ((val_1 & 0xF) << 4)
                    val_u8 = np.uint8(val_i8)
                    u8_data.append(val_u8)
            elif self.dtype == DType.INT8:
                for val in self.data:
                    val_u8 = np.array(val).astype(dtype=np.uint8)
                    u8_data.append(val_u8)
            elif self.dtype == DType.INT16:
                for val in self.data:
                    val_u16 = np.array(val).astype(dtype=np.uint16)
                    b0 = val_u16 & ByteMask
                    b1 = (val_u16 >> np.uint16(8)) & ByteMask
                    u8_data.extend([b0, b1])
            elif self.dtype == DType.INT32:
                for val in self.data:
                    val_u32 = np.array(val).astype(dtype=np.uint32)
                    b0 = val_u32 & ByteMask
                    b1 = (val_u32 >> np.uint32(8)) & ByteMask
                    b2 = (val_u32 >> np.uint32(16)) & ByteMask
                    b3 = (val_u32 >> np.uint32(24)) & ByteMask
                    u8_data.extend([b0, b1, b2, b3])
            elif self.dtype == DType.INT48 or self.dtype == DType.SHAPE:
                for val in self.data:
                    val_u64 = np.uint64(val)
                    b0 = val_u64 & ByteMask
                    b1 = (val_u64 >> np.uint64(8)) & ByteMask
                    b2 = (val_u64 >> np.uint64(16)) & ByteMask
                    b3 = (val_u64 >> np.uint64(24)) & ByteMask
                    b4 = (val_u64 >> np.uint64(32)) & ByteMask
                    b5 = (val_u64 >> np.uint64(40)) & ByteMask
                    u8_data.extend([b0, b1, b2, b3, b4, b5])
            elif self.dtype == DType.FP16:
                np_arr = np.array(self.data, dtype=np.float16)
                u8_data.extend(np_arr.view(np.uint8))
            elif self.dtype == DType.FP32 or self.dtype == DType.BF16:
                # for val in self.data:
                #     b = struct.pack("!f", val)
                #     u8_data.extend([b[3], b[2], b[1], b[0]])
                np_arr = np.array(self.data, dtype=np.float32)
                u8_data.extend(np_arr.view(np.uint8))
            elif self.dtype == TosaDType.DType:
                # Serialize DType enum data as uint8 bytes
                for val in self.data:
                    np_arr = np.array(self.data, dtype=np.uint32)
                    u8_data.extend(np_arr.view(np.uint8))
            else:
                raise Exception(
                    "unsupported data type {}".format(DTypeNames[self.dtype])
                )
            fb_data = TosaSerializer.serializeUint8Vec(builder, u8_data)

        TosaTensor.Start(builder)
        TosaTensor.AddName(builder, fb_name)
        TosaTensor.AddShape(builder, fb_shapes)
        TosaTensor.AddType(builder, self.dtype)
        if self.data:
            TosaTensor.AddData(builder, fb_data)

        return TosaTensor.End(builder)


class TosaSerializerOperator:
    def __init__(self, op, inputs, outputs, attributes=None):
        self.op = op
        self.attributes = attributes
        self.inputs = TosaSerializer.toList(inputs)
        self.outputs = TosaSerializer.toList(outputs)

    def __str__(self):
        concatString = "Op {}\n----\n".format(self.op)

        for i in self.inputs:
            concatString = concatString + "  Input:  {}\n".format(i)
        for o in self.outputs:
            concatString = concatString + "  Output: {}\n".format(o)

        return concatString

    def serialize(self, builder):
        fb_inputs = TosaSerializer.serializeStrVec(
            builder, self.inputs, TosaOperator.StartInputsVector
        )
        fb_outputs = TosaSerializer.serializeStrVec(
            builder, self.outputs, TosaOperator.StartOutputsVector
        )
        # Need to serialize attributes enums still
        if self.attributes is not None:
            fb_attributes = self.attributes.serialize(builder)

        TosaOperator.Start(builder)
        TosaOperator.AddOp(builder, self.op)
        TosaOperator.AddInputs(builder, fb_inputs)
        TosaOperator.AddOutputs(builder, fb_outputs)
        if self.attributes is not None:
            TosaOperator.AddAttributeType(builder, self.attributes.utype)
            TosaOperator.AddAttribute(builder, fb_attributes)

        return TosaOperator.End(builder)


class TosaSerializerBasicBlock:
    def __init__(self, name):
        self.name = name
        self.operators = []

        # Dict assures uniqueness, but allows us to look up by name
        self.tensors = dict()

        self.inputs = []
        self.outputs = []

    def addTensor(
        self,
        name,
        shape,
        dtype,
        data=None,
        placeholderFilename=None,
    ):
        if name not in self.tensors:
            self.tensors[name] = TosaSerializerTensor(
                name, shape, dtype, data, placeholderFilename
            )

        return self.tensors[name]

    def addInput(self, name):
        self.inputs.append(name)

    def addOutput(self, name):
        self.outputs.append(name)

    def addOperator(self, op, inputs, outputs, attributes=None):
        self.operators.append(TosaSerializerOperator(op, inputs, outputs, attributes))

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fbv_inputs = TosaSerializer.serializeStrVec(
            builder, list(self.inputs), TosaBasicBlock.StartInputsVector
        )
        fbv_outputs = TosaSerializer.serializeStrVec(
            builder, list(self.outputs), TosaBasicBlock.StartOutputsVector
        )
        fbv_tensors = TosaSerializer.serializeObjVec(
            builder,
            list(self.tensors.values()),
            TosaBasicBlock.StartTensorsVector,
        )
        fbv_operators = TosaSerializer.serializeObjVec(
            builder, self.operators, TosaBasicBlock.StartOperatorsVector
        )

        TosaBasicBlock.Start(builder)
        TosaBasicBlock.AddName(builder, fb_name)
        TosaBasicBlock.AddInputs(builder, fbv_inputs)
        TosaBasicBlock.AddOutputs(builder, fbv_outputs)
        TosaBasicBlock.AddTensors(builder, fbv_tensors)
        TosaBasicBlock.AddOperators(builder, fbv_operators)
        return TosaBasicBlock.End(builder)


# How CONSTs are treated in the flatbuffer
@unique
class ConstMode(IntEnum):
    EMBED = 0
    EMBED_DUMP = 1
    INPUTS = 2


class TosaSerializerRegion:
    def __init__(self, name, pathPrefix, constMode=ConstMode.EMBED):
        self.name = name
        self.basicBlocks = []
        self.currInputIdx = 0
        self.currConstIdx = 0
        self.currLayerIdx = 1
        self.currResultIdx = 0
        self.pathPrefix = pathPrefix
        self.constMode = constMode

    def addBasicBlock(self, name):
        self.currBasicBlock = TosaSerializerBasicBlock(name)
        self.basicBlocks.append(self.currBasicBlock)

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fbv_basicBlocks = TosaSerializer.serializeObjVec(
            builder, self.basicBlocks, TosaRegion.StartBlocksVector
        )

        TosaRegion.Start(builder)
        TosaRegion.AddName(builder, fb_name)
        TosaRegion.AddBlocks(builder, fbv_basicBlocks)
        return TosaRegion.End(builder)

    def addPlaceholder(self, shape, dtype, vals):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "input-{}".format(self.currInputIdx)
        filename = "{}.npy".format(name)
        self.currInputIdx = self.currInputIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None, filename)
        # This is always an input to the block
        self.currBasicBlock.addInput(name)

        if vals is not None:
            np.save(os.path.join(self.pathPrefix, filename), vals, False)

        return tens

    def addConst(self, shape, dtype, vals, name=None):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        if name is None:
            name = "const-{}".format(self.currInputIdx)
            self.currInputIdx = self.currInputIdx + 1

        if self.constMode == ConstMode.INPUTS:
            # Save const as input file
            filename = "{}.npy".format(name)
            tensor_vals = None
            self.currBasicBlock.addInput(name)
        else:
            # Embed const in flatbuffer
            filename = None
            tensor_vals = vals

        tens = self.currBasicBlock.addTensor(name, shape, dtype, tensor_vals, filename)
        # Add the operator now
        self.currBasicBlock.addOperator(TosaOp.Op().CONST, [], name)

        # Save the const data to file for debug or as input files
        if vals is not None and self.constMode in [
            ConstMode.EMBED_DUMP,
            ConstMode.INPUTS,
        ]:
            filename = "{}.npy".format(name)
            np.save(os.path.join(self.pathPrefix, filename), vals, False)

        return tens

    def addIntermediate(self, shape, dtype):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "layer-{}".format(self.currLayerIdx)
        self.currLayerIdx = self.currLayerIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None)

        return tens

    def addInputTensor(self, tensor):
        self.currBasicBlock.addTensor(
            tensor.name,
            tensor.shape,
            tensor.dtype,
            tensor.data,
            tensor.placeholderFilename,
        )
        self.currBasicBlock.addInput(tensor.name)

    def addOutputTensor(self, tensor):
        self.currBasicBlock.addOutput(tensor.name)

    def addOutput(self, shape, dtype):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "result-{}".format(self.currResultIdx)
        self.currResultIdx = self.currResultIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None)
        self.currBasicBlock.addOutput(name)
        return tens

    def addOperator(self, op, inputs, outputs, attributes=None):
        if op == TosaOp.Op().CONST:
            raise Exception("Use addConstTensor() to add CONST ops")

        return self.currBasicBlock.addOperator(
            op,
            inputs,
            outputs,
            attributes,
        )


@unique
class TensorDir(IntEnum):
    PLACEHOLDER = 0
    CONST = 1
    INTERMEDIATE = 2
    RESULT = 3


class TosaSerializer:
    def __init__(self, pathPrefix, constMode=ConstMode.EMBED):
        self.builder = flatbuffers.Builder(0)

        # Enables inspection of constant data outside of graph
        self.constMode = constMode

        self.regions = []
        self.startRegion("main", pathPrefix)

        self.currRegion.addBasicBlock("main")

        # Is this an illegal test that is expected to fail?
        self.expectedReturnCode = 0
        self.expectedFailure = False
        self.expectedFailureDesc = ""

    def __str__(self):
        concatString = ""
        for region in self.regions:
            concatString = concatString + str(region)
        return concatString

    def addPlaceholder(self, shape, dtype, vals):
        return self.currRegion.addPlaceholder(shape, dtype, vals)

    def addConst(self, shape, dtype, vals, name=None):
        return self.currRegion.addConst(shape, dtype, vals, name)

    def addIntermediate(self, shape, dtype):
        return self.currRegion.addIntermediate(shape, dtype)

    def addInputTensor(self, tensor):
        self.currRegion.addInputTensor(tensor)

    def addOutputTensor(self, tensor):
        self.currRegion.addOutputTensor(tensor)

    def addOutput(self, shape, dtype):
        return self.currRegion.addOutput(shape, dtype)

    def addOperator(self, op, inputs, outputs, attributes=None):
        return self.currRegion.addOperator(op, inputs, outputs, attributes)

    def addBasicBlock(self, name):
        self.currRegion.addBasicBlock(name)

    def setExpectedReturnCode(self, val, fail, desc=""):

        self.expectedReturnCode = val
        self.expectedFailureDesc = desc
        self.expectedFailure = fail

    def serialize(self):

        builder = self.builder

        Version.Start(builder)
        Version.Add_Major(builder, TOSA_VERSION[0])
        Version.Add_Minor(builder, TOSA_VERSION[1])
        Version.Add_Patch(builder, TOSA_VERSION[2])
        Version.Add_Draft(builder, TOSA_VERSION[3])
        version = Version.End(builder)

        fbv_region = TosaSerializer.serializeObjVec(
            builder, self.regions, TosaGraph.StartRegionsVector
        )

        TosaGraph.Start(builder)
        TosaGraph.AddVersion(builder, version)
        TosaGraph.AddRegions(builder, fbv_region)
        graph = TosaGraph.End(builder)

        self.builder.Finish(graph, TOSA_GRAPH_IDENTIFIER)
        return self.builder.Output()

    def writeJson(self, tosa_filename):
        """Write a json test file so that it is fairly easy to pick up the test
        and generate commands for third party tool"""
        test_desc = dict()

        test_desc["tosa_file"] = tosa_filename
        ifm_name = []
        ifm_file = []
        ofm_name = []
        ofm_file = []

        for region in self.regions:
            for block in region.basicBlocks:
                if block and block.name == "main":
                    for i in block.inputs:
                        ifm_name.append(i)
                        ifm_file.append(block.tensors[i].placeholderFilename)
                    for o in block.outputs:
                        ofm_name.append(o)
                        # Make up an OFM filename here.  One isn't generated until the
                        # reference tool is run, so any name is a good name
                        ofm_file.append("ref-{}.npy".format(o))

        test_desc["ifm_name"] = ifm_name
        test_desc["ifm_file"] = ifm_file
        test_desc["ofm_name"] = ofm_name
        test_desc["ofm_file"] = ofm_file
        test_desc["expected_return_code"] = self.expectedReturnCode
        test_desc["expected_failure"] = self.expectedFailure
        if self.expectedFailureDesc:
            test_desc["expected_failure_desc"] = self.expectedFailureDesc

        return json.dumps(test_desc, indent="  ")

    def startRegion(self, name, pathPrefix):
        self.currRegion = TosaSerializerRegion(name, pathPrefix, self.constMode)
        self.regions.append(self.currRegion)

    @staticmethod
    def serializeStrVec(builder, vec, start_fcn):
        fb_strs = [builder.CreateString(i) for i in vec]
        start_fcn(builder, len(fb_strs))
        for s in fb_strs[::-1]:
            builder.PrependUOffsetTRelative(s)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeUint8Vec(builder, vec):
        builder.StartVector(1, len(vec), 8)
        for v in vec[::-1]:
            builder.PrependUint8(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeInt16Vec(builder, vec):
        builder.StartVector(2, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependInt16(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeInt32Vec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependInt32(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeFpVec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependFloat32(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeObjVec(builder, vec, start_fcn):
        serialized_vec = []
        for v in vec[::-1]:
            serialized_vec.append(v.serialize(builder))

        start_fcn(builder, len(vec))
        for v in serialized_vec:
            builder.PrependUOffsetTRelative(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def toList(val):
        if isinstance(val, list):
            return val
        else:
            return [val]
